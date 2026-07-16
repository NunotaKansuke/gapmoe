"""FlowJAX event-kernel backend combined with Galactic source densities."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from jax.scipy.special import logsumexp

from gapmoe.flow_source_grid import FlowSourceDistance


SOURCE_GROUP_BY_COMPONENT = np.asarray((0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4), dtype=int)
SOURCE_GROUP_MATRIX = jnp.asarray(
    [[float(component_group == group) for component_group in SOURCE_GROUP_BY_COMPONENT] for group in range(5)]
)


@dataclass(frozen=True)
class FlowConfig:
    dim: int
    cond_dim: int
    flow_layers: int
    nn_width: int
    nn_depth: int
    seed: int

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> "FlowConfig":
        return cls(**{name: int(value) for name, value in values.items()})


@dataclass(frozen=True)
class StandardTransform:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> "StandardTransform":
        return cls(
            mean=tuple(float(value) for value in values["mean"]),
            std=tuple(float(value) for value in values["std"]),
        )

    def apply(self, values: Any):
        return (jnp.asarray(values) - jnp.asarray(self.mean)) / jnp.asarray(self.std)


@dataclass(frozen=True)
class ResidualTransform(StandardTransform):
    def to_unconstrained(self, values: Any, condition: Any):
        values = jnp.asarray(values)
        ml, dl, mu_e, mu_n = jnp.moveaxis(values, -1, 0)
        ds = jnp.asarray(condition)[..., 2]
        ratio = dl / ds
        raw = jnp.stack(
            (jnp.log(ml), jnp.log(ratio) - jnp.log1p(-ratio), mu_e, mu_n),
            axis=-1,
        )
        return self.apply(raw)

    def from_unconstrained(self, values: Any, condition: Any):
        raw = jnp.asarray(values) * jnp.asarray(self.std) + jnp.asarray(self.mean)
        ds = jnp.asarray(condition)[..., 2]
        # In the default float32 JAX mode, a very large positive logit can
        # round ``DS * sigmoid(logit)`` back to DS. Keep the sampled lens
        # distance strictly inside its physical support so downstream
        # calculations of 1/DL - 1/DS cannot acquire a negative radicand.
        dl = jnp.minimum(
            ds * jax.nn.sigmoid(raw[..., 1]),
            jnp.nextafter(ds, jnp.zeros_like(ds)),
        )
        return jnp.stack(
            (jnp.exp(raw[..., 0]), dl, raw[..., 2], raw[..., 3]),
            axis=-1,
        )

    def log_abs_det_unconstrained_wrt_physical(self, values: Any, condition: Any):
        values = jnp.asarray(values)
        ds = jnp.asarray(condition)[..., 2]
        ml, dl = values[..., 0], values[..., 1]
        return (
            jnp.log(ds)
            - jnp.log(ml)
            - jnp.log(dl)
            - jnp.log(ds - dl)
            - jnp.sum(jnp.log(jnp.asarray(self.std)))
        )


class EventKernelFlow:
    """Trained p(ML, DL, mu_E, mu_N | l, b, DS, source_group)."""

    def __init__(
        self,
        flow: Any,
        transform: ResidualTransform,
        condition_transform: StandardTransform,
        config: FlowConfig,
        group_overrides: dict[int, "EventKernelFlow"] | None = None,
    ) -> None:
        self.flow = flow
        self.transform = transform
        self.condition_transform = condition_transform
        self.config = config
        self.group_overrides = group_overrides or {}

    @classmethod
    def load(cls, directory: str | Path) -> "EventKernelFlow":
        directory = Path(directory)
        model = cls._load_single(directory)
        override_path = directory / "group_overrides.json"
        if not override_path.exists():
            return model
        payload = json.loads(override_path.read_text())
        loaded: dict[str, EventKernelFlow] = {}
        overrides = {}
        for group, relative_path in payload.items():
            if relative_path not in loaded:
                loaded[relative_path] = cls._load_single(directory / relative_path)
            overrides[int(group)] = loaded[relative_path]
        if any(group < 0 or group >= 5 for group in overrides):
            raise ValueError("event-kernel group overrides must be in [0, 4]")
        return cls(
            model.flow,
            model.transform,
            model.condition_transform,
            model.config,
            group_overrides=overrides,
        )

    @classmethod
    def _load_single(cls, directory: Path) -> "EventKernelFlow":
        metadata = json.loads((directory / "config.json").read_text())
        config = FlowConfig.from_json(metadata["config"])
        transform = ResidualTransform.from_json(metadata["transform"])
        condition_transform = StandardTransform.from_json(metadata["condition_transform"])
        template = masked_autoregressive_flow(
            jax.random.key(config.seed),
            base_dist=Normal(jnp.zeros(config.dim)),
            cond_dim=config.cond_dim,
            flow_layers=config.flow_layers,
            nn_width=config.nn_width,
            nn_depth=config.nn_depth,
        )
        flow = eqx.tree_deserialise_leaves(directory / "flow.eqx", template)
        return cls(flow, transform, condition_transform, config)

    def log_density(self, values: Any, condition: Any):
        logp = self._log_density_single(values, condition)
        condition = jnp.asarray(condition)
        for groups, model in self._unique_group_overrides():
            override = model._log_density_single(values, condition)
            selected = jnp.any(
                jnp.stack([condition[..., 3 + group] > 0.5 for group in groups]),
                axis=0,
            )
            logp = jnp.where(selected, override, logp)
        return logp

    def _log_density_single(self, values: Any, condition: Any):
        values = jnp.asarray(values)
        condition = jnp.asarray(condition)
        unconstrained = self.transform.to_unconstrained(values, condition)
        logp = self.flow.log_prob(
            unconstrained,
            condition=self.condition_transform.apply(condition),
        )
        logp = logp + self.transform.log_abs_det_unconstrained_wrt_physical(values, condition)
        valid = (
            (values[..., 0] > 0.0)
            & (values[..., 1] > 0.0)
            & (condition[..., 2] > values[..., 1])
        )
        return jnp.where(valid & jnp.isfinite(logp), logp, -jnp.inf)

    def sample(self, key: Any, condition: Any):
        sample = self._sample_single(key, condition)
        condition = jnp.asarray(condition)
        for groups, model in self._unique_group_overrides():
            override = model._sample_single(key, condition)
            selected = jnp.any(
                jnp.stack([condition[..., 3 + group] > 0.5 for group in groups]),
                axis=0,
            )
            sample = jnp.where(selected[..., None], override, sample)
        return sample

    def _sample_single(self, key: Any, condition: Any):
        condition = jnp.asarray(condition)
        unconstrained = self.flow.sample(
            key,
            condition=self.condition_transform.apply(condition),
        )
        return self.transform.from_unconstrained(unconstrained, condition)

    def _unique_group_overrides(self):
        unique: list[tuple[list[int], EventKernelFlow]] = []
        for group, model in self.group_overrides.items():
            for groups, known_model in unique:
                if model is known_model:
                    groups.append(group)
                    break
            else:
                unique.append(([group], model))
        return unique


@dataclass(frozen=True)
class FlowDensity:
    """Five-dimensional density assembled from a Flow kernel and source grid."""

    # Training removes theta_E * mu_rel, while the conditional kernel keeps
    # genulens's DL**2 lens-area factor. EventPrior5D uses this marker to
    # avoid applying that factor a second time.
    event_rate_factor_includes_lens_area = True

    kernel: EventKernelFlow
    distance: FlowSourceDistance
    l_deg: float
    b_deg: float
    event_rate_included: bool = False

    def with_source_evidence(self, evidence: Any) -> "FlowDensity":
        distance_pc = np.asarray(self.distance.distance_pc)
        weights = evidence.evidence_on(distance_pc, self.distance.source_by_component.shape[1])
        selected = np.asarray(self.distance.source_by_component) * weights
        source_norm = float(np.trapezoid(np.sum(selected, axis=1), distance_pc / 1000.0))
        return replace(
            self,
            distance=FlowSourceDistance(
                distance_pc=self.distance.distance_pc,
                source_by_component=jnp.asarray(selected),
                source_norm=source_norm,
            ),
        )

    def log_density(self, ml: Any, dl: Any, ds: Any, mu_n: Any, mu_e: Any):
        component_density = self.distance.source_component_values(ds) / self.distance.source_norm
        return self._combine_kernel(ml, dl, ds, mu_n, mu_e, component_density)

    def log_cmd_joint_density(
        self,
        ml: Any,
        dl: Any,
        ds: Any,
        mu_n: Any,
        mu_e: Any,
        *,
        cmd_prior: Any,
        reference_magnitude: Any,
        color: Any,
        magnitude_offsets: Any,
        source_component_factor: Any = None,
    ):
        photometric = cmd_prior.density_all_components(reference_magnitude, color, magnitude_offsets)
        if source_component_factor is not None:
            photometric = photometric * jnp.asarray(source_component_factor)
        component_density = self.distance.source_component_values(ds) * photometric / self.distance.source_norm
        return self._combine_kernel(ml, dl, ds, mu_n, mu_e, component_density)

    def sample_kernel(self, key: Any, ds: Any, source_group: int):
        if not 0 <= int(source_group) < 5:
            raise ValueError("source_group must be in [0, 4]")
        return self._sample_kernel(key, ds, source_group)

    def sample_source_group(self, key: Any, component_weights: Any | None = None):
        """Sample ``(DS, source_group)`` from a component-resolved source measure."""

        components = self.distance.source_by_component if component_weights is None else jnp.asarray(component_weights)
        if components.shape != self.distance.source_by_component.shape:
            raise ValueError("component_weights must have the source-distance grid shape")
        group_weights = components @ SOURCE_GROUP_MATRIX.T
        distance_kpc = self.distance.distance_pc / 1000.0
        widths = jnp.concatenate(
            (
                0.5 * (distance_kpc[1:2] - distance_kpc[:1]),
                0.5 * (distance_kpc[2:] - distance_kpc[:-2]),
                0.5 * (distance_kpc[-1:] - distance_kpc[-2:-1]),
            )
        )
        weights = jnp.maximum(group_weights, 0.0) * widths[:, None]
        flat_index = jax.random.categorical(key, jnp.where(weights.ravel() > 0.0, jnp.log(weights.ravel()), -jnp.inf))
        distance_index, source_group = jnp.divmod(flat_index, 5)
        return distance_kpc[distance_index], source_group

    def _sample_kernel(self, key: Any, ds: Any, source_group: Any):
        condition = jnp.concatenate(
            (
                jnp.stack((jnp.asarray(self.l_deg), jnp.asarray(self.b_deg), jnp.asarray(ds))),
                jax.nn.one_hot(source_group, 5),
            )
        )
        sample = self.kernel.sample(key, condition)
        return jnp.asarray((sample[0], sample[1], ds, sample[3], sample[2]))

    def _combine_kernel(self, ml: Any, dl: Any, ds: Any, mu_n: Any, mu_e: Any, component_density: Any):
        group_density = SOURCE_GROUP_MATRIX @ jnp.asarray(component_density)
        conditions = jnp.column_stack(
            (
                jnp.full(5, self.l_deg),
                jnp.full(5, self.b_deg),
                jnp.full(5, ds),
                jnp.eye(5),
            )
        )
        values = jnp.broadcast_to(jnp.asarray((ml, dl, mu_e, mu_n)), (5, 4))
        kernel_logp = self.kernel.log_density(values, conditions)
        terms = jnp.where(group_density > 0.0, jnp.log(group_density) + kernel_logp, -jnp.inf)
        return logsumexp(terms)
