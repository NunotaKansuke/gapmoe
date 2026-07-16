"""Local FlowJAX model factorized from the source-selection model.

The flow learns p(ML, DL, mu_E, mu_N | l, b, DS, source_group).  A histogram
or another source model supplies p(DS, source_group | l, b, selection), so a
new magnitude/color/theta-star selection does not require retraining the flow.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow

REST_FEATURE_NAMES = ("ML", "DL", "mu_E", "mu_N")
CONDITION_NAMES = ("l", "b", "DS", "source_group_thin", "source_group_thick", "source_group_bulge", "source_group_NSD", "source_group_halo")


@dataclass(frozen=True)
class ConditionTransform:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    names: tuple[str, ...] = CONDITION_NAMES

    @classmethod
    def fit(
        cls, condition: np.ndarray, *, names: tuple[str, ...] = CONDITION_NAMES
    ) -> "ConditionTransform":
        condition = _as_condition_array(condition)
        std = np.std(condition, axis=0)
        return cls(
            mean=tuple(float(value) for value in np.mean(condition, axis=0)),
            std=tuple(float(value) for value in np.where(std > 0.0, std, 1.0)),
            names=tuple(names),
        )

    def to_standard_np(self, condition: np.ndarray) -> np.ndarray:
        condition = _as_condition_array(condition)
        return (condition - np.asarray(self.mean)) / np.asarray(self.std)

    def to_standard(self, condition: Any) -> jax.Array:
        return (jnp.asarray(condition) - jnp.asarray(self.mean)) / jnp.asarray(self.std)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ConditionTransform":
        return cls(
            mean=tuple(float(value) for value in data["mean"]),
            std=tuple(float(value) for value in data["std"]),
            names=tuple(str(value) for value in data.get("names", CONDITION_NAMES)),
        )


@dataclass(frozen=True)
class FlowConfig:
    dim: int = 4
    cond_dim: int = 8
    flow_layers: int = 10
    nn_width: int = 96
    nn_depth: int = 2
    seed: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FlowConfig":
        return cls(**{key: int(value) for key, value in data.items()})


@dataclass(frozen=True)
class ResidualTransform:
    mean: tuple[float, float, float, float]
    std: tuple[float, float, float, float]

    @classmethod
    def fit(cls, rest: np.ndarray, condition: np.ndarray) -> "ResidualTransform":
        initial = cls(mean=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0))
        raw = initial._raw_np(rest, condition)
        std = np.std(raw, axis=0)
        return cls(tuple(np.mean(raw, axis=0)), tuple(np.where(std > 0.0, std, 1.0)))

    @staticmethod
    def valid_mask_np(rest: np.ndarray, condition: np.ndarray) -> np.ndarray:
        ml, dl, mu_e, mu_n = np.asarray(rest, dtype=float).T
        ds = np.asarray(condition, dtype=float)[:, 2]
        return np.isfinite(rest).all(axis=1) & np.isfinite(condition).all(axis=1) & (ml > 0.0) & (dl > 0.0) & (ds > dl)

    def _raw_np(self, rest: np.ndarray, condition: np.ndarray) -> np.ndarray:
        if not np.all(ResidualTransform.valid_mask_np(rest, condition)):
            raise ValueError("invalid residual-flow samples; require ML>0 and 0<DL<DS")
        ml, dl, mu_e, mu_n = np.asarray(rest, dtype=float).T
        ds = np.asarray(condition, dtype=float)[:, 2]
        ratio = dl / ds
        return np.column_stack([np.log(ml), np.log(ratio) - np.log1p(-ratio), mu_e, mu_n])

    def to_unconstrained_np(self, rest: np.ndarray, condition: np.ndarray) -> np.ndarray:
        return (self._raw_np(rest, condition) - np.asarray(self.mean)) / np.asarray(self.std)

    def to_unconstrained(self, rest: Any, condition: Any) -> jax.Array:
        x = jnp.asarray(rest)
        cond = jnp.asarray(condition)
        ml, dl, mu_e, mu_n = jnp.moveaxis(x, -1, 0)
        ds = cond[..., 2]
        ratio = dl / ds
        raw = jnp.stack([jnp.log(ml), jnp.log(ratio) - jnp.log1p(-ratio), mu_e, mu_n], axis=-1)
        return (raw - jnp.asarray(self.mean)) / jnp.asarray(self.std)

    def from_unconstrained(self, values: Any, condition: Any) -> jax.Array:
        raw = jnp.asarray(values) * jnp.asarray(self.std) + jnp.asarray(self.mean)
        ds = jnp.asarray(condition)[..., 2]
        return jnp.stack(
            [jnp.exp(raw[..., 0]), ds * jax.nn.sigmoid(raw[..., 1]), raw[..., 2], raw[..., 3]], axis=-1
        )

    def log_abs_det_unconstrained_wrt_physical(self, rest: Any, condition: Any) -> jax.Array:
        x = jnp.asarray(rest)
        ds = jnp.asarray(condition)[..., 2]
        ml, dl = x[..., 0], x[..., 1]
        return jnp.log(ds) - jnp.log(ml) - jnp.log(dl) - jnp.log(ds - dl) - jnp.sum(jnp.log(jnp.asarray(self.std)))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ResidualTransform":
        return cls(tuple(float(x) for x in data["mean"]), tuple(float(x) for x in data["std"]))


class ResidualFlowDensity:
    def __init__(self, flow: Any, transform: ResidualTransform, config: FlowConfig, condition_transform: ConditionTransform):
        self.flow = flow
        self.transform = transform
        self.config = config
        self.condition_transform = condition_transform

    @classmethod
    def init(cls, transform: ResidualTransform, config: FlowConfig, condition_transform: ConditionTransform) -> "ResidualFlowDensity":
        flow = masked_autoregressive_flow(
            jax.random.key(config.seed),
            base_dist=Normal(jnp.zeros(config.dim)),
            cond_dim=config.cond_dim,
            flow_layers=config.flow_layers,
            nn_width=config.nn_width,
            nn_depth=config.nn_depth,
        )
        return cls(flow, transform, config, condition_transform)

    def log_density_array(self, rest: Any, condition: Any) -> jax.Array:
        x, cond = jnp.asarray(rest), jnp.asarray(condition)
        z = self.transform.to_unconstrained(x, cond)
        std_cond = self.condition_transform.to_standard(cond)
        logp = self.flow.log_prob(z, condition=std_cond) + self.transform.log_abs_det_unconstrained_wrt_physical(x, cond)
        valid = (x[..., 0] > 0.0) & (x[..., 1] > 0.0) & (cond[..., 2] > x[..., 1])
        return jnp.where(valid & jnp.isfinite(logp), logp, -jnp.inf)

    def sample(self, key: jax.Array, condition: Any) -> jax.Array:
        cond = jnp.asarray(condition)
        z = self.flow.sample(key, condition=self.condition_transform.to_standard(cond))
        return self.transform.from_unconstrained(z, cond)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps({
            "config": self.config.to_json(),
            "transform": self.transform.to_json(),
            "condition_transform": self.condition_transform.to_json(),
            "feature_names": REST_FEATURE_NAMES,
            "condition_names": CONDITION_NAMES,
        }, indent=2))
        eqx.tree_serialise_leaves(model_dir / "flow.eqx", self.flow)

    @classmethod
    def load(cls, model_dir: str | Path) -> "ResidualFlowDensity":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "config.json").read_text())
        config = FlowConfig.from_json(meta["config"])
        transform = ResidualTransform.from_json(meta["transform"])
        condition_transform = ConditionTransform.from_json(meta["condition_transform"])
        template = cls.init(transform, config, condition_transform)
        flow = eqx.tree_deserialise_leaves(model_dir / "flow.eqx", template.flow)
        return cls(flow, transform, config, condition_transform)


def _as_condition_array(condition: np.ndarray) -> np.ndarray:
    values = np.asarray(condition, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"expected condition array with shape (n, d>0), got {values.shape}")
    return values
