"""Light-curve parameterization of a physical :class:`GalaxyModel`."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from .event_rate_backend import log_event_rate_backend, log_flow_kernel_rate_backend
from .galactic import _ParameterizedNumpyEngine
from .galactic_jax import _ParameterizedJaxEngine


_KAPPA = 8.1429


@dataclass(frozen=True)
class _ImportanceProposal:
    ds: Any
    x: Any
    phi: Any
    log_q: Any


@dataclass(frozen=True)
class _PhysicalDensityView:
    galaxy: Any
    magnitudes: Mapping[str, Any] | None = None
    joint: bool = False
    context: Mapping[str, Any] | None = None
    physical_priors: tuple[Any, ...] = ()
    proposal: _ImportanceProposal | None = None
    direction_phi: Any = None
    source_radius: bool = False

    @property
    def _prior(self):
        return self.galaxy._conditional_prior if self.magnitudes is not None else self.galaxy._selected_prior

    @property
    def distance(self):
        return self._prior.density.distance

    def log_density(self, ml, dl, ds, mu_n, mu_e):
        physical = (ml, dl, ds, mu_n, mu_e)
        if self.joint:
            theta_star_mas = None
            if self.source_radius:
                if self.context is None or "thS" not in self.context:
                    raise ValueError(
                        "source_radius=True requires context['thS'] in mas"
                    )
                theta_star_mas = self.context["thS"]
            value = self.galaxy.log_joint_density(
                physical,
                magnitudes=self.magnitudes,
                theta_star_mas=theta_star_mas,
                context=self.context,
            )
        else:
            value = self.galaxy.log_density(
                physical,
                magnitudes=self.magnitudes,
                context=self.context,
            )
        return value + _physical_prior_sum(
            self.physical_priors,
            ML=ml,
            DL=dl,
            DS=ds,
            mu_N=mu_n,
            mu_E=mu_e,
            mu=jnp.hypot(mu_n, mu_e),
        )

    def log_density_array(self, ml, dl, ds, mu_n, mu_e):
        return jax.vmap(self.log_density)(ml, dl, ds, mu_n, mu_e)

    def log_density_mu(self, ml, dl, ds, mu):
        density = self._prior.density
        evaluator = getattr(density, "log_density_mu", None)
        if evaluator is not None and self.magnitudes is None and not self.physical_priors:
            value = evaluator(ml, dl, ds, mu)
            if self._prior.include_event_rate and not getattr(density, "event_rate_included", False):
                value = value + self._event_rate(ml, dl, ds, mu)
            return value
        terms, _ = self.direction_log_terms(ml, dl, ds, mu)
        return logsumexp(terms) - jnp.log(terms.shape[0])

    def log_density_theta_mu(self, theta_e, mu, *, include_event_rate=False):
        density = self._prior.density
        evaluator = getattr(density, "log_density_theta_mu", None)
        if evaluator is not None and self.magnitudes is None and not self.physical_priors:
            return evaluator(
                theta_e,
                mu,
                include_event_rate=(
                    self._prior.include_event_rate
                    and not getattr(density, "event_rate_included", False)
                ),
            )
        terms, _ = self.theta_mu_log_terms(theta_e, mu)
        return logsumexp(terms) - jnp.log(terms.shape[0])

    def direction_log_terms(self, ml, dl, ds, mu):
        if self.direction_phi is None:
            raise RuntimeError("proper-motion direction proposal is unavailable")
        phi = self.direction_phi
        mu_n = mu * jnp.cos(phi)
        mu_e = mu * jnp.sin(phi)
        logp = jax.vmap(self.log_density)(
            jnp.full_like(phi, ml),
            jnp.full_like(phi, dl),
            jnp.full_like(phi, ds),
            mu_n,
            mu_e,
        )
        terms = logp + jnp.log(jnp.abs(mu)) + jnp.log(2.0 * jnp.pi)
        return terms, (mu_n, mu_e)

    def theta_mu_log_terms(self, theta_e, mu):
        if self.proposal is None:
            raise RuntimeError("distance importance proposal is unavailable")
        ds = self.proposal.ds
        dl = self.proposal.x * ds
        pi_rel = 1.0 / dl - 1.0 / ds
        ml = theta_e**2 / (_KAPPA * pi_rel)
        mu_n = mu * jnp.cos(self.proposal.phi)
        mu_e = mu * jnp.sin(self.proposal.phi)
        logp = jax.vmap(self.log_density)(ml, dl, ds, mu_n, mu_e)
        log_jacobian = (
            jnp.log(2.0 * jnp.abs(theta_e) / (_KAPPA * pi_rel))
            + jnp.log(ds)
            + jnp.log(jnp.abs(mu))
        )
        terms = logp + log_jacobian - self.proposal.log_q
        valid = (
            (theta_e > 0.0)
            & (mu > 0.0)
            & (dl > 0.0)
            & (dl < ds)
            & (ml > 0.0)
        )
        terms = jnp.where(valid & jnp.isfinite(terms), terms, -jnp.inf)
        return terms, (ml, dl, ds, mu_n, mu_e)

    def _event_rate(self, ml, dl, ds, mu):
        density = self._prior.density
        if getattr(density, "event_rate_factor_includes_lens_area", False):
            return log_flow_kernel_rate_backend(ml, dl, ds, mu)
        return log_event_rate_backend(ml, dl, ds, mu)


@dataclass
class ParameterizedGalaxyModel:
    """A physical Galaxy model expressed in light-curve parameters.

    The object is built independently by gapmoe and can be passed to an
    inference package through the small ``names``/``log_density`` protocol.
    """

    galaxy: Any
    param_type: Any
    integration_samples: int = 256
    direction_samples: int = 32
    seed: int = 0
    source_radius: bool = False
    _physical_priors: list[Any] = field(default_factory=list, init=False, repr=False)
    _proposal: _ImportanceProposal | None = field(default=None, init=False, repr=False)
    _compiled: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        names = getattr(self.param_type, "names", None)
        if names is None or not tuple(names):
            raise TypeError("param_type must expose a non-empty names sequence")
        for name, value in (
            ("integration_samples", self.integration_samples),
            ("direction_samples", self.direction_samples),
        ):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.source_radius, bool):
            raise TypeError("source_radius must be bool")
        table = self.galaxy.isochrone.table
        if self.source_radius and (
            table.log_radius_moment_by_component is None
            or table.log_radius_square_moment_by_component is None
        ):
            raise ValueError(
                "source_radius=True requires an isochrone table with radius moments"
            )
        if bool(getattr(self.param_type, "uses_theta_mu_physical", False)):
            self._importance_proposal()

    @property
    def names(self):
        return tuple(self.param_type.names)

    def prior(self, fn):
        """Add a JAX-compatible prior inside hidden-physical integration."""

        if not callable(fn):
            raise TypeError("parameterized Galaxy prior expects a callable")
        self._physical_priors.append(fn)
        self._compiled.clear()
        return fn

    def _view(self, magnitudes=None, *, joint=False, context=None):
        if joint and magnitudes is None:
            raise ValueError("joint source photometry requires magnitudes")
        density = (
            self.galaxy._conditional_prior.density
            if magnitudes is not None
            else self.galaxy._selected_prior.density
        )
        needs_distance_qmc = bool(
            getattr(self.param_type, "uses_theta_mu_physical", False)
        ) and (
            magnitudes is not None
            or bool(self._physical_priors)
            or not hasattr(density, "log_density_theta_mu")
        )
        needs_direction_qmc = bool(
            getattr(self.param_type, "uses_mu_physical", False)
        ) and (
            magnitudes is not None
            or bool(self._physical_priors)
            or not hasattr(density, "log_density_mu")
        )
        proposal = self._importance_proposal() if needs_distance_qmc else None
        direction_phi = (
            2.0
            * jnp.pi
            * (jnp.arange(self.direction_samples) + 0.5)
            / self.direction_samples
            if needs_direction_qmc
            else None
        )
        return _PhysicalDensityView(
            self.galaxy,
            magnitudes,
            joint,
            context,
            tuple(self._physical_priors),
            proposal,
            direction_phi,
            self.source_radius,
        )

    def _importance_proposal(self):
        if self._proposal is None:
            self._proposal = _build_importance_proposal(
                self.galaxy,
                self.integration_samples,
                self.seed,
            )
        return self._proposal

    def _jax_engine(self, magnitudes=None, *, joint=False, context=None):
        return _ParameterizedJaxEngine(
            self._view(magnitudes, joint=joint, context=context),
            param_type=self.param_type,
            include_event_rate=False,
        )

    def _numpy_engine(self, magnitudes=None, *, joint=False, context=None):
        return _ParameterizedNumpyEngine(
            self._view(magnitudes, joint=joint, context=context),
            param_type=self.param_type,
            include_event_rate=False,
        )

    def log_density(self, theta, *, context=None, magnitudes=None):
        if self.source_radius:
            raise ValueError(
                "source_radius=True represents a joint thetaS/photometry density; "
                "use log_joint_density(..., magnitudes=...)"
            )
        self._prepare_integration(magnitudes)
        evaluator = self._compiled_evaluator(
            joint=False,
            has_context=context is not None,
            has_magnitudes=magnitudes is not None,
        )
        args = [jnp.asarray(theta)]
        if context is not None:
            args.append(context)
        if magnitudes is not None:
            args.append(magnitudes)
        return evaluator(*args)

    def is_valid(self, theta, *, context=None):
        """Return whether a direct parameter transform is physically valid."""

        if bool(getattr(self.param_type, "marginalizes_distance", False)):
            return True
        value = self.param_type.log_abs_det_jacobian(jnp.asarray(theta), context)
        return jnp.isfinite(value)

    def log_joint_density(self, theta, *, magnitudes, context=None):
        self._prepare_integration(magnitudes)
        evaluator = self._compiled_evaluator(
            joint=True,
            has_context=context is not None,
            has_magnitudes=True,
        )
        args = [jnp.asarray(theta)]
        if context is not None:
            args.append(context)
        args.append(magnitudes)
        return evaluator(*args)

    def _prepare_integration(self, magnitudes):
        density = (
            self.galaxy._conditional_prior.density
            if magnitudes is not None
            else self.galaxy._selected_prior.density
        )
        if bool(getattr(self.param_type, "uses_theta_mu_physical", False)) and (
            magnitudes is not None
            or bool(self._physical_priors)
            or not hasattr(density, "log_density_theta_mu")
        ):
            self._importance_proposal()

    def _compiled_evaluator(self, *, joint, has_context, has_magnitudes):
        key = (joint, has_context, has_magnitudes)
        if key in self._compiled:
            return self._compiled[key]

        def evaluate(theta, context, magnitudes):
            return self._jax_engine(
                magnitudes,
                joint=joint,
                context=context,
            ).log_prob(theta, context=context)

        if has_context and has_magnitudes:
            fn = lambda theta, context, magnitudes: evaluate(
                theta, context, magnitudes
            )
        elif has_context:
            fn = lambda theta, context: evaluate(theta, context, None)
        elif has_magnitudes:
            fn = lambda theta, magnitudes: evaluate(theta, None, magnitudes)
        else:
            fn = lambda theta: evaluate(theta, None, None)
        self._compiled[key] = jax.jit(fn)
        return self._compiled[key]

    def log_density_batch(self, theta, *, context=None, magnitudes=None, joint=False):
        if joint and magnitudes is None:
            raise ValueError("joint source photometry requires magnitudes")
        evaluator = self.log_joint_density if joint else self.log_density
        if context is None and magnitudes is None:
            return jax.vmap(evaluator)(jnp.asarray(theta))
        return jax.vmap(
            lambda row: evaluator(row, context=context, magnitudes=magnitudes)
        )(jnp.asarray(theta))

    def to_physical(self, theta, *, context=None):
        return self._jax_engine().to_physical(theta, context=context)

    def to_deterministic_physical(self, theta, *, context=None):
        return self._jax_engine().to_deterministic_physical(theta, context=context)

    def to_derived(self, theta, *, context=None):
        if not hasattr(self.param_type, "to_derived"):
            return {}
        return dict(self.param_type.to_derived(theta, context))

    def log_abs_det_jacobian(self, theta, *, context=None):
        return self._jax_engine().log_abs_det_jacobian(theta, context=context)

    def sample_physical(self, theta, *, context=None, magnitudes=None, joint=False, rng=None):
        view = self._view(magnitudes, joint=joint, context=context)
        density = view._prior.density
        if bool(getattr(self.param_type, "uses_theta_mu_physical", False)) and (
            view.proposal is not None
        ):
            return self._sample_qmc_physical(
                theta, view, context=context, rng=rng, theta_mu=True
            )
        if bool(getattr(self.param_type, "uses_mu_physical", False)) and (
            view.direction_phi is not None
        ):
            return self._sample_qmc_physical(
                theta, view, context=context, rng=rng, theta_mu=False
            )
        return self._numpy_engine(
            magnitudes,
            joint=joint,
            context=context,
        ).sample_physical(
            theta,
            context=context,
            rng=rng,
        )

    def _sample_qmc_physical(self, theta, view, *, context, rng, theta_mu):
        rng = np.random.default_rng() if rng is None else rng
        array = np.asarray(theta)
        if array.ndim == 2:
            draws = [
                self._sample_qmc_physical(
                    row,
                    view,
                    context=context,
                    rng=rng,
                    theta_mu=theta_mu,
                )
                for row in array
            ]
            return {
                key: np.asarray([draw[key] for draw in draws])
                for key in draws[0]
            }
        engine = self._jax_engine(context=context)
        if theta_mu:
            theta_e, mu = engine.to_theta_mu_physical(theta, context=context)
            terms, values = view.theta_mu_log_terms(theta_e, mu)
            index = _draw_log_weight_index(terms, rng)
            ml, dl, ds, mu_n, mu_e = (float(value[index]) for value in values)
        else:
            ml, dl, ds, mu = engine.to_mu_physical(theta, context=context)
            terms, values = view.direction_log_terms(ml, dl, ds, mu)
            index = _draw_log_weight_index(terms, rng)
            mu_n, mu_e = (float(value[index]) for value in values)
            ml, dl, ds = float(ml), float(dl), float(ds)
        return {
            "ML": ml,
            "DL": dl,
            "DS": ds,
            "mu_N": mu_n,
            "mu_E": mu_e,
            "mu": float(np.hypot(mu_n, mu_e)),
        }


__all__ = ["ParameterizedGalaxyModel"]


def _build_importance_proposal(galaxy, samples, seed):
    density = galaxy.density
    distance = np.asarray(density.distance.distance_pc, dtype=float) / 1000.0
    source = np.sum(
        np.asarray(density.distance.source_by_component, dtype=float), axis=1
    )
    source = np.maximum(source, 0.0)
    norm = float(np.trapezoid(source, distance))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("source-distance proposal has zero probability")
    source_density = source / norm
    segment_mass = 0.5 * (source_density[1:] + source_density[:-1]) * np.diff(distance)
    cdf = np.concatenate(([0.0], np.cumsum(segment_mass)))
    cdf /= cdf[-1]

    points = _halton_points(samples, 4, seed)
    ds = np.interp(points[:, 0], cdf, distance)
    q_ds = np.interp(ds, distance, source_density)

    ratio_density, ratio_edges = _flow_ratio_histogram(density, seed)
    ratio_probability = ratio_density * np.diff(ratio_edges)
    ratio_cdf = np.concatenate(([0.0], np.cumsum(ratio_probability)))
    ratio_cdf /= ratio_cdf[-1]
    fitted_x = np.interp(points[:, 2], ratio_cdf, ratio_edges)
    x = np.where(points[:, 1] < 0.8, fitted_x, points[:, 2])
    bin_index = np.clip(
        np.searchsorted(ratio_edges, x, side="right") - 1,
        0,
        len(ratio_density) - 1,
    )
    q_x = 0.8 * ratio_density[bin_index] + 0.2
    phi = 2.0 * np.pi * points[:, 3]
    log_q = np.log(q_ds) + np.log(q_x) - np.log(2.0 * np.pi)
    return _ImportanceProposal(
        ds=jnp.asarray(ds),
        x=jnp.asarray(np.clip(x, 1.0e-6, 1.0 - 1.0e-6)),
        phi=jnp.asarray(phi),
        log_q=jnp.asarray(log_q),
    )


def _flow_ratio_histogram(density, seed, *, proposal_samples=2048, bins=32):
    sampler = getattr(density, "sample_source_group", None)
    kernel_sampler = getattr(density, "_sample_kernel", None)
    if sampler is None or kernel_sampler is None:
        edges = np.linspace(0.0, 1.0, bins + 1)
        return np.ones(bins, dtype=float), edges

    source_key, kernel_key = jax.random.split(jax.random.key(seed))
    source_keys = jax.random.split(source_key, proposal_samples)
    kernel_keys = jax.random.split(kernel_key, proposal_samples)
    ds, group = jax.vmap(sampler)(source_keys)
    physical = jax.vmap(kernel_sampler)(kernel_keys, ds, group)
    ratio = np.asarray(physical[:, 1] / physical[:, 2], dtype=float)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.0) & (ratio < 1.0)]
    edges = np.linspace(0.0, 1.0, bins + 1)
    if not len(ratio):
        return np.ones(bins, dtype=float), edges
    counts, _ = np.histogram(ratio, bins=edges)
    density_values = counts.astype(float) + 0.5
    density_values /= np.sum(density_values * np.diff(edges))
    return density_values, edges


def _halton_points(samples, dimensions, seed):
    primes = (2, 3, 5, 7, 11, 13)
    if dimensions > len(primes):
        raise ValueError("Halton generator supports at most six dimensions")
    indices = np.arange(1 + seed * samples, 1 + (seed + 1) * samples)
    return np.column_stack(
        [_radical_inverse(indices, base) for base in primes[:dimensions]]
    )


def _radical_inverse(indices, base):
    values = np.zeros(len(indices), dtype=float)
    factor = 1.0 / base
    current = indices.copy()
    while np.any(current):
        values += factor * (current % base)
        current //= base
        factor /= base
    return values


def _draw_log_weight_index(log_weights, rng):
    values = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("conditional physical distribution has zero probability")
    peak = np.max(values[finite])
    weights = np.where(finite, np.exp(values - peak), 0.0)
    weights /= np.sum(weights)
    return int(rng.choice(len(values), p=weights))


def _physical_prior_sum(priors, **values):
    total = jnp.asarray(0.0)
    for fn in priors:
        total = total + fn(**_accepted_kwargs(fn, values))
    return total


def _accepted_kwargs(fn, values):
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return values
    accepts_all = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_all:
        return values
    accepted = {
        name: values[name]
        for name, parameter in signature.parameters.items()
        if name in values
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in accepted
    ]
    if missing:
        raise TypeError(
            "physical prior requested unavailable value(s): " + ", ".join(missing)
        )
    return accepted
