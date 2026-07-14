from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp

from gapmoe.param_types.base import MappingContext, ParamTypeProtocol
from gapmoe.priors.event_rate_backend import log_event_rate_backend

PhysicalValues = tuple[Any, ...]
MuPhysicalValues = tuple[float, float, float, float]
ThetaMuValues = tuple[float, float]
ExtraLogPrior = Callable[[float, float, float, float, float], jnp.ndarray]


class MappedGalacticModel:
    """JAX prior composition for density backends with a JAX log_density method.

    Physical parameters use kpc for DL and DS.
    """

    def __init__(
        self,
        density: Any,
        *,
        param_type: Optional[ParamTypeProtocol] = None,
        include_event_rate: bool = True,
        extra_log_prior: Optional[ExtraLogPrior] = None,
    ) -> None:
        self.density = density
        self.param_type = param_type
        self.include_event_rate = include_event_rate
        self.extra_log_prior = extra_log_prior

    def log_prob(self, theta: Any, *args: Any, context: Optional[MappingContext] = None) -> jnp.ndarray:
        if len(args) == 1 and isinstance(args[0], dict) and context is None:
            context = args[0]
            args = ()
        if self._marginalizes_distance(args):
            return self._log_prob_marginalized_distance(theta, context)

        if self._uses_theta_mu_physical(args):
            params, log_jacobian = self._to_theta_mu_physical_with_jacobian(theta, context)
            logp = self.log_prob_theta_mu(params)
            logp = logp + log_jacobian
            return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)

        if self._uses_mu_physical(args):
            params, log_jacobian = self._to_mu_physical_with_jacobian(theta, context)
            logp = self.log_prob_mu(params)
            logp = logp + log_jacobian
            return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)

        params, log_jacobian = self._to_physical_with_jacobian(theta, args, context)
        logp = self.log_prob_physical(params)
        logp = logp + log_jacobian
        return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)

    def log_prob_physical(self, values: Any, *args: Any) -> jnp.ndarray:
        """Evaluate the Galactic prior directly in physical parameters.

        Parameters are ``(ML, DL, DS, mu_N, mu_E)`` with distances in kpc and
        proper motions in mas/yr. No light-curve param_type Jacobian is
        applied here.
        """
        ML, DL, DS, mu_N, mu_E = _raw_values(values, args)
        mu = jnp.hypot(mu_N, mu_E)
        logp = self.density.log_density(ML, DL, DS, mu_N, mu_E)
        if self.include_event_rate:
            logp = logp + log_event_rate_backend(ML, DL, DS, mu)

        if self.extra_log_prior is not None:
            logp = logp + self.extra_log_prior(ML, DL, DS, mu_N, mu_E)

        return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)

    def log_prob_mu(self, values: Any, *args: Any) -> jnp.ndarray:
        """Evaluate the prior in ``(ML, DL, DS, mu)`` after marginalizing phi."""
        if self.extra_log_prior is not None:
            raise TypeError("extra_log_prior is not supported for direction-marginalized mu priors.")
        if not hasattr(self.density, "log_density_mu"):
            raise TypeError("density must provide log_density_mu(ML, DL, DS, mu).")

        ML, DL, DS, mu = _raw_mu_values(values, args)
        logp = self.density.log_density_mu(ML, DL, DS, mu)
        if self.include_event_rate:
            logp = logp + log_event_rate_backend(ML, DL, DS, mu)
        return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)

    def log_prob_theta_mu(self, values: Any, *args: Any) -> jnp.ndarray:
        """Evaluate the prior in ``(thetaE, mu)`` after marginalizing DL, DS, and phi."""
        if self.extra_log_prior is not None:
            raise TypeError("extra_log_prior is not supported for distance-marginalized priors.")
        if not hasattr(self.density, "log_density_theta_mu"):
            raise TypeError("density must provide log_density_theta_mu(thetaE, mu).")

        theta_e, mu = _raw_theta_mu_values(values, args)
        return self.density.log_density_theta_mu(
            theta_e,
            mu,
            include_event_rate=self.include_event_rate,
        )

    def to_physical(self, theta: Any, *, context: Optional[MappingContext] = None) -> PhysicalValues:
        """Convert light-curve parameters to physical parameters.

        The first five values are always ``(ML, DL, DS, mu_N, mu_E)``. Param
        types with additional physical/derived coordinates append them after
        those density coordinates.
        """
        if self._uses_theta_mu_physical(()) or self._uses_mu_physical(()):
            raise TypeError(
                "This param_type marginalizes physical dimensions; use "
                "to_theta_mu_physical() or to_mu_physical()."
            )
        if self.param_type is None:
            return _raw_values(theta, ())
        return _coerce_physical(self.param_type.to_physical(theta, context))

    def to_mu_physical(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> MuPhysicalValues:
        """Convert light-curve parameters to ``(ML, DL, DS, mu)``."""
        values, _ = self._to_mu_physical_with_jacobian(theta, context)
        return values

    def to_theta_mu_physical(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> ThetaMuValues:
        """Convert light-curve parameters to ``(thetaE, mu)``."""
        values, _ = self._to_theta_mu_physical_with_jacobian(theta, context)
        return values

    def to_deterministic_physical(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> dict[str, Any]:
        """Return physical values that are deterministic for this theta."""
        if self.param_type is None:
            ML, DL, DS, mu_N, mu_E = _raw_values(theta, ())
            return {
                "ML": ML,
                "DL": DL,
                "DS": DS,
                "mu_N": mu_N,
                "mu_E": mu_E,
            }

        if hasattr(self.param_type, "to_deterministic_physical"):
            try:
                return dict(self.param_type.to_deterministic_physical(theta, context))
            except TypeError:
                pass

        if self._uses_theta_mu_physical(()):
            thetaE, mu = self.to_theta_mu_physical(theta, context=context)
            return {"thetaE": thetaE, "mu": mu}

        if self._uses_mu_physical(()):
            ML, DL, DS, mu = self.to_mu_physical(theta, context=context)
            return {"ML": ML, "DL": DL, "DS": DS, "mu": mu}

        physical = self.to_physical(theta, context=context)
        keys = ["ML", "DL", "DS", "mu_N", "mu_E"]
        keys.extend(getattr(self.param_type, "derived_names", ()))
        return {
            key: value
            for key, value in zip(keys, physical)
        }

    def log_abs_det_jacobian(
        self,
        theta: Any,
        *,
        context: Optional[MappingContext] = None,
    ) -> jnp.ndarray:
        """Return the log absolute Jacobian determinant for ``theta``."""
        if self._marginalizes_distance(()):
            raise TypeError("distance-marginalized param_types integrate the full Jacobian.")
        if self._uses_theta_mu_physical(()):
            _, log_jacobian = self._to_theta_mu_physical_with_jacobian(theta, context)
        elif self._uses_mu_physical(()):
            _, log_jacobian = self._to_mu_physical_with_jacobian(theta, context)
        else:
            _, log_jacobian = self._to_physical_with_jacobian(theta, (), context)
        return log_jacobian

    def _log_prob_marginalized_distance(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> jnp.ndarray:
        distances, weights = _distance_grid_and_weights(self.density)
        param_type = self.param_type
        if bool(getattr(param_type, "supports_distance_grid", False)):
            values = param_type.jax_physical_with_distance_grid(theta, distances, context)
            logp = vmap(
                lambda ML, DL, DS, mu_N, mu_E: self.log_prob_physical(
                    (ML, DL, DS, mu_N, mu_E)
                )
            )(*values)
            log_jacobian = param_type.jax_log_abs_det_jacobian_with_distance_grid(
                theta,
                distances,
                context,
            )
            terms = jnp.log(weights) + logp + log_jacobian
            return logsumexp(jnp.where(jnp.isfinite(terms), terms, -jnp.inf))

        full_impl = param_type.distance_impl

        def log_integrand(distance, weight):
            full_theta = param_type.with_distance(theta, distance)
            values = _coerce_values(full_impl.to_physical(full_theta, context))
            logp = self.log_prob_physical(values)
            log_jacobian = full_impl.log_abs_det_jacobian(full_theta, context)
            return jnp.log(weight) + logp + log_jacobian

        terms = vmap(log_integrand)(distances, weights)
        return logsumexp(terms)

    def _to_physical_with_jacobian(
        self,
        theta: Any,
        args: tuple[Any, ...],
        context: Optional[MappingContext],
    ) -> tuple[PhysicalValues, jnp.ndarray]:
        if self.param_type is None:
            return _raw_values(theta, args), jnp.asarray(0.0)

        if args:
            raise TypeError("MappedGalacticModel with a param_type expects one theta object, not raw arguments.")
        values = _coerce_values(self.param_type.to_physical(theta, context))
        log_jacobian = self.param_type.log_abs_det_jacobian(theta, context)
        return values, log_jacobian

    def _to_mu_physical_with_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> tuple[MuPhysicalValues, jnp.ndarray]:
        if self.param_type is None or not hasattr(self.param_type, "to_mu_physical"):
            return _raw_mu_values(theta, ()), jnp.asarray(0.0)

        values = _coerce_mu_values(self.param_type.to_mu_physical(theta, context))
        log_jacobian = self.param_type.log_abs_det_jacobian(theta, context)
        return values, log_jacobian

    def _to_theta_mu_physical_with_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext],
    ) -> tuple[ThetaMuValues, jnp.ndarray]:
        if self.param_type is None or not hasattr(self.param_type, "to_theta_mu_physical"):
            return _raw_theta_mu_values(theta, ()), jnp.asarray(0.0)

        values = _coerce_theta_mu_values(self.param_type.to_theta_mu_physical(theta, context))
        log_jacobian = self.param_type.log_abs_det_jacobian(theta, context)
        return values, log_jacobian

    def _uses_theta_mu_physical(self, args: tuple[Any, ...]) -> bool:
        return (
            self.param_type is not None
            and not args
            and bool(getattr(self.param_type, "uses_theta_mu_physical", False))
        )

    def _marginalizes_distance(self, args: tuple[Any, ...]) -> bool:
        return (
            self.param_type is not None
            and not args
            and bool(getattr(self.param_type, "marginalizes_distance", False))
        )

    def _uses_mu_physical(self, args: tuple[Any, ...]) -> bool:
        return (
            self.param_type is not None
            and not args
            and bool(getattr(self.param_type, "uses_mu_physical", False))
        )


def _raw_values(first: Any, rest: tuple[Any, ...]) -> PhysicalValues:
    if rest:
        return _coerce_values((first, *rest))
    return _coerce_values(first)


def _coerce_values(values: Any) -> tuple[Any, Any, Any, Any, Any]:
    if len(values) < 5:
        raise TypeError("log_prob expects at least ML, DL, DS, mu_N, mu_E.")
    ML, DL, DS, mu_N, mu_E = values[:5]
    return ML, DL, DS, mu_N, mu_E


def _coerce_physical(values: Any) -> PhysicalValues:
    if len(values) < 5:
        raise TypeError("to_physical expects at least ML, DL, DS, mu_N, mu_E.")
    return tuple(values)


def _raw_mu_values(first: Any, rest: tuple[Any, ...]) -> MuPhysicalValues:
    if rest:
        return _coerce_mu_values((first, *rest))
    return _coerce_mu_values(first)


def _coerce_mu_values(values: Any) -> MuPhysicalValues:
    if len(values) != 4:
        raise TypeError("log_prob_mu expects ML, DL, DS, mu.")
    ML, DL, DS, mu = values
    return ML, DL, DS, mu


def _raw_theta_mu_values(first: Any, rest: tuple[Any, ...]) -> ThetaMuValues:
    if rest:
        return _coerce_theta_mu_values((first, *rest))
    return _coerce_theta_mu_values(first)


def _coerce_theta_mu_values(values: Any) -> ThetaMuValues:
    if len(values) != 2:
        raise TypeError("log_prob_theta_mu expects thetaE, mu.")
    theta_e, mu = values
    return theta_e, mu


def _distance_grid_and_weights(density: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    if not hasattr(density, "distance") or not hasattr(density.distance, "distance_pc"):
        raise TypeError("density must expose distance.distance_pc for distance marginalization.")
    distances = jnp.asarray(density.distance.distance_pc) / 1000.0
    if distances.shape[0] < 2:
        raise ValueError("distance grid must contain at least two points.")
    weights = jnp.empty_like(distances)
    weights = weights.at[0].set(0.5 * (distances[1] - distances[0]))
    weights = weights.at[-1].set(0.5 * (distances[-1] - distances[-2]))
    weights = weights.at[1:-1].set(0.5 * (distances[2:] - distances[:-2]))
    return distances, weights
