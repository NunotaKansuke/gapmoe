from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import jax.numpy as jnp

from gapmoe.priors.event_rate_jax import jax_log_event_rate

PhysicalValues = tuple[float, float, float, float, float]
MappingContext = dict[str, Any]
ExtraLogPrior = Callable[[float, float, float, float, float], jnp.ndarray]


class JaxParameterization(Protocol):
    """Map user/light-curve parameters into ML, DL, DS, mu_N, mu_E."""

    def to_physical(self, theta: Any, context: Optional[MappingContext] = None) -> PhysicalValues:
        ...

    def log_abs_det_jacobian(self, theta: Any, context: Optional[MappingContext] = None) -> jnp.ndarray:
        ...


class JaxGalacticPrior:
    """JAX prior composition for density backends with a JAX log_density method.

    Physical parameters use kpc for DL and DS.
    """

    def __init__(
        self,
        density: Any,
        *,
        parameterization: Optional[JaxParameterization] = None,
        include_event_rate: bool = True,
        extra_log_prior: Optional[ExtraLogPrior] = None,
    ) -> None:
        self.density = density
        self.parameterization = parameterization
        self.include_event_rate = include_event_rate
        self.extra_log_prior = extra_log_prior

    def log_prob(self, theta: Any, *args: Any, context: Optional[MappingContext] = None) -> jnp.ndarray:
        if len(args) == 1 and isinstance(args[0], dict) and context is None:
            context = args[0]
            args = ()
        params, log_jacobian = self._to_physical(theta, args, context)
        ML, DL, DS, mu_N, mu_E = params
        mu = jnp.hypot(mu_N, mu_E)

        logp = self.density.log_density(ML, DL, DS, mu_N, mu_E)
        if self.include_event_rate:
            logp = logp + jax_log_event_rate(ML, DL, DS, mu)
        logp = logp + log_jacobian

        if self.extra_log_prior is not None:
            logp = logp + self.extra_log_prior(ML, DL, DS, mu_N, mu_E)

        return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)

    def _to_physical(
        self,
        theta: Any,
        args: tuple[Any, ...],
        context: Optional[MappingContext],
    ) -> tuple[PhysicalValues, jnp.ndarray]:
        if self.parameterization is None:
            return _raw_values(theta, args), jnp.asarray(0.0)

        if args:
            raise TypeError("JaxGalacticPrior with a parameterization expects one theta object, not raw arguments.")
        values = _coerce_values(self.parameterization.to_physical(theta, context))
        log_jacobian = self.parameterization.log_abs_det_jacobian(theta, context)
        return values, log_jacobian


def _raw_values(first: Any, rest: tuple[Any, ...]) -> PhysicalValues:
    if rest:
        return _coerce_values((first, *rest))
    return _coerce_values(first)


def _coerce_values(values: Any) -> PhysicalValues:
    if len(values) != 5:
        raise TypeError("log_prob expects ML, DL, DS, mu_N, mu_E.")
    ML, DL, DS, mu_N, mu_E = values
    return ML, DL, DS, mu_N, mu_E
