from __future__ import annotations

from math import isfinite
from typing import Any, Callable, Optional, Protocol

from gapmoe.density.base import DensityModel
from gapmoe.physical import PhysicalParams
from gapmoe.priors.event_rate import log_event_rate


class Parameterization(Protocol):
    """Map user/light-curve parameters into canonical physical parameters."""

    def to_physical(self, theta: Any, context: Optional[MappingContext] = None) -> PhysicalParams:
        ...

    def log_abs_det_jacobian(self, theta: Any, context: Optional[MappingContext] = None) -> float:
        ...


MappingContext = dict[str, Any]
ExtraLogPrior = Callable[[PhysicalParams], float]


class GalacticPrior:
    """Compose a Galactic density model with event-rate and optional mappings."""

    def __init__(
        self,
        density: DensityModel,
        *,
        parameterization: Optional[Parameterization] = None,
        include_event_rate: bool = True,
        extra_log_prior: Optional[ExtraLogPrior] = None,
    ) -> None:
        self.density = density
        self.parameterization = parameterization
        self.include_event_rate = include_event_rate
        self.extra_log_prior = extra_log_prior

    def log_prob(self, theta_or_params: Any, context: Optional[MappingContext] = None) -> float:
        params, log_jacobian = self._to_physical(theta_or_params, context)
        if not isfinite(log_jacobian):
            return float("-inf")

        logp = self.density.log_density(params)
        if not isfinite(logp):
            return float("-inf")

        if self.include_event_rate:
            logp += log_event_rate(params)
            if not isfinite(logp):
                return float("-inf")

        if log_jacobian != 0.0:
            logp += log_jacobian
            if not isfinite(logp):
                return float("-inf")

        if self.extra_log_prior is not None:
            logp += self.extra_log_prior(params)
            if not isfinite(logp):
                return float("-inf")

        return logp

    def _to_physical(self, theta_or_params: Any, context: Optional[MappingContext]) -> tuple[PhysicalParams, float]:
        if self.parameterization is None:
            if not isinstance(theta_or_params, PhysicalParams):
                raise TypeError("GalacticPrior without a parameterization expects PhysicalParams.")
            return theta_or_params, 0.0

        params = self.parameterization.to_physical(theta_or_params, context)
        log_jacobian = self.parameterization.log_abs_det_jacobian(theta_or_params, context)
        return params, log_jacobian
