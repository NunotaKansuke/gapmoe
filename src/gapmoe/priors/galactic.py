from __future__ import annotations

from math import hypot, isfinite
from typing import Any, Callable, Optional

from gapmoe.density.base import DensityModel
from gapmoe.parameterizations.base import MappingContext, Parameterization
from gapmoe.priors.event_rate import log_event_rate

PhysicalValues = tuple[float, float, float, float, float]
ExtraLogPrior = Callable[[float, float, float, float, float], float]


class GalacticPrior:
    """Compose a Galactic density model with event-rate and optional mappings.

    Physical parameters use kpc for DL and DS.
    """

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

    def log_prob(self, theta: Any, *args: Any, context: Optional[MappingContext] = None) -> float:
        if len(args) == 1 and isinstance(args[0], dict) and context is None:
            context = args[0]
            args = ()
        params, log_jacobian = self._to_physical(theta, args, context)
        if not isfinite(log_jacobian):
            return float("-inf")

        ML, DL, DS, mu_N, mu_E = params
        mu = hypot(mu_N, mu_E)

        logp = self.density.log_density(ML, DL, DS, mu_N, mu_E)
        if not isfinite(logp):
            return float("-inf")

        if self.include_event_rate:
            logp += log_event_rate(ML, DL, DS, mu)
            if not isfinite(logp):
                return float("-inf")

        if log_jacobian != 0.0:
            logp += log_jacobian
            if not isfinite(logp):
                return float("-inf")

        if self.extra_log_prior is not None:
            logp += self.extra_log_prior(ML, DL, DS, mu_N, mu_E)
            if not isfinite(logp):
                return float("-inf")

        return logp

    def _to_physical(
        self,
        theta: Any,
        args: tuple[Any, ...],
        context: Optional[MappingContext],
    ) -> tuple[PhysicalValues, float]:
        if self.parameterization is None:
            return _raw_values(theta, args), 0.0

        if args:
            raise TypeError("GalacticPrior with a parameterization expects one theta object, not raw arguments.")
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
    return float(ML), float(DL), float(DS), float(mu_N), float(mu_E)
