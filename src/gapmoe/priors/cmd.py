from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

Context = Mapping[str, Any] | None
PhysicalExtractor = Callable[[Any, Context], tuple[Any, Any, Any, Any, Any]]
CmdExtractor = Callable[[Any, Context], tuple[Any, Any]]


def _first_five(theta: Any, context: Context) -> tuple[Any, Any, Any, Any, Any]:
    del context
    return theta[0], theta[1], theta[2], theta[3], theta[4]


@dataclass(frozen=True)
class CmdGalacticModel:
    """Conditional five-dimensional Galactic prior for an MCMC state.

    The default is p(event | CMD). Set ``include_cmd_prior=True`` only when
    sampled source CMD variables should also receive p(CMD).
    """

    event_prior: Any
    cmd_extractor: CmdExtractor
    physical_extractor: PhysicalExtractor = _first_five
    include_cmd_prior: bool = False

    def log_prob(self, theta: Any, *, context: Context = None):
        """Evaluate the event prior at the current source CMD state."""

        ml, dl, ds, mu_n, mu_e = self.physical_extractor(theta, context)
        reference_magnitude, color = self.cmd_extractor(theta, context)
        log_density = self.event_prior.log_density(
            ml,
            dl,
            ds,
            mu_n,
            mu_e,
            reference_magnitude=reference_magnitude,
            color=color,
            context=context,
        )
        if self.include_cmd_prior:
            log_density = log_density + self.event_prior.source_prior.log_marginal_density(
                reference_magnitude,
                color,
                context=context,
            )
        return log_density
