from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax.numpy as jnp
from jax import vmap

from gapmoe.priors.event_rate_backend import log_event_rate_backend


Context = Mapping[str, Any] | None
OffsetCalculator = Callable[[Any, Context], jnp.ndarray]


@dataclass(frozen=True)
class SourceCmdPrior:
    """Source CMD population model, separate from the five-dimensional event prior."""

    density: Any
    cmd_prior: Any
    offset_calculator: OffsetCalculator

    def component_density_at_distance(
        self,
        ds_kpc: Any,
        reference_magnitude: Any,
        color: Any,
        *,
        context: Context = None,
    ):
        """Return p(DS, source component, CMD) for every source component."""

        offsets = self.offset_calculator(ds_kpc, context)
        photometric = self.cmd_prior.density_all_components(reference_magnitude, color, offsets)
        components = self.density.distance.source_component_values(ds_kpc)
        return components * photometric / self.density.distance.source_norm

    def density_at_distance(self, ds_kpc: Any, reference_magnitude: Any, color: Any, *, context: Context = None):
        """Return p(DS, CMD) after marginalizing the source component."""

        return jnp.sum(self.component_density_at_distance(ds_kpc, reference_magnitude, color, context=context))

    def marginal_density(self, reference_magnitude: Any, color: Any, *, context: Context = None):
        """Return the marginal source-CMD prior p(CMD | l,b)."""

        distances = self.density.distance.distance_pc / 1000.0
        values = vmap(lambda ds: self.density_at_distance(ds, reference_magnitude, color, context=context))(distances)
        return jnp.sum(0.5 * (values[1:] + values[:-1]) * (distances[1:] - distances[:-1]))

    def log_marginal_density(self, reference_magnitude: Any, color: Any, *, context: Context = None):
        value = self.marginal_density(reference_magnitude, color, context=context)
        return jnp.where(value > 0.0, jnp.log(value), -jnp.inf)



@dataclass(frozen=True)
class EventPrior5D:
    """Five-dimensional event prior conditionable on source CMD information.

    ``log_density(..., reference_magnitude=..., color=...)`` evaluates
    p(event | CMD). It intentionally excludes p(CMD), which callers may add
    later through ``source_prior.log_density`` when they want a CMD prior.
    """

    density: Any
    source_prior: SourceCmdPrior
    include_event_rate: bool = True

    def log_density(
        self,
        ml: Any,
        dl: Any,
        ds: Any,
        mu_n: Any,
        mu_e: Any,
        *,
        reference_magnitude: Any | None = None,
        color: Any | None = None,
        context: Context = None,
    ):
        if reference_magnitude is None or color is None:
            value = self.density.log_density(ml, dl, ds, mu_n, mu_e)
        else:
            offsets = self.source_prior.offset_calculator(ds, context)
            joint = self.density.log_cmd_joint_density(
                ml,
                dl,
                ds,
                mu_n,
                mu_e,
                cmd_prior=self.source_prior.cmd_prior,
                reference_magnitude=reference_magnitude,
                color=color,
                magnitude_offsets=offsets,
            )
            value = joint - self.source_prior.log_marginal_density(reference_magnitude, color, context=context)
        if self.include_event_rate:
            value = value + log_event_rate_backend(ml, dl, ds, jnp.hypot(mu_n, mu_e))
        return value
