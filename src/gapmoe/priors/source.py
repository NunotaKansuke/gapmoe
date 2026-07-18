from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple

import jax.numpy as jnp
from jax import vmap

from gapmoe.priors.event_rate_backend import log_event_rate_backend, log_flow_kernel_rate_backend


Context = Mapping[str, Any] | None
OffsetCalculator = Callable[[Any, Context], jnp.ndarray]


class SourceRadiusEstimate(NamedTuple):
    """Log-normal summary of the source-radius posterior, in solar radii."""

    mean_rsun: Any
    std_rsun: Any
    median_rsun: Any
    p16_rsun: Any
    p84_rsun: Any
    sigma_log_rsun: Any


class ThetaStarProposal(NamedTuple):
    """Moment-matched proposal for ``log(thetaS/mas)``."""

    log_center: Any
    log_sigma: Any


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

    def conditional_density_at_distance(
        self,
        ds_kpc: Any,
        reference_magnitude: Any,
        color: Any,
        *,
        context: Context = None,
    ):
        """Return p(photometry | DS) under the Galactic source-component mixture."""

        offsets = self.offset_calculator(ds_kpc, context)
        photometric = self.cmd_prior.density_all_components(reference_magnitude, color, offsets)
        components = self.density.distance.source_component_values(ds_kpc)
        normalisation = jnp.sum(components)
        return jnp.where(normalisation > 0.0, jnp.sum(components * photometric) / normalisation, 0.0)

    def log_conditional_density_at_distance(
        self,
        ds_kpc: Any,
        reference_magnitude: Any,
        color: Any,
        *,
        context: Context = None,
    ):
        value = self.conditional_density_at_distance(ds_kpc, reference_magnitude, color, context=context)
        return jnp.where(value > 0.0, jnp.log(value), -jnp.inf)

    def source_radius_at_distance(
        self,
        ds_kpc: Any,
        reference_magnitude: Any,
        color: Any,
        *,
        context: Context = None,
    ):
        """Return a lightweight log-normal source-radius posterior summary.

        The CMD table carries density-weighted log-radius moments. Component
        weights are updated by the supplied photometry at the given distance.
        """

        offsets = self.offset_calculator(ds_kpc, context)
        density = self.cmd_prior.density_all_components(reference_magnitude, color, offsets)
        first, second = self.cmd_prior.log_radius_moments_all_components(reference_magnitude, color, offsets)
        components = self.density.distance.source_component_values(ds_kpc)
        denominator = jnp.sum(components * density)
        mean_log_radius = jnp.where(denominator > 0.0, jnp.sum(components * first) / denominator, 0.0)
        second_log_radius = jnp.where(denominator > 0.0, jnp.sum(components * second) / denominator, 0.0)
        variance_log_radius = jnp.maximum(0.0, second_log_radius - mean_log_radius**2)
        sigma_log_radius = jnp.sqrt(variance_log_radius)
        median = jnp.exp(mean_log_radius)
        mean = jnp.exp(mean_log_radius + 0.5 * variance_log_radius)
        std = jnp.sqrt((jnp.exp(variance_log_radius) - 1.0) * jnp.exp(2.0 * mean_log_radius + variance_log_radius))
        valid = denominator > 0.0
        return SourceRadiusEstimate(
            mean_rsun=jnp.where(valid, mean, jnp.nan),
            std_rsun=jnp.where(valid, std, jnp.nan),
            median_rsun=jnp.where(valid, median, jnp.nan),
            p16_rsun=jnp.where(valid, jnp.exp(mean_log_radius - sigma_log_radius), jnp.nan),
            p84_rsun=jnp.where(valid, jnp.exp(mean_log_radius + sigma_log_radius), jnp.nan),
            sigma_log_rsun=jnp.where(valid, sigma_log_radius, jnp.nan),
        )

    def log_theta_star_density_at_distance(
        self,
        ds_kpc: Any,
        reference_magnitude: Any,
        color: Any,
        theta_star_mas: Any,
        *,
        context: Context = None,
    ):
        """Return ``log p(log thetaS | DS, apparent CMD)``."""

        offsets = self.offset_calculator(ds_kpc, context)
        photometric = self.cmd_prior.density_all_components(
            reference_magnitude, color, offsets
        )
        theta_density = self.cmd_prior.theta_star_log_density_all_components(
            reference_magnitude,
            color,
            offsets,
            theta_star_mas=theta_star_mas,
            ds_kpc=ds_kpc,
        )
        components = self.density.distance.source_component_values(ds_kpc)
        weights = components * photometric
        denominator = jnp.sum(weights)
        value = jnp.where(
            denominator > 0.0,
            jnp.sum(weights * theta_density) / denominator,
            0.0,
        )
        return jnp.where(value > 0.0, jnp.log(value), -jnp.inf)

    def _theta_star_proposal(
        self,
        reference_magnitude: Any,
        color: Any,
        *,
        context: Context = None,
    ) -> ThetaStarProposal:
        """Moment-match ``p(log thetaS | apparent CMD)`` over source distance."""

        distances = self.density.distance.distance_pc / 1000.0
        widths = jnp.empty_like(distances)
        widths = widths.at[0].set(0.5 * (distances[1] - distances[0]))
        widths = widths.at[-1].set(0.5 * (distances[-1] - distances[-2]))
        widths = widths.at[1:-1].set(0.5 * (distances[2:] - distances[:-2]))

        def moments_at_distance(ds_kpc):
            offsets = self.offset_calculator(ds_kpc, context)
            density = self.cmd_prior.density_all_components(
                reference_magnitude, color, offsets
            )
            first, second = self.cmd_prior.log_radius_moments_all_components(
                reference_magnitude, color, offsets
            )
            components = self.density.distance.source_component_values(ds_kpc)
            shift = jnp.log(4.650467260962157) - jnp.log(ds_kpc * 1000.0)
            zeroth = jnp.sum(components * density)
            first_theta = jnp.sum(components * (first + density * shift))
            second_theta = jnp.sum(
                components
                * (second + 2.0 * shift * first + density * shift**2)
            )
            return zeroth, first_theta, second_theta

        zeroth, first, second = vmap(moments_at_distance)(distances)
        normalisation = jnp.sum(widths * zeroth)
        mean = jnp.sum(widths * first) / normalisation
        variance = jnp.maximum(jnp.sum(widths * second) / normalisation - mean**2, 0.0)
        # This is only an importance proposal. A modest floor/inflation keeps
        # support broad when one CMD bin dominates.
        sigma = jnp.maximum(1.25 * jnp.sqrt(variance), 0.25)
        valid = (normalisation > 0.0) & jnp.isfinite(mean) & jnp.isfinite(sigma)
        return ThetaStarProposal(
            log_center=jnp.where(valid, mean, jnp.nan),
            log_sigma=jnp.where(valid, sigma, jnp.nan),
        )

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
        if self.include_event_rate and not getattr(self.density, "event_rate_included", False):
            rate = (
                log_flow_kernel_rate_backend(ml, dl, ds, jnp.hypot(mu_n, mu_e))
                if getattr(self.density, "event_rate_factor_includes_lens_area", False)
                else log_event_rate_backend(ml, dl, ds, jnp.hypot(mu_n, mu_e))
            )
            value = value + rate
        return value

    def log_joint_density(
        self,
        ml: Any,
        dl: Any,
        ds: Any,
        mu_n: Any,
        mu_e: Any,
        *,
        reference_magnitude: Any,
        color: Any,
        theta_star_mas: Any | None = None,
        context: Context = None,
    ):
        """Evaluate the joint event and source-photometry density."""

        offsets = self.source_prior.offset_calculator(ds, context)
        source_component_factor = None
        if theta_star_mas is not None:
            source_component_factor = (
                self.source_prior.cmd_prior.theta_star_log_density_all_components(
                    reference_magnitude,
                    color,
                    offsets,
                    theta_star_mas=theta_star_mas,
                    ds_kpc=ds,
                )
            )
        value = self.density.log_cmd_joint_density(
            ml,
            dl,
            ds,
            mu_n,
            mu_e,
            cmd_prior=self.source_prior.cmd_prior,
            reference_magnitude=reference_magnitude,
            color=color,
            magnitude_offsets=offsets,
            source_component_factor=source_component_factor,
        )
        if self.include_event_rate and not getattr(self.density, "event_rate_included", False):
            rate = (
                log_flow_kernel_rate_backend(ml, dl, ds, jnp.hypot(mu_n, mu_e))
                if getattr(self.density, "event_rate_factor_includes_lens_area", False)
                else log_event_rate_backend(ml, dl, ds, jnp.hypot(mu_n, mu_e))
            )
            value = value + rate
        return value
