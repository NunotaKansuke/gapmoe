from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np
from jax import vmap

from gapmoe.density.histogram_tables import COMPONENT_NAMES, SOURCE_GROUP_BY_COMPONENT, SOURCE_GROUP_NAMES, HistogramTables
from gapmoe.priors.event_rate import KAPPA
from gapmoe.priors.event_rate_backend import log_event_rate_backend
from gapmoe.source_selection import CmdPriorTable, SourceEvidenceGrid


SOURCE_GROUP_MATRIX = np.equal.outer(np.arange(len(SOURCE_GROUP_NAMES)), SOURCE_GROUP_BY_COMPONENT).astype(float)


@dataclass(frozen=True)
class MassHistogram:
    log_mass: jnp.ndarray
    pdf_mass_by_component: jnp.ndarray

    @classmethod
    def from_tables(cls, density: HistogramTables) -> "MassHistogram":
        return cls(
            log_mass=jnp.asarray(density.mass.log_mass),
            pdf_mass_by_component=jnp.asarray(density.mass.pdf_mass_by_component),
        )

    def density_given_component(self, mass: float) -> jnp.ndarray:
        log_mass = jnp.log10(mass)
        values = jnp.array(
            [
                _interp_mass_tail(log_mass, self.log_mass, self.pdf_mass_by_component[:, i])
                for i in range(self.pdf_mass_by_component.shape[1])
            ]
        )
        return jnp.where(mass > 0.0, values, jnp.zeros_like(values))


@dataclass(frozen=True)
class DistanceDensityTable:
    # distance_pc: raw distance grid from rho.dat, in pc.
    distance_pc: jnp.ndarray
    lens_density_by_component: jnp.ndarray
    base_source_density_by_component: jnp.ndarray
    source_density_by_component: jnp.ndarray
    source_density: jnp.ndarray
    lens_density_total: jnp.ndarray
    lens_cumulative_integral: jnp.ndarray
    source_norm: float

    @classmethod
    def from_tables(cls, density: HistogramTables) -> "DistanceDensityTable":
        return cls(
            distance_pc=jnp.asarray(density.distance.distance_pc),
            lens_density_by_component=jnp.asarray(density.distance.lens_density_by_component),
            base_source_density_by_component=jnp.asarray(density.distance.base_source_density_by_component),
            source_density_by_component=jnp.asarray(density.distance.source_density_by_component),
            source_density=jnp.asarray(density.distance.source_density),
            lens_density_total=jnp.asarray(density.distance.lens_density_total),
            lens_cumulative_integral=jnp.asarray(density.distance.lens_cumulative_integral),
            source_norm=float(density.distance.source_norm),
        )

    def source_pdf(self, ds_kpc: float) -> jnp.ndarray:
        val = _interp_positive_tail(ds_kpc * 1000.0, self.distance_pc, self.source_density, lower=0.0)
        return jnp.where((ds_kpc > 0.0) & (self.source_norm > 0.0), 1000.0 * val / self.source_norm, 0.0)

    def lens_pdf_given_source(self, dl_kpc: float, ds_kpc: float) -> jnp.ndarray:
        norm = self._lens_integral_until(ds_kpc)
        val = _interp_positive_tail(dl_kpc * 1000.0, self.distance_pc, self.lens_density_total, lower=0.0)
        return jnp.where((dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (norm > 0.0), 1000.0 * val / norm, 0.0)

    def _lens_integral_until(self, ds_kpc: float) -> jnp.ndarray:
        ds_pc = ds_kpc * 1000.0
        n = self.distance_pc.shape[0]
        idx = jnp.searchsorted(self.distance_pc, ds_pc, side="right") - 1
        idx = jnp.clip(idx, 0, n - 1)

        x0 = self.distance_pc[idx]
        y0 = self.lens_density_total[idx]
        y1 = _interp_positive_tail(ds_pc, self.distance_pc, self.lens_density_total, lower=0.0)
        partial = 0.5 * (y0 + y1) * (ds_pc - x0)
        value = self.lens_cumulative_integral[idx] + partial
        left_rate, right_rate = _tail_rates(self.distance_pc, self.lens_density_total)
        lower_total = self.lens_density_total[0] / left_rate * (1.0 - jnp.exp(-left_rate * self.distance_pc[0]))
        right_delta = jnp.maximum(ds_pc - self.distance_pc[-1], 0.0)
        left_delta = jnp.maximum(self.distance_pc[0] - ds_pc, 0.0)
        tail = self.lens_density_total[-1] / right_rate * (1.0 - jnp.exp(-right_rate * right_delta))
        value = jnp.where(ds_pc <= 0.0, 0.0, lower_total + value)
        value = jnp.where(
            ds_pc < self.distance_pc[0],
            lower_total - self.lens_density_total[0] / left_rate * jnp.exp(-left_rate * left_delta),
            value,
        )
        return jnp.where(ds_pc >= self.distance_pc[-1], lower_total + self.lens_cumulative_integral[-1] + tail, value)

    def component_fractions(self, dl_kpc: float) -> jnp.ndarray:
        vals = jnp.array(
            [
                _interp_positive_tail(
                    dl_kpc * 1000.0,
                    self.distance_pc,
                    self.lens_density_by_component[:, i],
                    lower=0.0,
                )
                for i in range(self.lens_density_by_component.shape[1])
            ]
        )
        total = jnp.sum(vals)
        return jnp.where((dl_kpc > 0.0) & (total > 0.0), vals / total, jnp.zeros_like(vals))

    def source_group_weights(self, ds_kpc: float) -> jnp.ndarray:
        values = self.source_component_values(ds_kpc)
        weights = jnp.asarray(SOURCE_GROUP_MATRIX) @ values
        total = jnp.sum(weights)
        return jnp.where(total > 0.0, weights / total, jnp.zeros_like(weights))

    def source_component_values(self, ds_kpc: float) -> jnp.ndarray:
        values = jnp.array(
            [
                jnp.interp(
                    ds_kpc * 1000.0,
                    self.distance_pc,
                    self.source_density_by_component[:, i],
                    left=0.0,
                    right=0.0,
                )
                for i in range(self.source_density_by_component.shape[1])
            ]
        )
        return values


@dataclass(frozen=True)
class CmdPriorEvaluator:
    """JAX representation of a component-conditional intrinsic CMD table."""

    reference_centers: jnp.ndarray
    color_centers: jnp.ndarray
    density_by_component: jnp.ndarray
    log_radius_moment_by_component: jnp.ndarray
    log_radius_square_moment_by_component: jnp.ndarray
    component_to_column: jnp.ndarray

    @classmethod
    def from_table(cls, table: CmdPriorTable, *, n_components: int = 11) -> "CmdPriorEvaluator":
        component_indices = (
            np.arange(table.density_by_component.shape[0])
            if table.component_indices is None
            else np.asarray(table.component_indices, dtype=int)
        )
        if np.any(component_indices < 0) or np.any(component_indices >= n_components):
            raise ValueError("CMD table component indices are incompatible with this Galactic density")
        component_to_column = np.full(n_components, -1, dtype=int)
        component_to_column[component_indices] = np.arange(len(component_indices), dtype=int)
        return cls(
            reference_centers=jnp.asarray(0.5 * (table.reference_edges[:-1] + table.reference_edges[1:])),
            color_centers=jnp.asarray(0.5 * (table.color_edges[:-1] + table.color_edges[1:])),
            density_by_component=jnp.asarray(table.density_by_component),
            log_radius_moment_by_component=jnp.asarray(
                table.log_radius_moment_by_component
                if table.log_radius_moment_by_component is not None else np.zeros_like(table.density_by_component)
            ),
            log_radius_square_moment_by_component=jnp.asarray(
                table.log_radius_square_moment_by_component
                if table.log_radius_square_moment_by_component is not None else np.zeros_like(table.density_by_component)
            ),
            component_to_column=jnp.asarray(component_to_column),
        )

    def density_all_components(
        self,
        reference_magnitude: float,
        color: float,
        magnitude_offsets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate apparent CMD density for every Galactic source component.

        ``magnitude_offsets`` has shape ``(3,)`` for component-independent
        offsets or ``(n_component, 3)`` for component-specific values ordered
        as ``(reference, blue, red)``.
        """

        offsets = jnp.asarray(magnitude_offsets)
        if offsets.ndim == 1:
            offsets = jnp.broadcast_to(offsets, (self.component_to_column.shape[0], 3))
        if offsets.ndim != 2 or offsets.shape != (self.component_to_column.shape[0], 3):
            raise ValueError("magnitude_offsets must have shape (3,) or (n_component, 3)")
        absolute_reference = reference_magnitude - offsets[:, 0]
        absolute_color = color - (offsets[:, 1] - offsets[:, 2])
        return self._bilinear_all_components(absolute_reference, absolute_color)

    def log_radius_moments_all_components(
        self,
        reference_magnitude: float,
        color: float,
        magnitude_offsets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return CMD-density-weighted first and second log-radius moments."""

        offsets = jnp.asarray(magnitude_offsets)
        if offsets.ndim == 1:
            offsets = jnp.broadcast_to(offsets, (self.component_to_column.shape[0], 3))
        absolute_reference = reference_magnitude - offsets[:, 0]
        absolute_color = color - (offsets[:, 1] - offsets[:, 2])
        return (
            self._bilinear_all_components(absolute_reference, absolute_color, self.log_radius_moment_by_component),
            self._bilinear_all_components(absolute_reference, absolute_color, self.log_radius_square_moment_by_component),
        )

    def theta_star_log_density_all_components(
        self,
        reference_magnitude: float,
        color: float,
        magnitude_offsets: jnp.ndarray,
        *,
        theta_star_mas: float,
        ds_kpc: float,
    ) -> jnp.ndarray:
        """Return ``p(log thetaS | CMD, DS, component)``."""

        density = self.density_all_components(
            reference_magnitude, color, magnitude_offsets
        )
        first, second = self.log_radius_moments_all_components(
            reference_magnitude, color, magnitude_offsets
        )
        safe_density = jnp.where(density > 0.0, density, 1.0)
        mean_log_radius = first / safe_density
        variance_log_radius = jnp.maximum(
            second / safe_density - mean_log_radius**2,
            1.0e-12,
        )
        sigma = jnp.sqrt(variance_log_radius)
        mean_log_theta = (
            mean_log_radius
            + jnp.log(4.650467260962157)
            - jnp.log(ds_kpc * 1000.0)
        )
        z = (jnp.log(theta_star_mas) - mean_log_theta) / sigma
        value = jnp.exp(-0.5 * z**2) / (sigma * jnp.sqrt(2.0 * jnp.pi))
        valid = (density > 0.0) & (theta_star_mas > 0.0) & (ds_kpc > 0.0)
        return jnp.where(valid & jnp.isfinite(value), value, 0.0)

    def _bilinear_all_components(
        self,
        reference: jnp.ndarray,
        color: jnp.ndarray,
        values_by_component: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        n_reference = self.reference_centers.shape[0]
        n_color = self.color_centers.shape[0]
        i1 = jnp.clip(jnp.searchsorted(self.reference_centers, reference, side="right"), 0, n_reference - 1)
        j1 = jnp.clip(jnp.searchsorted(self.color_centers, color, side="right"), 0, n_color - 1)
        i0 = jnp.maximum(0, i1 - 1)
        j0 = jnp.maximum(0, j1 - 1)
        x0, x1 = self.reference_centers[i0], self.reference_centers[i1]
        y0, y1 = self.color_centers[j0], self.color_centers[j1]
        tx = jnp.where(i0 == i1, 0.0, (reference - x0) / (x1 - x0))
        ty = jnp.where(j0 == j1, 0.0, (color - y0) / (y1 - y0))
        columns = self.component_to_column
        valid_component = columns >= 0
        safe_columns = jnp.maximum(columns, 0)
        component = jnp.arange(self.component_to_column.shape[0])
        values = self.density_by_component if values_by_component is None else values_by_component
        table = values[safe_columns]
        value = (
            (1.0 - tx) * (1.0 - ty) * table[component, i0, j0]
            + tx * (1.0 - ty) * table[component, i1, j0]
            + (1.0 - tx) * ty * table[component, i0, j1]
            + tx * ty * table[component, i1, j1]
        )
        in_range = (
            (reference >= self.reference_centers[0])
            & (reference <= self.reference_centers[-1])
            & (color >= self.color_centers[0])
            & (color <= self.color_centers[-1])
        )
        return jnp.where(valid_component & in_range, value, 0.0)


@dataclass(frozen=True)
class MurelHistogram:
    # pairs: raw (DS, DL) block centers from murel.dat, in pc.
    pairs: jnp.ndarray
    pair_scale: jnp.ndarray
    mu_x: jnp.ndarray
    mu_y: jnp.ndarray
    mu_len: jnp.ndarray
    phi_x: jnp.ndarray
    phi_y: jnp.ndarray
    phi_len: jnp.ndarray
    source_mu_y: jnp.ndarray
    source_phi_y: jnp.ndarray
    phi_region_areas: jnp.ndarray
    phi_left_rate: jnp.ndarray
    phi_right_rate: jnp.ndarray
    phi_first: jnp.ndarray
    phi_last: jnp.ndarray
    ds_values: jnp.ndarray
    dl_values: jnp.ndarray
    grid_index: jnp.ndarray
    interpolation: str
    grid: dict[str, float]

    @classmethod
    def from_tables(
        cls,
        density: HistogramTables,
        *,
        interpolation: Literal["nearest", "bilinear"] = "nearest",
    ) -> "MurelHistogram":
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("murel interpolation must be 'nearest' or 'bilinear'")
        if density.murel.has_source_groups and interpolation == "bilinear":
            raise ValueError("bilinear interpolation is not yet available for source-group murel histograms")

        blocks = []
        for pair in density.murel.pairs:
            key = (float(pair[0]), float(pair[1]))
            block_slice = density.murel.block_slices[key]
            block = density.murel.rows[block_slice]
            mu_x, mu_y = _unique_xy(block[block[:, 2] > 0.0, 2], block[block[:, 2] > 0.0, 4])
            # ``calc_murel_dist`` emits ``max(n_mu_bins, n_phi_bins)`` rows
            # per distance pair.  The shorter marginal is padded with zero
            # coordinates.  Treating that padding as a phi bin collapses many
            # source-group rows onto one coordinate, making its length differ
            # from ``phi_x`` (and, more importantly, mixes non-phi entries
            # into the conditional source histogram).
            valid_phi = block[:, 3] != 0.0
            phi_x, phi_y = _unique_xy(block[valid_phi, 3], block[valid_phi, 5])
            valid_mu = block[:, 2] > 0.0
            if density.murel.has_source_groups:
                assert density.murel.source_group_mu is not None
                assert density.murel.source_group_phi is not None
                mu_group = density.murel.source_group_mu[block_slice][valid_mu]
                phi_group = density.murel.source_group_phi[block_slice][valid_phi]
            else:
                mu_group = np.repeat(mu_y[:, None], len(SOURCE_GROUP_NAMES), axis=1)
                phi_group = np.repeat(phi_y[:, None], len(SOURCE_GROUP_NAMES), axis=1)
            blocks.append((mu_x, mu_y, phi_x, phi_y, mu_group, phi_group))

        max_mu_len = max(2, max((len(block[0]) for block in blocks), default=1))
        max_phi_len = max(2, max((len(block[2]) for block in blocks), default=1))

        mu_x, mu_y, mu_len = _pad_blocks([(block[0], block[1]) for block in blocks], max_mu_len)
        phi_x, phi_y, phi_len = _pad_blocks([(block[2], block[3]) for block in blocks], max_phi_len)
        source_mu_y = _pad_group_blocks([block[4] for block in blocks], max_mu_len)
        source_phi_y = _pad_group_blocks([block[5] for block in blocks], max_phi_len)
        grid_index = _make_pair_grid_index(density.murel.pairs, density.murel.ds_values, density.murel.dl_values)

        phi_x_jax = jnp.asarray(phi_x)
        source_phi_y_jax = jnp.asarray(source_phi_y)
        phi_len_jax = jnp.asarray(phi_len)

        def pair_parameters(current_x, current_y, current_len):
            return vmap(
                lambda y: _phi_distribution_parameters(
                    current_x, y, current_len
                ),
                in_axes=1,
            )(current_y)

        (
            phi_region_areas,
            phi_left_rate,
            phi_right_rate,
            phi_first,
            phi_last,
        ) = vmap(pair_parameters)(phi_x_jax, source_phi_y_jax, phi_len_jax)

        return cls(
            pairs=jnp.asarray(density.murel.pairs),
            pair_scale=jnp.asarray(density.murel.pair_scale),
            mu_x=jnp.asarray(mu_x),
            mu_y=jnp.asarray(mu_y),
            mu_len=jnp.asarray(mu_len),
            phi_x=phi_x_jax,
            phi_y=jnp.asarray(phi_y),
            phi_len=jnp.asarray(phi_len),
            source_mu_y=jnp.asarray(source_mu_y),
            source_phi_y=source_phi_y_jax,
            phi_region_areas=phi_region_areas,
            phi_left_rate=phi_left_rate,
            phi_right_rate=phi_right_rate,
            phi_first=phi_first,
            phi_last=phi_last,
            ds_values=jnp.asarray(density.murel.ds_values),
            dl_values=jnp.asarray(density.murel.dl_values),
            grid_index=jnp.asarray(grid_index),
            interpolation=interpolation,
            grid=dict(density.murel.grid),
        )

    def densities(
        self,
        dl_kpc: float,
        ds_kpc: float,
        mu: float,
        phi: float,
        source_group_weights: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if source_group_weights is None:
            source_group_weights = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        if self.interpolation == "bilinear":
            p_mu, p_phi = self._bilinear_densities(dl_kpc, ds_kpc, mu, phi)
            valid = (dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0)
            return jnp.where(valid, p_mu, 0.0), jnp.where(valid, p_phi, 0.0)

        idx = self._nearest_pair_index(dl_kpc, ds_kpc)
        p_mu = _interp_padded_group(mu, self.mu_x[idx], self.source_mu_y[idx], self.mu_len[idx], source_group_weights)
        p_phi = _interp_padded_group(
            _wrap_phi(phi), self.phi_x[idx], self.source_phi_y[idx], self.phi_len[idx], source_group_weights
        )
        valid = (dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0)
        return jnp.where(valid, p_mu, 0.0), jnp.where(valid, p_phi, 0.0)

    def mu_density(
        self, dl_kpc: float, ds_kpc: float, mu: float, source_group_weights: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        if source_group_weights is None:
            source_group_weights = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        if self.interpolation == "bilinear":
            p_mu = self._bilinear_mu_density(dl_kpc, ds_kpc, mu)
            return jnp.where((dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0), p_mu, 0.0)

        idx = self._nearest_pair_index(dl_kpc, ds_kpc)
        p_mu = _interp_padded_group(mu, self.mu_x[idx], self.source_mu_y[idx], self.mu_len[idx], source_group_weights)
        return jnp.where((dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0), p_mu, 0.0)

    def mu_density_for_pair_indices(
        self, pair_indices: jnp.ndarray, mu: float, source_group_weights: jnp.ndarray
    ) -> jnp.ndarray:
        safe_indices = jnp.maximum(pair_indices, 0)

        flat_weights = source_group_weights.reshape((-1, source_group_weights.shape[-1]))

        def interp_one(idx, weights):
            return _interp_padded_group(mu, self.mu_x[idx], self.source_mu_y[idx], self.mu_len[idx], weights)

        values = vmap(interp_one)(safe_indices.ravel(), flat_weights).reshape(pair_indices.shape)
        return jnp.where(pair_indices >= 0, values, 0.0)

    def sample_phi(
        self,
        dl_kpc: float,
        ds_kpc: float,
        uniform: float,
        source_group_weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Sample the piecewise-linear conditional direction density."""
        if source_group_weights is None:
            source_group_weights = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        if self.interpolation == "bilinear":
            ds0, ds1, w_ds, in_ds = _bracket(
                self.ds_values, ds_kpc * 1000.0
            )
            dl0, dl1, w_dl, in_dl = _bracket(
                self.dl_values, dl_kpc * 1000.0
            )
            indices = jnp.asarray((
                self.grid_index[ds0, dl0],
                self.grid_index[ds0, dl1],
                self.grid_index[ds1, dl0],
                self.grid_index[ds1, dl1],
            ))
            weights = jnp.asarray((
                (1.0 - w_ds) * (1.0 - w_dl),
                (1.0 - w_ds) * w_dl,
                w_ds * (1.0 - w_dl),
                w_ds * w_dl,
            ))
            pair_weights = jnp.where(indices >= 0, weights, 0.0)
            valid = in_ds & in_dl & (jnp.sum(pair_weights) > 0.0)
        else:
            nearest = self._nearest_pair_index(dl_kpc, ds_kpc)
            indices = jnp.full((4,), nearest)
            pair_weights = jnp.asarray((1.0, 0.0, 0.0, 0.0))
            valid = True

        safe_indices = jnp.maximum(indices, 0)
        group_areas = vmap(self._phi_group_areas)(safe_indices)
        mixture = (
            pair_weights[:, None]
            * source_group_weights[None, :]
            * group_areas
        )
        component, local_uniform = _weighted_index_and_local_uniform(
            mixture.ravel(), uniform
        )
        pair_index, group_index = jnp.divmod(
            component, len(SOURCE_GROUP_NAMES)
        )
        phi = self._sample_phi_column(
            safe_indices[pair_index],
            group_index,
            local_uniform,
        )
        return jnp.where(valid, phi, 0.0)

    def _phi_group_areas(self, index):
        return jnp.sum(self.phi_region_areas[index], axis=1)

    def _sample_phi_column(self, index, group_index, uniform):
        x = self.phi_x[index]
        y = self.source_phi_y[index, :, group_index]
        first = self.phi_first[index, group_index]
        last = self.phi_last[index, group_index]
        left_rate = self.phi_left_rate[index, group_index]
        right_rate = self.phi_right_rate[index, group_index]
        left_span = jnp.maximum(x[first] + jnp.pi, 0.0)
        right_span = jnp.maximum(jnp.pi - x[last], 0.0)
        widths = x[1:] - x[:-1]
        regions = self.phi_region_areas[index, group_index]
        middle_areas = regions[1:-1]
        region, local_uniform = _weighted_index_and_local_uniform(
            regions, uniform
        )
        segment = jnp.clip(region - 1, 0, widths.shape[0] - 1)
        width = widths[segment]
        y0 = y[segment]
        slope = (y[segment + 1] - y0) / width
        target = local_uniform * middle_areas[segment]
        discriminant = jnp.maximum(y0 * y0 + 2.0 * slope * target, 0.0)
        curved = 2.0 * target / (y0 + jnp.sqrt(discriminant))
        linear = target / jnp.where(y0 > 0.0, y0, 1.0)
        offset = jnp.where(jnp.abs(slope) > 1.0e-12, curved, linear)
        middle_phi = x[segment] + jnp.clip(offset, 0.0, width)

        left_fraction = jnp.exp(-left_rate * left_span) + local_uniform * (
            1.0 - jnp.exp(-left_rate * left_span)
        )
        left_phi = x[first] + jnp.log(left_fraction) / left_rate
        right_fraction = 1.0 - local_uniform * (
            1.0 - jnp.exp(-right_rate * right_span)
        )
        right_phi = x[last] - jnp.log(right_fraction) / right_rate
        return jnp.where(
            region == 0,
            left_phi,
            jnp.where(region == regions.shape[0] - 1, right_phi, middle_phi),
        )

    def _nearest_pair_index(self, dl_kpc: float, ds_kpc: float) -> jnp.ndarray:
        target = jnp.array([ds_kpc * 1000.0, dl_kpc * 1000.0])
        return jnp.argmin(jnp.sum(((self.pairs - target) / self.pair_scale) ** 2, axis=1))

    def nearest_pair_indices(self, dl_kpc: jnp.ndarray, ds_kpc: jnp.ndarray) -> jnp.ndarray:
        targets = jnp.stack([ds_kpc.ravel() * 1000.0, dl_kpc.ravel() * 1000.0], axis=1)
        scaled = (self.pairs[None, :, :] - targets[:, None, :]) / self.pair_scale
        indices = jnp.argmin(jnp.sum(scaled * scaled, axis=2), axis=1)
        return indices.reshape(dl_kpc.shape)

    def _bilinear_densities(
        self,
        dl_kpc: float,
        ds_kpc: float,
        mu: float,
        phi: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        ds_pc = ds_kpc * 1000.0
        dl_pc = dl_kpc * 1000.0
        ds0, ds1, w_ds, in_ds = _bracket(self.ds_values, ds_pc)
        dl0, dl1, w_dl, in_dl = _bracket(self.dl_values, dl_pc)

        p00_mu, p00_phi, v00 = self._densities_at_grid(ds0, dl0, mu, phi)
        p01_mu, p01_phi, v01 = self._densities_at_grid(ds0, dl1, mu, phi)
        p10_mu, p10_phi, v10 = self._densities_at_grid(ds1, dl0, mu, phi)
        p11_mu, p11_phi, v11 = self._densities_at_grid(ds1, dl1, mu, phi)

        w00 = (1.0 - w_ds) * (1.0 - w_dl) * v00
        w01 = (1.0 - w_ds) * w_dl * v01
        w10 = w_ds * (1.0 - w_dl) * v10
        w11 = w_ds * w_dl * v11
        wsum = w00 + w01 + w10 + w11
        p_mu = w00 * p00_mu + w01 * p01_mu + w10 * p10_mu + w11 * p11_mu
        p_phi = w00 * p00_phi + w01 * p01_phi + w10 * p10_phi + w11 * p11_phi
        p_mu = jnp.where(wsum > 0.0, p_mu / wsum, 0.0)
        p_phi = jnp.where(wsum > 0.0, p_phi / wsum, 0.0)
        valid = in_ds & in_dl & (wsum > 0.0)
        return jnp.where(valid, p_mu, 0.0), jnp.where(valid, p_phi, 0.0)

    def _bilinear_mu_density(
        self,
        dl_kpc: float,
        ds_kpc: float,
        mu: float,
    ) -> jnp.ndarray:
        ds_pc = ds_kpc * 1000.0
        dl_pc = dl_kpc * 1000.0
        ds0, ds1, w_ds, in_ds = _bracket(self.ds_values, ds_pc)
        dl0, dl1, w_dl, in_dl = _bracket(self.dl_values, dl_pc)

        p00_mu, v00 = self._mu_density_at_grid(ds0, dl0, mu)
        p01_mu, v01 = self._mu_density_at_grid(ds0, dl1, mu)
        p10_mu, v10 = self._mu_density_at_grid(ds1, dl0, mu)
        p11_mu, v11 = self._mu_density_at_grid(ds1, dl1, mu)

        w00 = (1.0 - w_ds) * (1.0 - w_dl) * v00
        w01 = (1.0 - w_ds) * w_dl * v01
        w10 = w_ds * (1.0 - w_dl) * v10
        w11 = w_ds * w_dl * v11
        wsum = w00 + w01 + w10 + w11
        p_mu = w00 * p00_mu + w01 * p01_mu + w10 * p10_mu + w11 * p11_mu
        p_mu = jnp.where(wsum > 0.0, p_mu / wsum, 0.0)
        valid = in_ds & in_dl & (wsum > 0.0)
        return jnp.where(valid, p_mu, 0.0)

    def _densities_at_grid(
        self,
        ds_index: jnp.ndarray,
        dl_index: jnp.ndarray,
        mu: float,
        phi: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        idx = self.grid_index[ds_index, dl_index]
        valid = idx >= 0
        safe_idx = jnp.maximum(idx, 0)
        p_mu = _interp_padded(mu, self.mu_x[safe_idx], self.mu_y[safe_idx], self.mu_len[safe_idx])
        p_phi = _interp_padded(_wrap_phi(phi), self.phi_x[safe_idx], self.phi_y[safe_idx], self.phi_len[safe_idx])
        return jnp.where(valid, p_mu, 0.0), jnp.where(valid, p_phi, 0.0), valid

    def _mu_density_at_grid(
        self,
        ds_index: jnp.ndarray,
        dl_index: jnp.ndarray,
        mu: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        idx = self.grid_index[ds_index, dl_index]
        valid = idx >= 0
        safe_idx = jnp.maximum(idx, 0)
        p_mu = _interp_padded(mu, self.mu_x[safe_idx], self.mu_y[safe_idx], self.mu_len[safe_idx])
        return jnp.where(valid, p_mu, 0.0), valid


@dataclass(frozen=True)
class DistanceMarginalizationGrid:
    dl: jnp.ndarray
    ds: jnp.ndarray
    valid: jnp.ndarray
    pi_rel: jnp.ndarray
    weights: jnp.ndarray
    component_fractions: jnp.ndarray
    source_group_weights: jnp.ndarray
    pair_indices: jnp.ndarray


class HistogramDensity:
    """JAX histogram-backed Galactic density model.

    This backend uses the same files and probability semantics as
    `HistogramDensity`, but stores evaluation arrays as JAX arrays.
    The canonical distance unit is kpc. The underlying table data is stored
    in pc (as generated by PreRunner), and conversions are applied internally.
    """

    def __init__(
        self,
        mass: MassHistogram,
        distance: DistanceDensityTable,
        murel: MurelHistogram,
        *,
        component_names: Sequence[str] = COMPONENT_NAMES,
    ) -> None:
        self.mass = mass
        self.distance = distance
        self.murel = murel
        self.component_names = tuple(component_names)
        self._distance_marginalization_grid = None

    @classmethod
    def from_paths(
        cls,
        mass_path: str | Path,
        rho_path: str | Path,
        murel_path: str | Path,
        *,
        murel_interpolation: Literal["nearest", "bilinear"] = "nearest",
        source_evidence: SourceEvidenceGrid | None = None,
    ) -> "HistogramDensity":
        return cls.from_tables(
            HistogramTables.from_paths(
                mass_path,
                rho_path,
                murel_path,
                source_evidence=source_evidence,
            ),
            murel_interpolation=murel_interpolation,
        )

    @classmethod
    def from_pre_run(
        cls,
        pre_run_result,
        *,
        murel_interpolation: Literal["nearest", "bilinear"] = "nearest",
        source_evidence: SourceEvidenceGrid | None = None,
    ) -> "HistogramDensity":
        if source_evidence is None:
            path = getattr(pre_run_result, "source_evidence_path", None)
            if path is not None and Path(path).is_file():
                source_evidence = SourceEvidenceGrid.load_npz(path)
        return cls.from_paths(
            pre_run_result.mass_path,
            pre_run_result.rho_path,
            pre_run_result.murel_path,
            murel_interpolation=murel_interpolation,
            source_evidence=source_evidence,
        )

    @classmethod
    def from_tables(
        cls,
        density: HistogramTables,
        *,
        murel_interpolation: Literal["nearest", "bilinear"] = "nearest",
    ) -> "HistogramDensity":
        return cls(
            mass=MassHistogram.from_tables(density),
            distance=DistanceDensityTable.from_tables(density),
            murel=MurelHistogram.from_tables(density, interpolation=murel_interpolation),
            component_names=density.component_names,
        )

    def density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> jnp.ndarray:
        """Return density with respect to dML dDL dDS dmu_N dmu_E. Distances in kpc."""
        mu = jnp.hypot(mu_N, mu_E)
        phi = jnp.arctan2(mu_E, mu_N)
        val = self.density_mu_phi(ML, DL, DS, mu, phi)
        return jnp.where(mu > 0.0, val / mu, 0.0)

    def cmd_joint_density(
        self,
        ML: float,
        DL: float,
        DS: float,
        mu_N: float,
        mu_E: float,
        *,
        cmd_prior: CmdPriorEvaluator,
        reference_magnitude: float,
        color: float,
        magnitude_offsets: jnp.ndarray,
        source_component_factor: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Joint density in event variables and apparent CMD coordinates.

        ``magnitude_offsets`` is a JAX array with shape ``(3,)`` or
        ``(n_component, 3)`` ordered as ``(reference, blue, red)``. Supplying
        it as an array keeps the evaluator compatible with ``jax.jit``.
        """

        mu = jnp.hypot(mu_N, mu_E)
        photometric_density = cmd_prior.density_all_components(
            reference_magnitude,
            color,
            magnitude_offsets,
        )
        if source_component_factor is not None:
            photometric_density = photometric_density * jnp.asarray(
                source_component_factor
            )
        source_values = self.distance.source_component_values(DS)
        source_norm = jnp.where(self.distance.source_norm > 0.0, self.distance.source_norm, 1.0)
        component_values = source_values * photometric_density / source_norm
        source_density = jnp.sum(component_values)
        group_values = jnp.asarray(SOURCE_GROUP_MATRIX) @ component_values
        group_weights = jnp.where(
            source_density > 0.0,
            group_values / source_density,
            jnp.zeros_like(group_values),
        )
        phi = jnp.arctan2(mu_E, mu_N)
        p_mass = self.mass_density_given_dl(ML, DL)
        p_dl = self.distance.lens_pdf_given_source(DL, DS)
        p_mu, p_phi = self.murel.densities(DL, DS, mu, phi, group_weights)
        value = 1000.0 * p_mass * p_dl * source_density * p_mu * p_phi / mu
        valid = (ML > 0.0) & (DL > 0.0) & (mu > 0.0) & (DS > DL) & (self.distance.source_norm > 0.0)
        return jnp.where(valid, value, 0.0)

    def log_cmd_joint_density(self, *args, **kwargs) -> jnp.ndarray:
        density = self.cmd_joint_density(*args, **kwargs)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

    def log_density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> jnp.ndarray:
        density = self.density(ML, DL, DS, mu_N, mu_E)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

    def log_prior(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> jnp.ndarray:
        mu = jnp.hypot(mu_N, mu_E)
        return self.log_density(ML, DL, DS, mu_N, mu_E) + log_event_rate_backend(ML, DL, DS, mu)

    def with_source_evidence(self, evidence: SourceEvidenceGrid) -> "HistogramDensity":
        """Return a density with source evidence applied to the forward base."""

        distance_pc = np.asarray(self.distance.distance_pc)
        weights = evidence.evidence_on(distance_pc, self.distance.base_source_density_by_component.shape[1])
        source_by_component = self.distance.base_source_density_by_component * jnp.asarray(weights)
        source_density = jnp.sum(source_by_component, axis=1)
        source_norm = float(_trapz_jax(source_density, self.distance.distance_pc))
        distance = DistanceDensityTable(
            distance_pc=self.distance.distance_pc,
            lens_density_by_component=self.distance.lens_density_by_component,
            base_source_density_by_component=self.distance.base_source_density_by_component,
            source_density_by_component=source_by_component,
            source_density=source_density,
            lens_density_total=self.distance.lens_density_total,
            lens_cumulative_integral=self.distance.lens_cumulative_integral,
            source_norm=source_norm,
        )
        return HistogramDensity(
            mass=self.mass,
            distance=distance,
            murel=self.murel,
            component_names=self.component_names,
        )

    def with_genulens_source_evidence(self, builder) -> "HistogramDensity":
        evidence = builder.build(
            np.asarray(self.distance.distance_pc),
            component_indices=range(self.distance.source_density_by_component.shape[1]),
        )
        return self.with_source_evidence(evidence)

    def density_mu_phi(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> jnp.ndarray:
        """Return density with respect to dML dDL dDS dmu dphi. Distances in kpc."""
        valid = (mass > 0.0) & (dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0)
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu, p_phi = self.murel.densities(
            dl_kpc, ds_kpc, mu, phi, self.distance.source_group_weights(ds_kpc)
        )
        return jnp.where(valid, p_mass * p_dl * p_ds * p_mu * p_phi, 0.0)

    def log_density_mu_phi(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> jnp.ndarray:
        """Return log density with respect to dML dDL dDS dmu dphi. Distances in kpc."""
        density = self.density_mu_phi(mass, dl_kpc, ds_kpc, mu, phi)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

    def density_mu(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float) -> jnp.ndarray:
        """Return density with respect to dML dDL dDS dmu, marginalized over phi."""
        valid = (mass > 0.0) & (dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0)
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu = self.murel.mu_density(dl_kpc, ds_kpc, mu, self.distance.source_group_weights(ds_kpc))
        return jnp.where(valid, p_mass * p_dl * p_ds * p_mu, 0.0)

    def log_density_mu(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float) -> jnp.ndarray:
        """Return log density with respect to dML dDL dDS dmu, marginalized over phi."""
        density = self.density_mu(mass, dl_kpc, ds_kpc, mu)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

    def density_theta_mu(
        self,
        theta_e: float,
        mu: float,
        *,
        include_event_rate: bool = False,
    ) -> jnp.ndarray:
        """Return density with respect to dthetaE dmu, marginalized over DL and DS."""
        integrand, _ = self._theta_mu_terms(
            theta_e,
            mu,
            include_event_rate=include_event_rate,
        )
        density = jnp.sum(integrand)
        return jnp.where((theta_e > 0.0) & (mu > 0.0), density, 0.0)

    def log_density_theta_mu(
        self,
        theta_e: float,
        mu: float,
        *,
        include_event_rate: bool = False,
    ) -> jnp.ndarray:
        """Return log density with respect to dthetaE dmu, marginalized over DL and DS."""
        density = self.density_theta_mu(theta_e, mu, include_event_rate=include_event_rate)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

    def theta_mu_log_terms(
        self,
        theta_e: float,
        mu: float,
        *,
        include_event_rate: bool = False,
    ):
        """Return analytic distance-quadrature terms and their coordinates."""
        integrand, physical = self._theta_mu_terms(
            theta_e,
            mu,
            include_event_rate=include_event_rate,
        )
        terms = jnp.where(integrand > 0.0, jnp.log(integrand), -jnp.inf)
        return terms, physical

    def theta_mu_terms(
        self,
        theta_e: float,
        mu: float,
        *,
        include_event_rate: bool = False,
    ):
        """Return the native linear quadrature terms and coordinates."""
        return self._theta_mu_terms(
            theta_e,
            mu,
            include_event_rate=include_event_rate,
        )

    def prepare_theta_mu_integration(self):
        """Build immutable quadrature arrays before entering a JAX trace."""
        self._distance_grid()

    def sample_direction(self, dl, ds, mu, uniform):
        """Sample phi conditional on the analytically marginalized mu."""
        del mu
        group_weights = self.distance.source_group_weights(ds)
        return self.murel.sample_phi(dl, ds, uniform, group_weights)

    def _theta_mu_terms(
        self,
        theta_e: float,
        mu: float,
        *,
        include_event_rate: bool,
    ):
        grid = self._distance_grid()
        safe_pi_rel = jnp.where(grid.valid, grid.pi_rel, 1.0)
        mass = theta_e * theta_e / (KAPPA * safe_pi_rel)
        jac = 2.0 * theta_e / (KAPPA * safe_pi_rel)
        p_mass = self._mass_density_grid(mass, grid.component_fractions)
        p_mu = self.murel.mu_density_for_pair_indices(
            grid.pair_indices, mu, grid.source_group_weights
        )
        integrand = grid.weights * p_mass * p_mu * jac
        if include_event_rate:
            integrand = integrand * grid.dl * grid.dl * theta_e * mu
        valid = grid.valid & (integrand > 0.0) & (theta_e > 0.0) & (mu > 0.0)
        integrand = jnp.where(valid, integrand, 0.0)
        return integrand.ravel(), (
            mass.ravel(),
            grid.dl.ravel(),
            grid.ds.ravel(),
        )

    def _theta_mu_integrand(
        self,
        theta_e: float,
        mu: float,
        dl_kpc: float,
        ds_kpc: float,
        include_event_rate: bool,
    ) -> jnp.ndarray:
        pi_rel = (1.0 / dl_kpc) - (1.0 / ds_kpc)
        safe_pi_rel = jnp.where(pi_rel > 0.0, pi_rel, 1.0)
        mass = theta_e * theta_e / (KAPPA * safe_pi_rel)
        jac = 2.0 * theta_e / (KAPPA * safe_pi_rel)
        density = self.density_mu(mass, dl_kpc, ds_kpc, mu) * jac
        if include_event_rate:
            density = density * jnp.exp(log_event_rate_backend(mass, dl_kpc, ds_kpc, mu))
        return jnp.where(pi_rel > 0.0, density, 0.0)

    def _distance_grid(self):
        if self._distance_marginalization_grid is None:
            self._distance_marginalization_grid = self._build_distance_grid()
        return self._distance_marginalization_grid

    def _build_distance_grid(self):
        distances = self.distance.distance_pc / 1000.0
        dl, ds = jnp.meshgrid(distances, distances, indexing="ij")
        valid = dl < ds
        pi_rel = jnp.where(valid, (1.0 / dl) - (1.0 / ds), 0.0)

        one_d_weights = _trapz_weights_jax(distances)
        dl_weights, ds_weights = jnp.meshgrid(one_d_weights, one_d_weights, indexing="ij")
        p_ds = vmap(self.distance.source_pdf)(distances)
        lens_norm = vmap(self.distance._lens_integral_until)(distances)
        lens_pdf = jnp.where(
            lens_norm[None, :] > 0.0,
            self.distance.lens_density_total[:, None] / lens_norm[None, :],
            0.0,
        )
        weights = dl_weights * ds_weights * p_ds[None, :] * lens_pdf
        weights = jnp.where(valid, weights, 0.0)

        component_fractions_1d = vmap(self.distance.component_fractions)(distances)
        component_fractions = jnp.broadcast_to(
            component_fractions_1d[:, None, :],
            dl.shape + (component_fractions_1d.shape[1],),
        )
        source_group_weights_1d = vmap(self.distance.source_group_weights)(distances)
        source_group_weights = jnp.broadcast_to(
            source_group_weights_1d[None, :, :],
            dl.shape + (source_group_weights_1d.shape[1],),
        )
        pair_indices = jnp.where(valid, self.murel.nearest_pair_indices(dl, ds), -1)
        return DistanceMarginalizationGrid(
            dl=dl,
            ds=ds,
            valid=valid,
            pi_rel=pi_rel,
            weights=weights,
            component_fractions=component_fractions,
            source_group_weights=source_group_weights,
            pair_indices=pair_indices,
        )

    def _mass_density_grid(self, mass: jnp.ndarray, component_fractions: jnp.ndarray) -> jnp.ndarray:
        log_mass = jnp.where(mass > 0.0, jnp.log10(mass), 0.0)
        values = jnp.stack(
            [
                jnp.interp(
                    log_mass,
                    self.mass.log_mass,
                    self.mass.pdf_mass_by_component[:, i],
                    left=0.0,
                    right=0.0,
                )
                for i in range(self.mass.pdf_mass_by_component.shape[1])
            ],
            axis=-1,
        )
        values = jnp.where(mass[..., None] > 0.0, values, 0.0)
        return jnp.sum(values * component_fractions, axis=-1)

    def mass_density_given_dl(self, mass: float, dl_kpc: float) -> jnp.ndarray:
        p_mass_given_component = self.mass.density_given_component(mass)
        component_fraction = self.distance.component_fractions(dl_kpc)
        return jnp.sum(p_mass_given_component * component_fraction)

    def component_fractions(self, dl_kpc: float) -> dict[str, jnp.ndarray]:
        fractions = self.distance.component_fractions(dl_kpc)
        return {name: value for name, value in zip(self.component_names, fractions)}


def _unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([0.0]), np.array([0.0])
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    return unique_x, y_sorted[unique_idx]


def _trapz_jax(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))


def _trapz_weights_jax(x: jnp.ndarray) -> jnp.ndarray:
    weights = jnp.zeros_like(x)
    weights = weights.at[0].set(0.5 * (x[1] - x[0]))
    weights = weights.at[-1].set(0.5 * (x[-1] - x[-2]))
    weights = weights.at[1:-1].set(0.5 * (x[2:] - x[:-2]))
    return weights


def _weighted_index_and_local_uniform(weights, uniform):
    weights = jnp.maximum(jnp.asarray(weights), 0.0)
    total = jnp.sum(weights)
    target = jnp.clip(uniform, 0.0, 1.0 - jnp.finfo(weights.dtype).eps) * total
    cumulative = jnp.cumsum(weights)
    index = jnp.searchsorted(cumulative, target, side="right")
    index = jnp.minimum(index, weights.shape[0] - 1)
    previous = jnp.where(index > 0, cumulative[index - 1], 0.0)
    selected = weights[index]
    local = (target - previous) / jnp.where(selected > 0.0, selected, 1.0)
    return index, jnp.clip(local, 0.0, 1.0)


def _phi_distribution_parameters(x, y, valid_len):
    first, last, _, _ = _positive_sample_indices(y)
    left_rate, right_rate = _tail_rates(x, y)
    left_span = jnp.maximum(x[first] + jnp.pi, 0.0)
    right_span = jnp.maximum(jnp.pi - x[last], 0.0)
    left = y[first] * (1.0 - jnp.exp(-left_rate * left_span)) / left_rate
    right = y[last] * (1.0 - jnp.exp(-right_rate * right_span)) / right_rate
    widths = x[1:] - x[:-1]
    indices = jnp.arange(widths.shape[0])
    valid = (
        (indices >= first)
        & (indices < last)
        & (indices < valid_len - 1)
    )
    middle = jnp.where(
        valid, 0.5 * (y[:-1] + y[1:]) * widths, 0.0
    )
    regions = jnp.concatenate((
        jnp.asarray((left,)),
        middle,
        jnp.asarray((right,)),
    ))
    return regions, left_rate, right_rate, first, last


def _pad_blocks(blocks: list[tuple[np.ndarray, np.ndarray]], width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_out = np.zeros((len(blocks), width), dtype=float)
    y_out = np.zeros((len(blocks), width), dtype=float)
    lengths = np.zeros(len(blocks), dtype=int)
    for i, (x, y) in enumerate(blocks):
        n = len(x)
        lengths[i] = n
        x_out[i, :n] = x
        y_out[i, :n] = y
        if n < width:
            step = _padding_step(x)
            x_out[i, n:] = x[-1] + step * np.arange(1, width - n + 1)
    return x_out, y_out, lengths


def _pad_group_blocks(blocks: list[np.ndarray], width: int) -> np.ndarray:
    out = np.zeros((len(blocks), width, len(SOURCE_GROUP_NAMES)), dtype=float)
    for i, values in enumerate(blocks):
        out[i, : len(values)] = values
    return out


def _make_pair_grid_index(pairs: np.ndarray, ds_values: np.ndarray, dl_values: np.ndarray) -> np.ndarray:
    grid = np.full((len(ds_values), len(dl_values)), -1, dtype=int)
    ds_lookup = {float(value): index for index, value in enumerate(ds_values)}
    dl_lookup = {float(value): index for index, value in enumerate(dl_values)}
    for index, pair in enumerate(pairs):
        ds_index = ds_lookup[float(pair[0])]
        dl_index = dl_lookup[float(pair[1])]
        grid[ds_index, dl_index] = index
    return grid


def _bracket(values: jnp.ndarray, value: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = values.shape[0]
    right = jnp.searchsorted(values, value, side="right")
    right = jnp.clip(right, 1, n - 1)
    left = right - 1
    x0 = values[left]
    x1 = values[right]
    denom = jnp.where(x1 != x0, x1 - x0, 1.0)
    weight = (value - x0) / denom
    in_range = (value >= values[0]) & (value <= values[-1])
    return left, right, jnp.clip(weight, 0.0, 1.0), in_range


def _padding_step(x: np.ndarray) -> float:
    if len(x) >= 2:
        return max(float(np.nanmedian(np.diff(x))), 1.0)
    return 1.0


def _interp_padded(value: float, x: jnp.ndarray, y: jnp.ndarray, valid_len: jnp.ndarray) -> jnp.ndarray:
    first, last, left_idx, right_idx = _positive_sample_indices(y)
    below = value < x[first]
    above = value > x[last]
    right = jnp.searchsorted(x, value, side="left")
    exact_idx = jnp.clip(right, 0, last)
    exact = (right <= last) & (x[exact_idx] == value)
    right = jnp.clip(right, 1, last)
    left = right - 1
    x0 = x[left]
    x1 = x[right]
    y0 = y[left]
    y1 = y[right]
    denom = jnp.where(x1 != x0, x1 - x0, 1.0)
    interpolated = y0 + (y1 - y0) * (value - x0) / denom
    exact_value = y[exact_idx]
    interpolated = jnp.where(exact, exact_value, interpolated)
    floor = jnp.finfo(y.dtype).tiny
    def slope(sample):
        xs, ys = x[sample], jnp.log(jnp.maximum(y[sample], floor))
        centered = xs - jnp.mean(xs)
        return jnp.sum(centered * (ys - jnp.mean(ys))) / jnp.maximum(jnp.sum(centered * centered), 1e-12)
    left_step = jnp.maximum(jnp.median(jnp.diff(x[left_idx])), 1e-12)
    right_step = jnp.maximum(jnp.median(jnp.diff(x[right_idx])), 1e-12)
    left_rate = jnp.clip(slope(left_idx), 1.0 / (3.0 * left_step), 5.0 / left_step)
    right_rate = jnp.clip(-slope(right_idx), 1.0 / (3.0 * right_step), 5.0 / right_step)
    left_delta = jnp.maximum(x[first] - value, 0.0)
    right_delta = jnp.maximum(value - x[last], 0.0)
    tailed = jnp.where(
        below,
        y[first] * jnp.exp(-left_rate * left_delta),
        jnp.where(above, y[last] * jnp.exp(-right_rate * right_delta), interpolated),
    )
    return jnp.where(valid_len > 0, tailed, 0.0)


def _interp_padded_group(
    value: float,
    x: jnp.ndarray,
    y: jnp.ndarray,
    valid_len: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    values = vmap(lambda column: _interp_padded(value, x, column, valid_len), in_axes=1)(y)
    return jnp.sum(values * weights)


def _tail_rates(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    floor = jnp.finfo(y.dtype).tiny
    _, _, left_idx, right_idx = _positive_sample_indices(y)
    def slope(sample):
        xs, ys = x[sample], jnp.log(jnp.maximum(y[sample], floor))
        centered = xs - jnp.mean(xs)
        return jnp.sum(centered * (ys - jnp.mean(ys))) / jnp.maximum(jnp.sum(centered * centered), 1e-12)
    left_step = jnp.maximum(jnp.median(jnp.diff(x[left_idx])), 1e-12)
    right_step = jnp.maximum(jnp.median(jnp.diff(x[right_idx])), 1e-12)
    left = jnp.clip(slope(left_idx), 1.0 / (3.0 * left_step), 5.0 / left_step)
    right = jnp.clip(-slope(right_idx), 1.0 / (3.0 * right_step), 5.0 / right_step)
    return left, right


def _positive_sample_indices(y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """First/last and the first/last three *positive* table indices."""
    positive = y > 0.0
    indices = jnp.arange(y.shape[0])
    first = jnp.argmax(positive)
    last = y.shape[0] - 1 - jnp.argmax(positive[::-1])
    left = jnp.minimum(jnp.sort(jnp.where(positive, indices, y.shape[0]))[:3], last)
    right = jnp.maximum(jnp.sort(jnp.where(positive, indices, -1))[-3:], first)
    return first, last, left, right


def _interp_positive_tail(value: float, x: jnp.ndarray, y: jnp.ndarray, *, lower: float | None = None) -> jnp.ndarray:
    left_rate, right_rate = _tail_rates(x, y)
    first, last, _, _ = _positive_sample_indices(y)
    x0, xn, y0, yn = x[first], x[last], y[first], y[last]
    left_delta = jnp.maximum(x0 - value, 0.0)
    right_delta = jnp.maximum(value - xn, 0.0)
    left_tail = y0 * jnp.exp(-left_rate * left_delta)
    right_tail = yn * jnp.exp(-right_rate * right_delta)
    result = jnp.where(value < x0, left_tail, jnp.where(value > xn, right_tail, jnp.interp(value, x, y)))
    return jnp.where(value > lower, result, 0.0) if lower is not None else result


def _interp_mass_tail(value: float, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    support = y > 0.0
    first, last, _, _ = _positive_sample_indices(y)
    x0, xn, y0, yn = x[first], x[last], y[first], y[last]
    left_rate, right_rate = _tail_rates(x, y)
    right_rate = jnp.maximum(right_rate, jnp.log(10.0) + 1e-12)
    left_delta = jnp.maximum(x0 - value, 0.0)
    right_delta = jnp.maximum(value - xn, 0.0)
    left_tail = y0 * jnp.exp(-left_rate * left_delta)
    right_tail = yn * jnp.exp(-right_rate * right_delta)
    result = jnp.where(value < x0, left_tail, jnp.where(value > xn, right_tail, jnp.interp(value, x, y)))
    return jnp.where(jnp.any(support), result, 0.0)


def _wrap_phi(phi: float) -> jnp.ndarray:
    return jnp.mod(phi + pi, 2.0 * pi) - pi
