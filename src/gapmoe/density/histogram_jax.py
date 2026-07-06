from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Literal, Sequence

import jax.numpy as jnp
import numpy as np
from jax import vmap

from gapmoe.density.histogram_numpy import COMPONENT_NAMES, HistogramDensity
from gapmoe.priors.event_rate import KAPPA
from gapmoe.priors.event_rate_jax import jax_log_event_rate


@dataclass(frozen=True)
class JaxMassHistogram:
    log_mass: jnp.ndarray
    pdf_mass_by_component: jnp.ndarray

    @classmethod
    def from_numpy(cls, density: HistogramDensity) -> "JaxMassHistogram":
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
class JaxDistanceDensityTable:
    # distance_pc: raw distance grid from rho.dat, in pc.
    distance_pc: jnp.ndarray
    lens_density_by_component: jnp.ndarray
    source_density: jnp.ndarray
    lens_density_total: jnp.ndarray
    lens_cumulative_integral: jnp.ndarray
    source_norm: float

    @classmethod
    def from_numpy(cls, density: HistogramDensity) -> "JaxDistanceDensityTable":
        return cls(
            distance_pc=jnp.asarray(density.distance.distance_pc),
            lens_density_by_component=jnp.asarray(density.distance.lens_density_by_component),
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
        lower_total = self.lens_density_total[0] / left_rate * (1.0 - jnp.exp(-left_rate * jnp.maximum(self.distance_pc[0], 0.0)))
        tail = self.lens_density_total[-1] / right_rate * (1.0 - jnp.exp(-right_rate * (ds_pc - self.distance_pc[-1])))
        value = jnp.where(ds_pc <= 0.0, 0.0, lower_total + value)
        value = jnp.where(ds_pc < self.distance_pc[0], lower_total - self.lens_density_total[0] / left_rate * jnp.exp(-left_rate * (self.distance_pc[0] - ds_pc)), value)
        value = jnp.where(ds_pc >= self.distance_pc[-1], lower_total + self.lens_cumulative_integral[-1] + tail, value)
        return value

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


@dataclass(frozen=True)
class JaxMurelHistogram:
    # pairs: raw (DS, DL) block centers from murel.dat, in pc.
    pairs: jnp.ndarray
    pair_scale: jnp.ndarray
    mu_x: jnp.ndarray
    mu_y: jnp.ndarray
    mu_len: jnp.ndarray
    phi_x: jnp.ndarray
    phi_y: jnp.ndarray
    phi_len: jnp.ndarray
    ds_values: jnp.ndarray
    dl_values: jnp.ndarray
    grid_index: jnp.ndarray
    interpolation: str
    grid: dict[str, float]

    @classmethod
    def from_numpy(
        cls,
        density: HistogramDensity,
        *,
        interpolation: Literal["nearest", "bilinear"] = "nearest",
    ) -> "JaxMurelHistogram":
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("murel interpolation must be 'nearest' or 'bilinear'")

        blocks = []
        for pair in density.murel.pairs:
            key = (float(pair[0]), float(pair[1]))
            block_slice = density.murel.block_slices[key]
            block = density.murel.rows[block_slice]
            mu_x, mu_y = _unique_xy(block[block[:, 2] > 0.0, 2], block[block[:, 2] > 0.0, 4])
            phi_x, phi_y = _unique_xy(block[:, 3], block[:, 5])
            blocks.append((mu_x, mu_y, phi_x, phi_y))

        max_mu_len = max(2, max((len(block[0]) for block in blocks), default=1))
        max_phi_len = max(2, max((len(block[2]) for block in blocks), default=1))

        mu_x, mu_y, mu_len = _pad_blocks([(block[0], block[1]) for block in blocks], max_mu_len)
        phi_x, phi_y, phi_len = _pad_blocks([(block[2], block[3]) for block in blocks], max_phi_len)
        grid_index = _make_pair_grid_index(density.murel.pairs, density.murel.ds_values, density.murel.dl_values)

        return cls(
            pairs=jnp.asarray(density.murel.pairs),
            pair_scale=jnp.asarray(density.murel.pair_scale),
            mu_x=jnp.asarray(mu_x),
            mu_y=jnp.asarray(mu_y),
            mu_len=jnp.asarray(mu_len),
            phi_x=jnp.asarray(phi_x),
            phi_y=jnp.asarray(phi_y),
            phi_len=jnp.asarray(phi_len),
            ds_values=jnp.asarray(density.murel.ds_values),
            dl_values=jnp.asarray(density.murel.dl_values),
            grid_index=jnp.asarray(grid_index),
            interpolation=interpolation,
            grid=dict(density.murel.grid),
        )

    def densities(self, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.interpolation == "bilinear":
            p_mu, p_phi = self._bilinear_densities(dl_kpc, ds_kpc, mu, phi)
            valid = (dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0)
            return jnp.where(valid, p_mu, 0.0), jnp.where(valid, p_phi, 0.0)

        idx = self._nearest_pair_index(dl_kpc, ds_kpc)
        p_mu = _interp_padded(mu, self.mu_x[idx], self.mu_y[idx], self.mu_len[idx])
        p_phi = _interp_padded(_wrap_phi(phi), self.phi_x[idx], self.phi_y[idx], self.phi_len[idx], tails=False)
        valid = (dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0)
        return jnp.where(valid, p_mu, 0.0), jnp.where(valid, p_phi, 0.0)

    def mu_density(self, dl_kpc: float, ds_kpc: float, mu: float) -> jnp.ndarray:
        if self.interpolation == "bilinear":
            p_mu = self._bilinear_mu_density(dl_kpc, ds_kpc, mu)
            return jnp.where(ds_kpc > dl_kpc, p_mu, 0.0)

        idx = self._nearest_pair_index(dl_kpc, ds_kpc)
        p_mu = _interp_padded(mu, self.mu_x[idx], self.mu_y[idx], self.mu_len[idx])
        return jnp.where(ds_kpc > dl_kpc, p_mu, 0.0)

    def mu_density_for_pair_indices(self, pair_indices: jnp.ndarray, mu: float) -> jnp.ndarray:
        safe_indices = jnp.maximum(pair_indices, 0)

        def interp_one(idx):
            return _interp_padded(mu, self.mu_x[idx], self.mu_y[idx], self.mu_len[idx])

        values = vmap(interp_one)(safe_indices.ravel()).reshape(pair_indices.shape)
        return jnp.where(pair_indices >= 0, values, 0.0)

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
        p_phi = _interp_padded(_wrap_phi(phi), self.phi_x[safe_idx], self.phi_y[safe_idx], self.phi_len[safe_idx], tails=False)
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
class JaxDistanceMarginalizationGrid:
    dl: jnp.ndarray
    ds: jnp.ndarray
    valid: jnp.ndarray
    pi_rel: jnp.ndarray
    weights: jnp.ndarray
    component_fractions: jnp.ndarray
    pair_indices: jnp.ndarray


class JaxHistogramDensity:
    """JAX histogram-backed Galactic density model.

    This backend uses the same files and probability semantics as
    `HistogramDensity`, but stores evaluation arrays as JAX arrays.
    The canonical distance unit is kpc. The underlying table data is stored
    in pc (as generated by PreRunner), and conversions are applied internally.
    """

    def __init__(
        self,
        mass: JaxMassHistogram,
        distance: JaxDistanceDensityTable,
        murel: JaxMurelHistogram,
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
        require_source_selection: bool = True,
        murel_interpolation: Literal["nearest", "bilinear"] = "nearest",
    ) -> "JaxHistogramDensity":
        return cls.from_numpy(
            HistogramDensity.from_paths(
                mass_path,
                rho_path,
                murel_path,
                require_source_selection=require_source_selection,
            ),
            murel_interpolation=murel_interpolation,
        )

    @classmethod
    def from_pre_run(
        cls,
        pre_run_result,
        *,
        require_source_selection: bool = True,
        murel_interpolation: Literal["nearest", "bilinear"] = "nearest",
    ) -> "JaxHistogramDensity":
        return cls.from_paths(
            pre_run_result.mass_path,
            pre_run_result.rho_path,
            pre_run_result.murel_path,
            require_source_selection=require_source_selection,
            murel_interpolation=murel_interpolation,
        )

    @classmethod
    def from_numpy(
        cls,
        density: HistogramDensity,
        *,
        murel_interpolation: Literal["nearest", "bilinear"] = "nearest",
    ) -> "JaxHistogramDensity":
        return cls(
            mass=JaxMassHistogram.from_numpy(density),
            distance=JaxDistanceDensityTable.from_numpy(density),
            murel=JaxMurelHistogram.from_numpy(density, interpolation=murel_interpolation),
            component_names=density.component_names,
        )

    def density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> jnp.ndarray:
        """Return density with respect to dML dDL dDS dmu_N dmu_E. Distances in kpc."""
        mu = jnp.hypot(mu_N, mu_E)
        phi = jnp.arctan2(mu_E, mu_N)
        val = self.density_mu_phi(ML, DL, DS, mu, phi)
        return jnp.where(mu > 0.0, val / mu, 0.0)

    def log_density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> jnp.ndarray:
        density = self.density(ML, DL, DS, mu_N, mu_E)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

    def density_mu_phi(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> jnp.ndarray:
        """Return density with respect to dML dDL dDS dmu dphi. Distances in kpc."""
        valid = (mass > 0.0) & (dl_kpc > 0.0) & (ds_kpc > dl_kpc) & (mu > 0.0)
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu, p_phi = self.murel.densities(dl_kpc, ds_kpc, mu, phi)
        return jnp.where(valid, p_mass * p_dl * p_ds * p_mu * p_phi, 0.0)

    def log_density_mu_phi(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> jnp.ndarray:
        """Return log density with respect to dML dDL dDS dmu dphi. Distances in kpc."""
        density = self.density_mu_phi(mass, dl_kpc, ds_kpc, mu, phi)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

    def density_mu(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float) -> jnp.ndarray:
        """Return density with respect to dML dDL dDS dmu, marginalized over phi."""
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu = self.murel.mu_density(dl_kpc, ds_kpc, mu)
        return p_mass * p_dl * p_ds * p_mu

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
        grid = self._distance_grid()
        safe_pi_rel = jnp.where(grid.valid, grid.pi_rel, 1.0)
        mass = theta_e * theta_e / (KAPPA * safe_pi_rel)
        jac = 2.0 * theta_e / (KAPPA * safe_pi_rel)
        p_mass = self._mass_density_grid(mass, grid.component_fractions)
        p_mu = self.murel.mu_density_for_pair_indices(grid.pair_indices, mu)
        integrand = grid.weights * p_mass * p_mu * jac
        if include_event_rate:
            integrand = integrand * grid.dl * grid.dl * theta_e * mu
        density = jnp.sum(jnp.where(grid.valid, integrand, 0.0))
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
            density = density * jnp.exp(jax_log_event_rate(mass, dl_kpc, ds_kpc, mu))
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
        pair_indices = jnp.where(valid, self.murel.nearest_pair_indices(dl, ds), -1)
        return JaxDistanceMarginalizationGrid(
            dl=dl,
            ds=ds,
            valid=valid,
            pi_rel=pi_rel,
            weights=weights,
            component_fractions=component_fractions,
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


def _interp_padded(value: float, x: jnp.ndarray, y: jnp.ndarray, valid_len: jnp.ndarray, *, tails: bool = True) -> jnp.ndarray:
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
    def slope(sample):
        xs, ys = x[sample], jnp.log(jnp.maximum(y[sample], jnp.finfo(y.dtype).tiny))
        centered = xs - jnp.mean(xs)
        return jnp.sum(centered * (ys - jnp.mean(ys))) / jnp.maximum(jnp.sum(centered * centered), 1e-12)
    left_step = jnp.maximum(jnp.median(jnp.diff(x[left_idx])), 1e-12)
    right_step = jnp.maximum(jnp.median(jnp.diff(x[right_idx])), 1e-12)
    left_rate = jnp.clip(slope(left_idx), 1.0 / (3.0 * left_step), 5.0 / left_step)
    right_rate = jnp.clip(-slope(right_idx), 1.0 / (3.0 * right_step), 5.0 / right_step)
    tailed = jnp.where(below, y[first] * jnp.exp(-left_rate * (x[first] - value)), jnp.where(above, y[last] * jnp.exp(-right_rate * (value - x[last])), interpolated))
    result = tailed if tails else jnp.where(below | above, 0.0, interpolated)
    return jnp.where(valid_len > 0, result, 0.0)


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
    interior = jnp.interp(value, x, y)
    below = y0 * jnp.exp(-left_rate * (x0 - value))
    above = yn * jnp.exp(-right_rate * (value - xn))
    result = jnp.where(value < x0, below, jnp.where(value > xn, above, interior))
    return jnp.where(value > lower, result, 0.0) if lower is not None else result


def _interp_mass_tail(value: float, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    support = y > 0.0
    first, last, _, _ = _positive_sample_indices(y)
    x0, xn = x[first], x[last]
    y0, yn = y[first], y[last]
    left_rate, right_rate = _tail_rates(x, y)
    right_rate = jnp.maximum(right_rate, jnp.log(10.0) + 1e-12)
    interior = jnp.interp(value, x, y)
    below = y0 * jnp.exp(-left_rate * (x0 - value))
    above = yn * jnp.exp(-right_rate * (value - xn))
    result = jnp.where(value < x0, below, jnp.where(value > xn, above, interior))
    return jnp.where(jnp.any(support), result, 0.0)


def _wrap_phi(phi: float) -> jnp.ndarray:
    return jnp.mod(phi + pi, 2.0 * pi) - pi
