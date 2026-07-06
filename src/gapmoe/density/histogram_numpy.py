from __future__ import annotations

import re
from dataclasses import dataclass
from math import atan2, hypot, isfinite, log, pi
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from gapmoe.density.base import DensityModel
from gapmoe.priors.event_rate import KAPPA, log_event_rate


COMPONENT_NAMES = (
    "thin_disk_0",
    "thin_disk_1",
    "thin_disk_2",
    "thin_disk_3",
    "thin_disk_4",
    "thin_disk_5",
    "thin_disk_6",
    "thick_disk",
    "bulge",
    "NSD",
    "halo",
)


@dataclass(frozen=True)
class MassHistogram:
    log_mass: np.ndarray
    pdf_mass_by_component: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path) -> "MassHistogram":
        data = _load_2d(path)
        log_mass = data[:, 0]
        component_density_logm = data[:, 1:12]
        mass = np.power(10.0, log_mass)
        pdf_mass = np.zeros_like(component_density_logm)

        for i in range(component_density_logm.shape[1]):
            integral_logm = _trapz(component_density_logm[:, i], log_mass)
            if integral_logm <= 0.0:
                continue
            pdf_logm = component_density_logm[:, i] / integral_logm
            pdf_mass[:, i] = pdf_logm / (mass * np.log(10.0))
            # Include the same exponentially decaying tails used at evaluation
            # time in the normalisation of this physical-mass PDF.
            normalisation = _integral_with_tails(log_mass, pdf_mass[:, i], lower=None, measure=mass)
            if normalisation > 0.0:
                pdf_mass[:, i] /= normalisation

        return cls(log_mass=log_mass, pdf_mass_by_component=pdf_mass)

    def density_given_component(self, mass: float) -> np.ndarray:
        if mass <= 0.0:
            return np.zeros(self.pdf_mass_by_component.shape[1])
        log_mass = log(mass) / log(10.0)
        return np.array(
            [
                _interp_mass_tail(log_mass, self.log_mass, self.pdf_mass_by_component[:, i])
                for i in range(self.pdf_mass_by_component.shape[1])
            ]
        )


@dataclass(frozen=True)
class DistanceDensityTable:
    # distance_pc: raw distance grid from rho.dat, in pc.
    distance_pc: np.ndarray
    lens_density_by_component: np.ndarray
    source_density: np.ndarray
    lens_density_total: np.ndarray
    lens_cumulative_integral: np.ndarray
    source_norm: float

    @classmethod
    def from_file(cls, path: str | Path, *, require_source_selection: bool = True) -> "DistanceDensityTable":
        data = _load_2d(path)
        distance_pc = data[:, 0]

        if data.shape[1] < 25:
            raise ValueError(f"rho table has {data.shape[1]} columns, expected at least 25: {path}")
        if require_source_selection and data.shape[1] < 37:
            raise ValueError(
                "rho table does not contain rhoD_S source-density columns. "
                "Run PreRunner/calc_rho_profile with SOURCE=1 so p(DS) is "
                f"conditioned on observed source selection: {path}"
            )

        lens_density_by_component = data[:, 1:12]
        source_density = data[:, 36] if data.shape[1] >= 37 else data[:, 12]
        lens_density_total = lens_density_by_component.sum(axis=1)
        return cls(
            distance_pc=distance_pc,
            lens_density_by_component=lens_density_by_component,
            source_density=source_density,
            lens_density_total=lens_density_total,
            lens_cumulative_integral=_cumulative_trapezoid(distance_pc, lens_density_total),
            source_norm=_integral_with_tails(distance_pc, source_density, lower=0.0),
        )

    def source_pdf(self, ds_kpc: float) -> float:
        if self.source_norm <= 0.0:
            return 0.0
        if ds_kpc <= 0.0:
            return 0.0
        val = _interp_positive_tail(ds_kpc * 1000.0, self.distance_pc, self.source_density, lower=0.0)
        return float(1000.0 * val / self.source_norm)

    def source_pdf_array(self, ds_kpc: np.ndarray) -> np.ndarray:
        if self.source_norm <= 0.0:
            return np.zeros_like(ds_kpc, dtype=float)
        val = np.interp(ds_kpc * 1000.0, self.distance_pc, self.source_density, left=0.0, right=0.0)
        return val / self.source_norm

    def lens_pdf_given_source(self, dl_kpc: float, ds_kpc: float) -> float:
        if dl_kpc <= 0.0 or ds_kpc <= dl_kpc:
            return 0.0
        norm = self._lens_integral_until(ds_kpc)
        if norm <= 0.0:
            return 0.0
        val = _interp_positive_tail(dl_kpc * 1000.0, self.distance_pc, self.lens_density_total, lower=0.0)
        return float(1000.0 * val / norm)

    def lens_pdf_given_source_array(self, dl_kpc: np.ndarray, ds_kpc: np.ndarray) -> np.ndarray:
        dl_kpc, ds_kpc = np.broadcast_arrays(dl_kpc, ds_kpc)
        norm = self._lens_integral_until_array(ds_kpc)
        val = np.interp(dl_kpc * 1000.0, self.distance_pc, self.lens_density_total, left=0.0, right=0.0)
        out = np.zeros_like(dl_kpc, dtype=float)
        valid = (ds_kpc > dl_kpc) & (norm > 0.0)
        np.divide(val, norm, out=out, where=valid)
        return out

    def _lens_integral_until(self, ds_kpc: float) -> float:
        ds_pc = ds_kpc * 1000.0
        if ds_pc <= 0.0:
            return 0.0
        if ds_pc >= self.distance_pc[-1]:
            return _integral_with_tails(self.distance_pc, self.lens_density_total, lower=0.0, upper=ds_pc)
        base = float(np.interp(ds_pc, self.distance_pc, self.lens_cumulative_integral))
        left_idx = int(np.searchsorted(self.distance_pc, ds_pc, side="right")) - 1
        if left_idx < 0 or self.distance_pc[left_idx] == ds_pc:
            return base
        x0 = self.distance_pc[left_idx]
        y0 = self.lens_density_total[left_idx]
        y1 = _interp_positive_tail(ds_pc, self.distance_pc, self.lens_density_total, lower=0.0)
        partial = 0.5 * (y0 + y1) * (ds_pc - x0)
        return float(self.lens_cumulative_integral[left_idx] + partial)

    def _lens_integral_until_array(self, ds_kpc: np.ndarray) -> np.ndarray:
        ds_pc = np.asarray(ds_kpc, dtype=float) * 1000.0
        out = np.interp(
            ds_pc,
            self.distance_pc,
            self.lens_cumulative_integral,
            left=0.0,
            right=float(self.lens_cumulative_integral[-1]),
        )
        left_idx = np.searchsorted(self.distance_pc, ds_pc, side="right") - 1
        partial = (
            (ds_pc > self.distance_pc[0])
            & (ds_pc < self.distance_pc[-1])
            & (left_idx >= 0)
            & (left_idx < len(self.distance_pc))
            & (self.distance_pc[left_idx] != ds_pc)
        )
        if np.any(partial):
            x0 = self.distance_pc[left_idx[partial]]
            y0 = self.lens_density_total[left_idx[partial]]
            y1 = np.interp(ds_pc[partial], self.distance_pc, self.lens_density_total, left=0.0, right=0.0)
            out[partial] = (
                self.lens_cumulative_integral[left_idx[partial]]
                + 0.5 * (y0 + y1) * (ds_pc[partial] - x0)
            )
        return out

    def component_fractions(self, dl_kpc: float) -> np.ndarray:
        if dl_kpc <= 0.0:
            return np.zeros(self.lens_density_by_component.shape[1])
        vals = np.array(
            [
                _interp_positive_tail(dl_kpc * 1000.0, self.distance_pc, self.lens_density_by_component[:, i], lower=0.0)
                for i in range(self.lens_density_by_component.shape[1])
            ]
        )
        total = vals.sum()
        if total <= 0.0:
            return np.zeros_like(vals)
        return vals / total

    def component_fractions_array(self, dl_kpc: np.ndarray) -> np.ndarray:
        dl_kpc = np.asarray(dl_kpc, dtype=float)
        vals = np.zeros(dl_kpc.shape + (self.lens_density_by_component.shape[1],), dtype=float)
        for i in range(self.lens_density_by_component.shape[1]):
            vals[..., i] = np.interp(
                dl_kpc * 1000.0,
                self.distance_pc,
                self.lens_density_by_component[:, i],
                left=0.0,
                right=0.0,
            )
        total = np.sum(vals, axis=-1)
        out = np.zeros_like(vals)
        np.divide(vals, total[..., None], out=out, where=total[..., None] > 0.0)
        return out


@dataclass(frozen=True)
class MurelHistogram:
    # pairs: raw (DS, DL) block centers from murel.dat, in pc.
    rows: np.ndarray
    pairs: np.ndarray
    block_slices: dict[tuple[float, float], slice]
    ds_values: np.ndarray
    dl_values: np.ndarray
    pair_scale: np.ndarray
    grid: dict[str, float]

    @classmethod
    def from_file(cls, path: str | Path) -> "MurelHistogram":
        grid = _parse_murel_grid(path)
        rows = _load_2d(path)
        if rows.shape[1] < 6:
            raise ValueError(f"murel table has {rows.shape[1]} columns, expected at least 6: {path}")
        pairs, block_slices = _build_pair_blocks(rows)
        ds_values = np.unique(pairs[:, 0])
        dl_values = np.unique(pairs[:, 1])
        pair_scale = np.array([max(np.ptp(ds_values), 1.0), max(np.ptp(dl_values), 1.0)])
        return cls(
            rows=rows,
            pairs=pairs,
            block_slices=block_slices,
            ds_values=ds_values,
            dl_values=dl_values,
            pair_scale=pair_scale,
            grid=grid,
        )

    def densities(self, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> tuple[float, float]:
        if dl_kpc <= 0.0 or ds_kpc <= dl_kpc or mu <= 0.0:
            return 0.0, 0.0
        return self.nearest_densities(dl_kpc, ds_kpc, mu, phi)

    def mu_density(self, dl_kpc: float, ds_kpc: float, mu: float) -> float:
        if ds_kpc <= dl_kpc:
            return 0.0
        pair_pc = self._nearest_pair(dl_kpc * 1000.0, ds_kpc * 1000.0)
        return self._mu_density_for_pair(pair_pc, mu)

    def mu_density_for_pair_indices(self, pair_indices: np.ndarray, mu: float) -> np.ndarray:
        out = np.zeros(pair_indices.shape, dtype=float)
        for idx in np.unique(pair_indices[pair_indices >= 0]):
            pair = self.pairs[int(idx)]
            out[pair_indices == idx] = self._mu_density_for_pair(pair, mu)
        return out

    def densities_for_pair_indices(
        self,
        pair_indices: np.ndarray,
        mu: np.ndarray,
        phi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        p_mu = np.zeros(pair_indices.shape, dtype=float)
        p_phi = np.zeros(pair_indices.shape, dtype=float)
        phi = _wrap_phi_array(phi)
        for idx in np.unique(pair_indices[pair_indices >= 0]):
            mask = pair_indices == idx
            pair = self.pairs[int(idx)]
            key = (float(pair[0]), float(pair[1]))
            block_slice = self.block_slices.get(key)
            if block_slice is None:
                continue
            block = self.rows[block_slice]
            mu_x = block[:, 2]
            mu_y = block[:, 4]
            valid_mu = mu_x > 0.0
            p_mu[mask] = _interp_unique_array(mu[mask], mu_x[valid_mu], mu_y[valid_mu])
            p_phi[mask] = _interp_unique_array(phi[mask], block[:, 3], block[:, 5])
        return p_mu, p_phi


    def nearest_densities(self, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> tuple[float, float]:
        if dl_kpc <= 0.0 or ds_kpc <= dl_kpc or mu <= 0.0:
            return 0.0, 0.0
        pair_pc = self._nearest_pair(dl_kpc * 1000.0, ds_kpc * 1000.0)
        return self._densities_for_pair(pair_pc, mu, phi)

    def nearest_pair(self, dl_kpc: float, ds_kpc: float) -> tuple[float, float]:
        """Return the nearest tabulated (DS, DL) block center in kpc."""
        pair_pc = self._nearest_pair(dl_kpc * 1000.0, ds_kpc * 1000.0)
        return float(pair_pc[0]) / 1000.0, float(pair_pc[1]) / 1000.0

    def _densities_for_pair(self, pair: np.ndarray, mu: float, phi: float) -> tuple[float, float]:
        key = (float(pair[0]), float(pair[1]))
        block_slice = self.block_slices.get(key)
        if block_slice is None:
            return 0.0, 0.0
        block = self.rows[block_slice]

        mu_x = block[:, 2]
        mu_y = block[:, 4]
        valid_mu = mu_x > 0.0
        p_mu = _interp_positive_tail(mu, mu_x[valid_mu], mu_y[valid_mu], lower=0.0)

        phi_x = block[:, 3]
        phi_y = block[:, 5]
        p_phi = _interp_unique(_wrap_phi(phi), phi_x, phi_y)
        return p_mu, p_phi

    def _mu_density_for_pair(self, pair: np.ndarray, mu: float) -> float:
        key = (float(pair[0]), float(pair[1]))
        block_slice = self.block_slices.get(key)
        if block_slice is None:
            return 0.0
        block = self.rows[block_slice]

        mu_x = block[:, 2]
        mu_y = block[:, 4]
        valid_mu = mu_x > 0.0
        return _interp_unique(mu, mu_x[valid_mu], mu_y[valid_mu])

    def _nearest_pair(self, dl_pc: float, ds_pc: float) -> np.ndarray:
        target = np.array([ds_pc, dl_pc])
        idx = np.argmin(np.sum(((self.pairs - target) / self.pair_scale) ** 2, axis=1))
        return self.pairs[idx]

    def nearest_pair_indices(self, dl_kpc: np.ndarray, ds_kpc: np.ndarray) -> np.ndarray:
        targets = np.stack([ds_kpc.ravel() * 1000.0, dl_kpc.ravel() * 1000.0], axis=1)
        scaled = (self.pairs[None, :, :] - targets[:, None, :]) / self.pair_scale
        indices = np.argmin(np.sum(scaled * scaled, axis=2), axis=1)
        return indices.reshape(dl_kpc.shape)


@dataclass(frozen=True)
class DistanceMarginalizationGrid:
    dl: np.ndarray
    ds: np.ndarray
    valid: np.ndarray
    pi_rel: np.ndarray
    weights: np.ndarray
    component_fractions: np.ndarray
    pair_indices: np.ndarray


class HistogramDensity(DensityModel):
    """Histogram-backed Galactic density model.

    This reads the current `PreRunner` outputs: `mass.dat`, `rho.dat`, and
    `murel.dat`. The canonical distance unit is kpc. The underlying table data
    is stored in pc (as generated by PreRunner), and conversions are applied
    internally when evaluating probabilities.
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
        self._distance_marginalization_grid: DistanceMarginalizationGrid | None = None

    @classmethod
    def from_paths(
        cls,
        mass_path: str | Path,
        rho_path: str | Path,
        murel_path: str | Path,
        *,
        require_source_selection: bool = True,
    ) -> "HistogramDensity":
        return cls(
            mass=MassHistogram.from_file(mass_path),
            distance=DistanceDensityTable.from_file(rho_path, require_source_selection=require_source_selection),
            murel=MurelHistogram.from_file(murel_path),
        )

    @classmethod
    def from_pre_run(cls, pre_run_result, *, require_source_selection: bool = True) -> "HistogramDensity":
        return cls.from_paths(
            pre_run_result.mass_path,
            pre_run_result.rho_path,
            pre_run_result.murel_path,
            require_source_selection=require_source_selection,
        )

    def log_density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> float:
        density = self.density(ML, DL, DS, mu_N, mu_E)
        if density <= 0.0 or not isfinite(density):
            return float("-inf")
        return log(density)

    def density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> float:
        """Return density with respect to dML dDL dDS dmu_N dmu_E. Distances in kpc."""
        mu = hypot(mu_N, mu_E)
        if mu <= 0.0:
            return 0.0
        phi = atan2(mu_E, mu_N)
        return self.density_mu_phi(ML, DL, DS, mu, phi) / mu

    def density_array(
        self,
        ML: np.ndarray,
        DL: np.ndarray,
        DS: np.ndarray,
        mu_N: np.ndarray,
        mu_E: np.ndarray,
    ) -> np.ndarray:
        """Vectorized density with respect to dML dDL dDS dmu_N dmu_E."""
        ML, DL, DS, mu_N, mu_E = np.broadcast_arrays(ML, DL, DS, mu_N, mu_E)
        mu = np.hypot(mu_N, mu_E)
        phi = np.arctan2(mu_E, mu_N)
        component_fractions = self.distance.component_fractions_array(DL)
        p_mass = self._mass_density_grid(ML, component_fractions)
        p_dl = self.distance.lens_pdf_given_source_array(DL, DS)
        p_ds = self.distance.source_pdf_array(DS)
        pair_indices = np.where(DS > DL, self.murel.nearest_pair_indices(DL, DS), -1)
        p_mu, p_phi = self.murel.densities_for_pair_indices(pair_indices, mu, phi)
        density = p_mass * p_dl * p_ds * p_mu * p_phi
        return np.where(mu > 0.0, density / mu, 0.0)

    def log_density_array(
        self,
        ML: np.ndarray,
        DL: np.ndarray,
        DS: np.ndarray,
        mu_N: np.ndarray,
        mu_E: np.ndarray,
    ) -> np.ndarray:
        density = self.density_array(ML, DL, DS, mu_N, mu_E)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_density = np.log(density)
        return np.where((density > 0.0) & np.isfinite(density), log_density, -np.inf)

    def density_mu_phi(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> float:
        """Return density with respect to dML dDL dDS dmu dphi. Distances in kpc."""
        if mass <= 0.0 or dl_kpc <= 0.0 or ds_kpc <= dl_kpc or mu <= 0.0:
            return 0.0
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu, p_phi = self.murel.densities(dl_kpc, ds_kpc, mu, phi)
        return p_mass * p_dl * p_ds * p_mu * p_phi

    def log_density_mu_phi(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> float:
        """Return log density with respect to dML dDL dDS dmu dphi. Distances in kpc."""
        density = self.density_mu_phi(mass, dl_kpc, ds_kpc, mu, phi)
        if density <= 0.0 or not isfinite(density):
            return float("-inf")
        return log(density)

    def density_mu(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float) -> float:
        """Return density with respect to dML dDL dDS dmu, marginalized over phi."""
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu = self.murel.mu_density(dl_kpc, ds_kpc, mu)
        return p_mass * p_dl * p_ds * p_mu

    def log_density_mu(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float) -> float:
        """Return log density with respect to dML dDL dDS dmu, marginalized over phi."""
        density = self.density_mu(mass, dl_kpc, ds_kpc, mu)
        if density <= 0.0 or not isfinite(density):
            return float("-inf")
        return log(density)

    def density_theta_mu(
        self,
        theta_e: float,
        mu: float,
        *,
        include_event_rate: bool = False,
    ) -> float:
        """Return density with respect to dthetaE dmu, marginalized over DL and DS."""
        if theta_e <= 0.0 or mu <= 0.0:
            return 0.0

        grid = self._distance_grid()
        safe_pi_rel = np.where(grid.valid, grid.pi_rel, 1.0)
        mass = theta_e * theta_e / (KAPPA * safe_pi_rel)
        jac = 2.0 * theta_e / (KAPPA * safe_pi_rel)
        p_mass = self._mass_density_grid(mass, grid.component_fractions)
        p_mu = self.murel.mu_density_for_pair_indices(grid.pair_indices, mu)
        integrand = grid.weights * p_mass * p_mu * jac
        if include_event_rate:
            integrand *= grid.dl * grid.dl * theta_e * mu
        return float(np.sum(np.where(grid.valid, integrand, 0.0)))

    def log_density_theta_mu(
        self,
        theta_e: float,
        mu: float,
        *,
        include_event_rate: bool = False,
    ) -> float:
        """Return log density with respect to dthetaE dmu, marginalized over DL and DS."""
        density = self.density_theta_mu(theta_e, mu, include_event_rate=include_event_rate)
        if density <= 0.0 or not isfinite(density):
            return float("-inf")
        return log(density)

    def _theta_mu_integrand(
        self,
        theta_e: float,
        mu: float,
        dl_kpc: float,
        ds_kpc: float,
        include_event_rate: bool,
    ) -> float:
        pi_rel = (1.0 / dl_kpc) - (1.0 / ds_kpc)
        if pi_rel <= 0.0:
            return 0.0
        mass = theta_e * theta_e / (KAPPA * pi_rel)
        jac = 2.0 * theta_e / (KAPPA * pi_rel)
        density = self.density_mu(mass, dl_kpc, ds_kpc, mu) * jac
        if include_event_rate:
            log_gamma = log_event_rate(mass, dl_kpc, ds_kpc, mu)
            if log_gamma == float("-inf"):
                return 0.0
            density *= np.exp(log_gamma)
        return float(density)

    def _distance_grid(self) -> DistanceMarginalizationGrid:
        if self._distance_marginalization_grid is None:
            self._distance_marginalization_grid = self._build_distance_grid()
        return self._distance_marginalization_grid

    def _build_distance_grid(self) -> DistanceMarginalizationGrid:
        distances = self.distance.distance_pc / 1000.0
        dl, ds = np.meshgrid(distances, distances, indexing="ij")
        valid = dl < ds
        pi_rel = np.where(valid, (1.0 / dl) - (1.0 / ds), 0.0)

        one_d_weights = _trapz_weights(distances)
        dl_weights, ds_weights = np.meshgrid(one_d_weights, one_d_weights, indexing="ij")

        p_ds = np.array([self.distance.source_pdf(x) for x in distances])
        lens_norm = np.array([self.distance._lens_integral_until(x) for x in distances])
        lens_pdf = np.zeros_like(dl)
        ok_norm = lens_norm > 0.0
        lens_pdf[:, ok_norm] = (
            self.distance.lens_density_total[:, None] / lens_norm[None, ok_norm]
        )
        weights = dl_weights * ds_weights * p_ds[None, :] * lens_pdf
        weights = np.where(valid, weights, 0.0)

        component_fractions_1d = np.array(
            [self.distance.component_fractions(x) for x in distances],
            dtype=float,
        )
        component_fractions = np.broadcast_to(
            component_fractions_1d[:, None, :],
            dl.shape + (component_fractions_1d.shape[1],),
        )
        pair_indices = np.where(valid, self.murel.nearest_pair_indices(dl, ds), -1)
        return DistanceMarginalizationGrid(
            dl=dl,
            ds=ds,
            valid=valid,
            pi_rel=pi_rel,
            weights=weights,
            component_fractions=component_fractions,
            pair_indices=pair_indices,
        )

    def _mass_density_grid(self, mass: np.ndarray, component_fractions: np.ndarray) -> np.ndarray:
        out = np.zeros(mass.shape + (self.mass.pdf_mass_by_component.shape[1],), dtype=float)
        positive = mass > 0.0
        log_mass = np.zeros_like(mass, dtype=float)
        log_mass[positive] = np.log10(mass[positive])
        for i in range(self.mass.pdf_mass_by_component.shape[1]):
            out[..., i] = np.interp(
                log_mass,
                self.mass.log_mass,
                self.mass.pdf_mass_by_component[:, i],
                left=0.0,
                right=0.0,
            )
        out[~positive, :] = 0.0
        return np.sum(out * component_fractions, axis=-1)

    def log_prior(
        self,
        ML: float,
        DL: float,
        DS: float,
        mu_N: float,
        mu_E: float,
        *,
        include_event_rate: bool = True,
    ) -> float:
        logp = self.log_density(ML, DL, DS, mu_N, mu_E)
        if not include_event_rate:
            return logp
        log_gamma = log_event_rate(ML, DL, DS, hypot(mu_N, mu_E))
        if logp == float("-inf") or log_gamma == float("-inf"):
            return float("-inf")
        return logp + log_gamma

    def mass_density_given_dl(self, mass: float, dl_kpc: float) -> float:
        p_mass_given_component = self.mass.density_given_component(mass)
        component_fraction = self.distance.component_fractions(dl_kpc)
        return float(np.sum(p_mass_given_component * component_fraction))

    def component_fractions(self, dl_kpc: float) -> dict[str, float]:
        fractions = self.distance.component_fractions(dl_kpc)
        return {name: float(value) for name, value in zip(self.component_names, fractions)}


def _load_2d(path: str | Path) -> np.ndarray:
    data = np.genfromtxt(path, comments="#")
    if data.size == 0:
        raise ValueError(f"empty table: {path}")
    return np.atleast_2d(data)


def _parse_murel_grid(path: str | Path) -> dict[str, float]:
    grid: dict[str, float] = {}
    pattern = re.compile(
        r"# Grid: (?P<name>DL|DS) "
        r"\[(?P<min>[-+0-9.eE]+), (?P<max>[-+0-9.eE]+)\] "
        r"step (?P<step>[-+0-9.eE]+) pc "
        r"\((?P<bins>\d+) bins\)"
    )
    with Path(path).open() as file:
        for line in file:
            if not line.startswith("#"):
                break
            match = pattern.match(line.strip())
            if match is None:
                continue
            name = match.group("name")
            grid[f"{name}min"] = float(match.group("min"))
            grid[f"{name}max"] = float(match.group("max"))
            grid[f"{name}step"] = float(match.group("step"))
            grid[f"{name}bins"] = float(match.group("bins"))
    return grid


def _build_pair_blocks(rows: np.ndarray) -> tuple[np.ndarray, dict[tuple[float, float], slice]]:
    pair_columns = rows[:, 0:2]
    starts = [0]
    for idx in np.nonzero(np.any(pair_columns[1:] != pair_columns[:-1], axis=1))[0] + 1:
        starts.append(int(idx))
    starts_arr = np.array(starts, dtype=int)
    ends_arr = np.append(starts_arr[1:], len(rows))

    pairs = pair_columns[starts_arr]
    block_slices = {
        (float(pair[0]), float(pair[1])): slice(int(start), int(end))
        for pair, start, end in zip(pairs, starts_arr, ends_arr)
    }
    return pairs, block_slices


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x))


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    weights = np.zeros_like(x, dtype=float)
    if len(x) < 2:
        return weights
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    if len(x) > 2:
        weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def _integral_until(x: np.ndarray, y: np.ndarray, xmax: float) -> float:
    if xmax <= x[0]:
        return 0.0
    use = x < xmax
    x_part = x[use]
    y_part = y[use]
    if len(x_part) == 0 or x_part[-1] < xmax:
        x_part = np.append(x_part, xmax)
        y_part = np.append(y_part, np.interp(xmax, x, y, left=0.0, right=0.0))
    return _trapz(y_part, x_part)


def _cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=float)
    cumulative = np.zeros_like(x, dtype=float)
    if len(x) < 2:
        return cumulative
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    cumulative[1:] = np.cumsum(increments)
    return cumulative


def _tail_rates(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return robust exponential decay rates (units: inverse ``x``)."""
    positive = np.flatnonzero(y > 0.0)
    if len(positive) < 2:
        return 1.0, 1.0
    left_idx, right_idx = positive[:3], positive[-3:]
    left_step = max(float(np.median(np.diff(x[left_idx]))), 1e-12)
    right_step = max(float(np.median(np.diff(x[right_idx]))), 1e-12)
    left_slope = np.polyfit(x[left_idx], np.log(y[left_idx]), 1)[0]
    right_slope = np.polyfit(x[right_idx], np.log(y[right_idx]), 1)[0]
    left = np.clip(left_slope, 1.0 / (3.0 * left_step), 5.0 / left_step)
    right = np.clip(-right_slope, 1.0 / (3.0 * right_step), 5.0 / right_step)
    return float(left), float(right)


def _positive_support(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(y > 0.0)
    if len(positive) == 0:
        return x[:0], y[:0]
    return x[positive[0] : positive[-1] + 1], y[positive[0] : positive[-1] + 1]


def _interp_positive_tail(value: float, x: Iterable[float], y: Iterable[float], *, lower: float | None = None) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) == 0 or (lower is not None and value <= lower):
        return 0.0
    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_idx]
    unique_x, unique_y = _positive_support(unique_x, unique_y)
    if len(unique_x) == 0:
        return 0.0
    left_rate, right_rate = _tail_rates(unique_x, unique_y)
    if value < unique_x[0]:
        return float(unique_y[0] * np.exp(-left_rate * (unique_x[0] - value)))
    if value > unique_x[-1]:
        return float(unique_y[-1] * np.exp(-right_rate * (value - unique_x[-1])))
    return float(np.interp(value, unique_x, unique_y))


def _interp_mass_tail(value: float, x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    support = y_arr > 0.0
    if not np.any(support):
        return 0.0
    x_arr, y_arr = x_arr[support], y_arr[support]
    left_rate, right_rate = _tail_rates(x_arr, y_arr)
    # dM = ln(10) * 10**log10(M) dlog10(M): the upper tail must beat
    # ln(10) to remain integrable in physical mass.
    right_rate = max(right_rate, np.log(10.0) + 1e-12)
    if value < x_arr[0]:
        return float(y_arr[0] * np.exp(-left_rate * (x_arr[0] - value)))
    if value > x_arr[-1]:
        return float(y_arr[-1] * np.exp(-right_rate * (value - x_arr[-1])))
    return float(np.interp(value, x_arr, y_arr))


def _interp_unique(value: float, x: Iterable[float], y: Iterable[float]) -> float:
    """Periodic-angle helper: unlike positive PDFs, phi has no tail."""
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) == 0:
        return 0.0
    order = np.argsort(x_arr)
    return float(np.interp(value, x_arr[order], y_arr[order], left=0.0, right=0.0))


def _integral_with_tails(
    x: np.ndarray, y: np.ndarray, *, lower: float | None, upper: float | None = None, measure: np.ndarray | None = None
) -> float:
    """Integral of a linearly interpolated positive table plus its tails."""
    if len(x) == 0:
        return 0.0
    # Mass tables are parameterised by log10(M), but their PDF is dM.
    if measure is not None:
        support = y > 0.0
        if not np.any(support):
            return 0.0
        x, y, measure = x[support], y[support], measure[support]
        interior = _trapz(y, measure)
        left_rate, right_rate = _tail_rates(x, y)
        ln10 = np.log(10.0)
        left = y[0] * ln10 * 10.0**x[0] / (left_rate + ln10)
        right_rate = max(right_rate, ln10 + 1e-12)
        right = y[-1] * ln10 * 10.0**x[-1] / (right_rate - ln10)
        return float(interior + left + right)
    x, y = _positive_support(x, y)
    if len(x) == 0:
        return 0.0
    left_rate, right_rate = _tail_rates(x, y)
    lower_limit = 0.0 if lower is None else lower
    left = y[0] / left_rate * (1.0 - np.exp(-left_rate * max(x[0] - lower_limit, 0.0)))
    interior = _trapz(y, x)
    if upper is None:
        right = y[-1] / right_rate
    elif upper <= x[-1]:
        return _integral_until(x, y, upper) + left
    else:
        right = y[-1] / right_rate * (1.0 - np.exp(-right_rate * (upper - x[-1])))
    return float(left + interior + right)


def _interp_unique_array(value: np.ndarray, x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) == 0:
        return np.zeros_like(value, dtype=float)
    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_idx]
    return np.interp(value, unique_x, unique_y, left=0.0, right=0.0)


def _wrap_phi(phi: float) -> float:
    while phi < -pi:
        phi += 2.0 * pi
    while phi > pi:
        phi -= 2.0 * pi
    return phi


def _wrap_phi_array(phi: np.ndarray) -> np.ndarray:
    return (phi + pi) % (2.0 * pi) - pi
