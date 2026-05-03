from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from gapmoe.density.histogram_numpy import COMPONENT_NAMES, HistogramDensity


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
                jnp.interp(log_mass, self.log_mass, self.pdf_mass_by_component[:, i], left=0.0, right=0.0)
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
        val = jnp.interp(ds_kpc * 1000.0, self.distance_pc, self.source_density, left=0.0, right=0.0)
        return jnp.where(self.source_norm > 0.0, val / self.source_norm, 0.0)

    def lens_pdf_given_source(self, dl_kpc: float, ds_kpc: float) -> jnp.ndarray:
        norm = self._lens_integral_until(ds_kpc)
        val = jnp.interp(dl_kpc * 1000.0, self.distance_pc, self.lens_density_total, left=0.0, right=0.0)
        return jnp.where((ds_kpc > dl_kpc) & (norm > 0.0), val / norm, 0.0)

    def _lens_integral_until(self, ds_kpc: float) -> jnp.ndarray:
        ds_pc = ds_kpc * 1000.0
        n = self.distance_pc.shape[0]
        idx = jnp.searchsorted(self.distance_pc, ds_pc, side="right") - 1
        idx = jnp.clip(idx, 0, n - 1)

        x0 = self.distance_pc[idx]
        y0 = self.lens_density_total[idx]
        y1 = jnp.interp(ds_pc, self.distance_pc, self.lens_density_total, left=0.0, right=0.0)
        partial = 0.5 * (y0 + y1) * (ds_pc - x0)
        value = self.lens_cumulative_integral[idx] + partial
        value = jnp.where(ds_pc <= self.distance_pc[0], 0.0, value)
        value = jnp.where(ds_pc >= self.distance_pc[-1], self.lens_cumulative_integral[-1], value)
        return value

    def component_fractions(self, dl_kpc: float) -> jnp.ndarray:
        vals = jnp.array(
            [
                jnp.interp(
                    dl_kpc * 1000.0,
                    self.distance_pc,
                    self.lens_density_by_component[:, i],
                    left=0.0,
                    right=0.0,
                )
                for i in range(self.lens_density_by_component.shape[1])
            ]
        )
        total = jnp.sum(vals)
        return jnp.where(total > 0.0, vals / total, jnp.zeros_like(vals))


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
    grid: dict[str, float]

    @classmethod
    def from_numpy(cls, density: HistogramDensity) -> "JaxMurelHistogram":
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

        return cls(
            pairs=jnp.asarray(density.murel.pairs),
            pair_scale=jnp.asarray(density.murel.pair_scale),
            mu_x=jnp.asarray(mu_x),
            mu_y=jnp.asarray(mu_y),
            mu_len=jnp.asarray(mu_len),
            phi_x=jnp.asarray(phi_x),
            phi_y=jnp.asarray(phi_y),
            phi_len=jnp.asarray(phi_len),
            grid=dict(density.murel.grid),
        )

    def densities(self, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        idx = self._nearest_pair_index(dl_kpc, ds_kpc)
        p_mu = _interp_padded(mu, self.mu_x[idx], self.mu_y[idx], self.mu_len[idx])
        p_phi = _interp_padded(_wrap_phi(phi), self.phi_x[idx], self.phi_y[idx], self.phi_len[idx])
        valid = ds_kpc > dl_kpc
        return jnp.where(valid, p_mu, 0.0), jnp.where(valid, p_phi, 0.0)

    def _nearest_pair_index(self, dl_kpc: float, ds_kpc: float) -> jnp.ndarray:
        target = jnp.array([ds_kpc * 1000.0, dl_kpc * 1000.0])
        return jnp.argmin(jnp.sum(((self.pairs - target) / self.pair_scale) ** 2, axis=1))


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

    @classmethod
    def from_paths(
        cls,
        mass_path: str | Path,
        rho_path: str | Path,
        murel_path: str | Path,
        *,
        require_source_selection: bool = True,
    ) -> "JaxHistogramDensity":
        return cls.from_numpy(
            HistogramDensity.from_paths(
                mass_path,
                rho_path,
                murel_path,
                require_source_selection=require_source_selection,
            )
        )

    @classmethod
    def from_pre_run(cls, pre_run_result, *, require_source_selection: bool = True) -> "JaxHistogramDensity":
        return cls.from_paths(
            pre_run_result.mass_path,
            pre_run_result.rho_path,
            pre_run_result.murel_path,
            require_source_selection=require_source_selection,
        )

    @classmethod
    def from_numpy(cls, density: HistogramDensity) -> "JaxHistogramDensity":
        return cls(
            mass=JaxMassHistogram.from_numpy(density),
            distance=JaxDistanceDensityTable.from_numpy(density),
            murel=JaxMurelHistogram.from_numpy(density),
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
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu, p_phi = self.murel.densities(dl_kpc, ds_kpc, mu, phi)
        return p_mass * p_dl * p_ds * p_mu * p_phi

    def log_density_mu_phi(self, mass: float, dl_kpc: float, ds_kpc: float, mu: float, phi: float) -> jnp.ndarray:
        """Return log density with respect to dML dDL dDS dmu dphi. Distances in kpc."""
        density = self.density_mu_phi(mass, dl_kpc, ds_kpc, mu, phi)
        return jnp.where(density > 0.0, jnp.log(density), -jnp.inf)

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


def _padding_step(x: np.ndarray) -> float:
    if len(x) >= 2:
        return max(float(np.nanmedian(np.diff(x))), 1.0)
    return 1.0


def _interp_padded(value: float, x: jnp.ndarray, y: jnp.ndarray, valid_len: jnp.ndarray) -> jnp.ndarray:
    last = valid_len - 1
    below = value < x[0]
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
    return jnp.where((valid_len > 0) & ~(below | above), interpolated, 0.0)


def _wrap_phi(phi: float) -> jnp.ndarray:
    return jnp.mod(phi + pi, 2.0 * pi) - pi
