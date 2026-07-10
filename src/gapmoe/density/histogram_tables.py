from __future__ import annotations

import re
from dataclasses import dataclass
from math import atan2, hypot, isfinite, log, pi
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from gapmoe.priors.event_rate import KAPPA, log_event_rate
from gapmoe.source_selection import CmdPriorTable, ConditionedSourceDensity, OffsetProvider, SourceEvidenceGrid

if TYPE_CHECKING:
    from gapmoe.source_selection import GenulensSourceEvidenceBuilder


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

SOURCE_GROUP_NAMES = ("thin_disk", "thick_disk", "bulge", "NSD", "halo")
SOURCE_GROUP_BY_COMPONENT = np.asarray((0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4), dtype=int)


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

        return cls(log_mass=log_mass, pdf_mass_by_component=pdf_mass)

    def density_given_component(self, mass: float) -> np.ndarray:
        if mass <= 0.0:
            return np.zeros(self.pdf_mass_by_component.shape[1])
        log_mass = log(mass) / log(10.0)
        return np.array(
            [
                np.interp(log_mass, self.log_mass, self.pdf_mass_by_component[:, i], left=0.0, right=0.0)
                for i in range(self.pdf_mass_by_component.shape[1])
            ]
        )


@dataclass(frozen=True)
class DistanceDensityTable:
    # distance_pc: raw distance grid from rho.dat, in pc.
    distance_pc: np.ndarray
    lens_density_by_component: np.ndarray
    base_source_density_by_component: np.ndarray
    source_density_by_component: np.ndarray
    source_density: np.ndarray
    lens_density_total: np.ndarray
    lens_cumulative_integral: np.ndarray
    source_norm: float

    @classmethod
    def from_file(cls, path: str | Path) -> "DistanceDensityTable":
        data = _load_2d(path)
        distance_pc = data[:, 0]

        if data.shape[1] < 25:
            raise ValueError(f"rho table has {data.shape[1]} columns, expected at least 25: {path}")

        lens_density_by_component = data[:, 1:12]
        # genulens' forward-source grid uses nMS * P(select) * 1e-6 * D^2.
        # Keep that unselected geometric factor separately from rhoD_S, whose
        # no-cut fallback carries the legacy gammaDs distance weighting.
        base_source_density_by_component = lens_density_by_component * 1.0e-6 * distance_pc[:, None] ** 2
        # ``rhoD_S`` is a legacy, selection-specific output which can carry a
        # gammaDs weighting.  The canonical source base is always nMS * D_S^2;
        # hard cuts and CMD density factors are applied explicitly afterward.
        source_density_by_component = base_source_density_by_component
        source_density = source_density_by_component.sum(axis=1)
        lens_density_total = lens_density_by_component.sum(axis=1)
        return cls(
            distance_pc=distance_pc,
            lens_density_by_component=lens_density_by_component,
            base_source_density_by_component=base_source_density_by_component,
            source_density_by_component=source_density_by_component,
            source_density=source_density,
            lens_density_total=lens_density_total,
            lens_cumulative_integral=_cumulative_trapezoid(distance_pc, lens_density_total),
            source_norm=_trapz(source_density, distance_pc),
        )

    def with_source_evidence(self, evidence: SourceEvidenceGrid) -> "DistanceDensityTable":
        selected = ConditionedSourceDensity.from_base_density(
            self.distance_pc,
            self.base_source_density_by_component,
            evidence,
        )
        return DistanceDensityTable(
            distance_pc=self.distance_pc,
            lens_density_by_component=self.lens_density_by_component,
            base_source_density_by_component=self.base_source_density_by_component,
            source_density_by_component=selected.source_density_by_component,
            source_density=selected.source_density,
            lens_density_total=self.lens_density_total,
            lens_cumulative_integral=self.lens_cumulative_integral,
            source_norm=selected.source_norm,
        )

    def source_pdf(self, ds_kpc: float) -> float:
        if self.source_norm <= 0.0:
            return 0.0
        val = np.interp(ds_kpc * 1000.0, self.distance_pc, self.source_density, left=0.0, right=0.0)
        return float(val / self.source_norm)

    def source_pdf_array(self, ds_kpc: np.ndarray) -> np.ndarray:
        if self.source_norm <= 0.0:
            return np.zeros_like(ds_kpc, dtype=float)
        val = np.interp(ds_kpc * 1000.0, self.distance_pc, self.source_density, left=0.0, right=0.0)
        return val / self.source_norm

    def lens_pdf_given_source(self, dl_kpc: float, ds_kpc: float) -> float:
        if ds_kpc <= dl_kpc:
            return 0.0
        norm = self._lens_integral_until(ds_kpc)
        if norm <= 0.0:
            return 0.0
        val = np.interp(dl_kpc * 1000.0, self.distance_pc, self.lens_density_total, left=0.0, right=0.0)
        return float(val / norm)

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
        if ds_pc <= self.distance_pc[0]:
            return 0.0
        if ds_pc >= self.distance_pc[-1]:
            return float(self.lens_cumulative_integral[-1])
        base = float(np.interp(ds_pc, self.distance_pc, self.lens_cumulative_integral))
        left_idx = int(np.searchsorted(self.distance_pc, ds_pc, side="right")) - 1
        if left_idx < 0 or self.distance_pc[left_idx] == ds_pc:
            return base
        x0 = self.distance_pc[left_idx]
        y0 = self.lens_density_total[left_idx]
        y1 = np.interp(ds_pc, self.distance_pc, self.lens_density_total, left=0.0, right=0.0)
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
        vals = np.array(
            [
                np.interp(dl_kpc * 1000.0, self.distance_pc, self.lens_density_by_component[:, i], left=0.0, right=0.0)
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

    def source_group_weights(self, ds_kpc: float) -> np.ndarray:
        component_values = np.array(
            [
                np.interp(ds_kpc * 1000.0, self.distance_pc, self.source_density_by_component[:, i], left=0.0, right=0.0)
                for i in range(self.source_density_by_component.shape[1])
            ]
        )
        weights = np.bincount(SOURCE_GROUP_BY_COMPONENT, weights=component_values, minlength=len(SOURCE_GROUP_NAMES))
        total = weights.sum()
        return weights / total if total > 0.0 else np.zeros_like(weights)

    def source_group_weights_array(self, ds_kpc: np.ndarray) -> np.ndarray:
        ds_kpc = np.asarray(ds_kpc, dtype=float)
        component_values = np.zeros(ds_kpc.shape + (self.source_density_by_component.shape[1],), dtype=float)
        for i in range(self.source_density_by_component.shape[1]):
            component_values[..., i] = np.interp(
                ds_kpc * 1000.0,
                self.distance_pc,
                self.source_density_by_component[:, i],
                left=0.0,
                right=0.0,
            )
        weights = np.zeros(ds_kpc.shape + (len(SOURCE_GROUP_NAMES),), dtype=float)
        for group in range(len(SOURCE_GROUP_NAMES)):
            weights[..., group] = component_values[..., SOURCE_GROUP_BY_COMPONENT == group].sum(axis=-1)
        total = weights.sum(axis=-1)
        out = np.zeros_like(weights)
        np.divide(weights, total[..., None], out=out, where=total[..., None] > 0.0)
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
    source_group_mu: np.ndarray | None = None
    source_group_phi: np.ndarray | None = None

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
        source_group_mu = rows[:, 6:11] if rows.shape[1] >= 16 else None
        source_group_phi = rows[:, 11:16] if rows.shape[1] >= 16 else None
        return cls(
            rows=rows,
            pairs=pairs,
            block_slices=block_slices,
            ds_values=ds_values,
            dl_values=dl_values,
            pair_scale=pair_scale,
            grid=grid,
            source_group_mu=source_group_mu,
            source_group_phi=source_group_phi,
        )

    @property
    def has_source_groups(self) -> bool:
        return self.source_group_mu is not None and self.source_group_phi is not None

    def densities(
        self,
        dl_kpc: float,
        ds_kpc: float,
        mu: float,
        phi: float,
        source_group_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        if ds_kpc <= dl_kpc:
            return 0.0, 0.0
        return self.nearest_densities(dl_kpc, ds_kpc, mu, phi, source_group_weights)

    def mu_density(
        self,
        dl_kpc: float,
        ds_kpc: float,
        mu: float,
        source_group_weights: np.ndarray | None = None,
    ) -> float:
        if ds_kpc <= dl_kpc:
            return 0.0
        pair_pc = self._nearest_pair(dl_kpc * 1000.0, ds_kpc * 1000.0)
        return self._mu_density_for_pair(pair_pc, mu, source_group_weights)

    def mu_density_for_pair_indices(
        self,
        pair_indices: np.ndarray,
        mu: float,
        source_group_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.zeros(pair_indices.shape, dtype=float)
        for idx in np.unique(pair_indices[pair_indices >= 0]):
            pair = self.pairs[int(idx)]
            mask = pair_indices == idx
            weights = source_group_weights[mask] if source_group_weights is not None else None
            out[mask] = self._mu_density_for_pair_array(pair, np.full(mask.sum(), mu), weights)
        return out

    def densities_for_pair_indices(
        self,
        pair_indices: np.ndarray,
        mu: np.ndarray,
        phi: np.ndarray,
        source_group_weights: np.ndarray | None = None,
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
            valid_mu = mu_x > 0.0
            weights = source_group_weights[mask] if source_group_weights is not None else None
            p_mu[mask] = self._interpolate_source_groups(
                mu[mask], mu_x[valid_mu], block_slice, 4, weights, valid_mu
            )
            p_phi[mask] = self._interpolate_source_groups(phi[mask], block[:, 3], block_slice, 5, weights)
        return p_mu, p_phi


    def nearest_densities(
        self,
        dl_kpc: float,
        ds_kpc: float,
        mu: float,
        phi: float,
        source_group_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        if ds_kpc <= dl_kpc:
            return 0.0, 0.0
        pair_pc = self._nearest_pair(dl_kpc * 1000.0, ds_kpc * 1000.0)
        return self._densities_for_pair(pair_pc, mu, phi, source_group_weights)

    def nearest_pair(self, dl_kpc: float, ds_kpc: float) -> tuple[float, float]:
        """Return the nearest tabulated (DS, DL) block center in kpc."""
        pair_pc = self._nearest_pair(dl_kpc * 1000.0, ds_kpc * 1000.0)
        return float(pair_pc[0]) / 1000.0, float(pair_pc[1]) / 1000.0

    def _densities_for_pair(
        self,
        pair: np.ndarray,
        mu: float,
        phi: float,
        source_group_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        key = (float(pair[0]), float(pair[1]))
        block_slice = self.block_slices.get(key)
        if block_slice is None:
            return 0.0, 0.0
        block = self.rows[block_slice]

        mu_x = block[:, 2]
        valid_mu = mu_x > 0.0
        p_mu = self._interpolate_source_groups(
            np.asarray([mu]), mu_x[valid_mu], block_slice, 4, source_group_weights, valid_mu
        )[0]
        p_phi = self._interpolate_source_groups(
            np.asarray([_wrap_phi(phi)]), block[:, 3], block_slice, 5, source_group_weights
        )[0]
        return p_mu, p_phi

    def _mu_density_for_pair(
        self,
        pair: np.ndarray,
        mu: float,
        source_group_weights: np.ndarray | None = None,
    ) -> float:
        return float(self._mu_density_for_pair_array(pair, np.asarray([mu]), source_group_weights)[0])

    def _mu_density_for_pair_array(
        self,
        pair: np.ndarray,
        mu: np.ndarray,
        source_group_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        key = (float(pair[0]), float(pair[1]))
        block_slice = self.block_slices.get(key)
        if block_slice is None:
            return np.zeros_like(mu, dtype=float)
        block = self.rows[block_slice]

        mu_x = block[:, 2]
        valid_mu = mu_x > 0.0
        return self._interpolate_source_groups(mu, mu_x[valid_mu], block_slice, 4, source_group_weights, valid_mu)

    def _interpolate_source_groups(
        self,
        values: np.ndarray,
        x: np.ndarray,
        block_slice: slice,
        total_column: int,
        source_group_weights: np.ndarray | None,
        row_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        total_density = self.rows[block_slice, total_column]
        if row_mask is not None:
            total_density = total_density[row_mask]
        if not self.has_source_groups or source_group_weights is None:
            return _interp_unique_array(values, x, total_density)
        group_density = self.source_group_mu if total_column == 4 else self.source_group_phi
        assert group_density is not None
        weights = np.asarray(source_group_weights, dtype=float)
        if weights.ndim == 1:
            weights = np.broadcast_to(weights, (len(values), len(SOURCE_GROUP_NAMES)))
        out = np.zeros_like(values)
        for group in range(len(SOURCE_GROUP_NAMES)):
            density = group_density[block_slice, group]
            if row_mask is not None:
                density = density[row_mask]
            out += weights[:, group] * _interp_unique_array(values, x, density)
        return out

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
    source_group_weights: np.ndarray
    pair_indices: np.ndarray


class HistogramTables:
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
        source_evidence: SourceEvidenceGrid | None = None,
    ) -> "HistogramTables":
        density = cls(
            mass=MassHistogram.from_file(mass_path),
            distance=DistanceDensityTable.from_file(rho_path),
            murel=MurelHistogram.from_file(murel_path),
        )
        return density.with_source_evidence(source_evidence) if source_evidence is not None else density

    @classmethod
    def from_pre_run(
        cls,
        pre_run_result,
        *,
        source_evidence: SourceEvidenceGrid | None = None,
    ) -> "HistogramTables":
        if source_evidence is None:
            path = getattr(pre_run_result, "source_evidence_path", None)
            if path is not None and Path(path).is_file():
                source_evidence = SourceEvidenceGrid.load_npz(path)
        return cls.from_paths(
            pre_run_result.mass_path,
            pre_run_result.rho_path,
            pre_run_result.murel_path,
            source_evidence=source_evidence,
        )

    def with_source_evidence(self, evidence: SourceEvidenceGrid) -> "HistogramTables":
        """Return a copy with ``p(DS)`` rebuilt from source-data evidence.

        This uses the unselected geometric factor ``nMS * 1e-6 * DS**2`` from
        ``rho.dat`` and is therefore equivalent to genulens' forward-source
        grid. It does not reuse the legacy no-cut ``rhoD_S`` fallback, which
        carries a ``gammaDs`` distance weighting.
        """

        return HistogramTables(
            mass=self.mass,
            distance=self.distance.with_source_evidence(evidence),
            murel=self.murel,
            component_names=self.component_names,
        )

    def with_genulens_source_evidence(
        self,
        builder: "GenulensSourceEvidenceBuilder",
    ) -> "HistogramTables":
        """Build and apply genulens source-data evidence on this grid."""

        evidence = builder.build(
            self.distance.distance_pc,
            component_indices=range(self.distance.source_density_by_component.shape[1]),
        )
        return self.with_source_evidence(evidence)

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

    def cmd_joint_density(
        self,
        ML: float,
        DL: float,
        DS: float,
        mu_N: float,
        mu_E: float,
        *,
        cmd_prior: CmdPriorTable,
        reference_magnitude: float,
        color: float,
        offset_provider: OffsetProvider,
    ) -> float:
        """Joint density in event variables and apparent CMD coordinates.

        The result is a density with respect to
        ``dML dDL dDS dmu_N dmu_E dm_reference dcolor``.  It is evaluated at
        the photometry represented by the current MCMC state; no photometric
        measurement likelihood is assumed here.
        """

        mu = hypot(mu_N, mu_E)
        if mu <= 0.0 or DS <= DL or self.distance.source_norm <= 0.0:
            return 0.0
        component_values = self._cmd_component_density(
            DS,
            cmd_prior,
            reference_magnitude,
            color,
            offset_provider,
        )
        source_density = float(component_values.sum())
        if source_density <= 0.0:
            return 0.0
        group_values = np.bincount(
            SOURCE_GROUP_BY_COMPONENT,
            weights=component_values,
            minlength=len(SOURCE_GROUP_NAMES),
        )
        group_weights = group_values / source_density
        phi = atan2(mu_E, mu_N)
        p_mass = self.mass_density_given_dl(ML, DL)
        p_dl = self.distance.lens_pdf_given_source(DL, DS)
        p_mu, p_phi = self.murel.densities(DL, DS, mu, phi, group_weights)
        return p_mass * p_dl * source_density * p_mu * p_phi / mu

    def log_cmd_joint_density(self, *args, **kwargs) -> float:
        density = self.cmd_joint_density(*args, **kwargs)
        if density <= 0.0 or not isfinite(density):
            return float("-inf")
        return log(density)

    def cmd_joint_density_from_fluxes(
        self,
        ML: float,
        DL: float,
        DS: float,
        mu_N: float,
        mu_E: float,
        *,
        cmd_prior: CmdPriorTable,
        flux_blue: float,
        flux_red: float,
        zero_point_blue: float,
        zero_point_red: float,
        offset_provider: OffsetProvider,
    ) -> float:
        """CMD joint density in two source-flux parameters.

        ``cmd_prior.coordinates`` defines the blue and red bands. The CMD
        density is transformed from magnitude-colour coordinates with the
        required flux Jacobian.
        """

        reference, color = cmd_prior.coordinates.apparent_from_fluxes(
            flux_blue,
            flux_red,
            zero_point_blue=zero_point_blue,
            zero_point_red=zero_point_red,
        )
        density = self.cmd_joint_density(
            ML,
            DL,
            DS,
            mu_N,
            mu_E,
            cmd_prior=cmd_prior,
            reference_magnitude=reference,
            color=color,
            offset_provider=offset_provider,
        )
        log_jacobian = cmd_prior.coordinates.log_flux_jacobian(flux_blue, flux_red)
        return float(density * np.exp(log_jacobian)) if np.isfinite(log_jacobian) else 0.0

    def _cmd_component_density(
        self,
        ds_kpc: float,
        cmd_prior: CmdPriorTable,
        reference_magnitude: float,
        color: float,
        offset_provider: OffsetProvider,
    ) -> np.ndarray:
        values = np.zeros(self.distance.source_density_by_component.shape[1], dtype=float)
        for component in range(len(values)):
            base = np.interp(
                ds_kpc * 1000.0,
                self.distance.distance_pc,
                self.distance.source_density_by_component[:, component],
                left=0.0,
                right=0.0,
            )
            if base <= 0.0:
                continue
            photometric_density = cmd_prior.density(
                component,
                reference_magnitude,
                color,
                distance_pc=ds_kpc * 1000.0,
                magnitude_offsets=offset_provider(component, ds_kpc * 1000.0),
            )
            values[component] = base * photometric_density / self.distance.source_norm
        return values

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
        p_mu, p_phi = self.murel.densities_for_pair_indices(
            pair_indices, mu, phi, self.distance.source_group_weights_array(DS)
        )
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
        p_mass = self.mass_density_given_dl(mass, dl_kpc)
        p_dl = self.distance.lens_pdf_given_source(dl_kpc, ds_kpc)
        p_ds = self.distance.source_pdf(ds_kpc)
        p_mu, p_phi = self.murel.densities(
            dl_kpc, ds_kpc, mu, phi, self.distance.source_group_weights(ds_kpc)
        )
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
        p_mu = self.murel.mu_density(dl_kpc, ds_kpc, mu, self.distance.source_group_weights(ds_kpc))
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
        p_mu = self.murel.mu_density_for_pair_indices(grid.pair_indices, mu, grid.source_group_weights)
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
        source_group_weights_1d = self.distance.source_group_weights_array(distances)
        source_group_weights = np.broadcast_to(
            source_group_weights_1d[None, :, :],
            dl.shape + (source_group_weights_1d.shape[1],),
        )
        pair_indices = np.where(valid, self.murel.nearest_pair_indices(dl, ds), -1)
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


def _interp_unique(value: float, x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) == 0:
        return 0.0
    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]
    unique_x, unique_idx = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_idx]
    return float(np.interp(value, unique_x, unique_y, left=0.0, right=0.0))


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
