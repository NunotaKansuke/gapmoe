from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite, log, pi
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from gapmoe.density.base import DensityModel
from gapmoe.physical import PhysicalParams
from gapmoe.priors.event_rate import log_event_rate


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
    distance_pc: np.ndarray
    lens_density_by_component: np.ndarray
    source_density: np.ndarray

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
        return cls(
            distance_pc=distance_pc,
            lens_density_by_component=lens_density_by_component,
            source_density=source_density,
        )

    def source_pdf(self, ds_pc: float) -> float:
        norm = _trapz(self.source_density, self.distance_pc)
        if norm <= 0.0:
            return 0.0
        val = np.interp(ds_pc, self.distance_pc, self.source_density, left=0.0, right=0.0)
        return float(val / norm)

    def lens_pdf_given_source(self, dl_pc: float, ds_pc: float) -> float:
        if ds_pc <= dl_pc:
            return 0.0
        lens_total = self.lens_density_by_component.sum(axis=1)
        norm = _integral_until(self.distance_pc, lens_total, ds_pc)
        if norm <= 0.0:
            return 0.0
        val = np.interp(dl_pc, self.distance_pc, lens_total, left=0.0, right=0.0)
        return float(val / norm)

    def component_fractions(self, dl_pc: float) -> np.ndarray:
        vals = np.array(
            [
                np.interp(dl_pc, self.distance_pc, self.lens_density_by_component[:, i], left=0.0, right=0.0)
                for i in range(self.lens_density_by_component.shape[1])
            ]
        )
        total = vals.sum()
        if total <= 0.0:
            return np.zeros_like(vals)
        return vals / total


@dataclass(frozen=True)
class MurelHistogram:
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

    def densities(self, dl_pc: float, ds_pc: float, mu: float, phi: float) -> tuple[float, float]:
        if ds_pc <= dl_pc:
            return 0.0, 0.0
        weighted = self._weighted_pair_densities(dl_pc, ds_pc, mu, phi)
        if weighted is not None:
            return weighted
        return self.nearest_densities(dl_pc, ds_pc, mu, phi)

    def nearest_densities(self, dl_pc: float, ds_pc: float, mu: float, phi: float) -> tuple[float, float]:
        if ds_pc <= dl_pc:
            return 0.0, 0.0
        pair = self.nearest_pair(dl_pc, ds_pc)
        return self._densities_for_pair(pair, mu, phi)

    def nearest_pair(self, dl_pc: float, ds_pc: float) -> tuple[float, float]:
        pair = self._nearest_pair(dl_pc, ds_pc)
        return float(pair[0]), float(pair[1])

    def _densities_for_pair(self, pair: np.ndarray, mu: float, phi: float) -> tuple[float, float]:
        key = (float(pair[0]), float(pair[1]))
        block_slice = self.block_slices.get(key)
        if block_slice is None:
            return 0.0, 0.0
        block = self.rows[block_slice]

        mu_x = block[:, 2]
        mu_y = block[:, 4]
        valid_mu = mu_x > 0.0
        p_mu = _interp_unique(mu, mu_x[valid_mu], mu_y[valid_mu])

        phi_x = block[:, 3]
        phi_y = block[:, 5]
        p_phi = _interp_unique(_wrap_phi(phi), phi_x, phi_y)
        return p_mu, p_phi

    def _weighted_pair_densities(self, dl_pc: float, ds_pc: float, mu: float, phi: float) -> Optional[tuple[float, float]]:
        dl_neighbors = _bracketing_values(self.dl_values, dl_pc)
        ds_neighbors = _bracketing_values(self.ds_values, ds_pc)
        if not dl_neighbors or not ds_neighbors:
            return None

        weighted_mu = 0.0
        weighted_phi = 0.0
        weight_sum = 0.0
        for ds_val in ds_neighbors:
            for dl_val in dl_neighbors:
                if dl_val >= ds_val:
                    continue
                if not self._has_pair(ds_val, dl_val):
                    continue
                distance = abs(ds_pc - ds_val) / self.pair_scale[0]
                distance += abs(dl_pc - dl_val) / self.pair_scale[1]
                weight = 1.0 / max(distance, 1e-12)
                p_mu, p_phi = self._densities_for_pair(np.array([ds_val, dl_val]), mu, phi)
                weighted_mu += weight * p_mu
                weighted_phi += weight * p_phi
                weight_sum += weight

        if weight_sum <= 0.0:
            return None
        return weighted_mu / weight_sum, weighted_phi / weight_sum

    def _has_pair(self, ds_pc: float, dl_pc: float) -> bool:
        return (float(ds_pc), float(dl_pc)) in self.block_slices

    def _nearest_pair(self, dl_pc: float, ds_pc: float) -> np.ndarray:
        target = np.array([ds_pc, dl_pc])
        idx = np.argmin(np.sum(((self.pairs - target) / self.pair_scale) ** 2, axis=1))
        return self.pairs[idx]


class HistogramDensity(DensityModel):
    """Histogram-backed Galactic density model.

    This reads the current `PreRunner` outputs: `mass.dat`, `rho.dat`, and
    `murel.dat`. The canonical distance unit is pc.
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

    def log_density(self, params: PhysicalParams) -> float:
        density = self.density(params)
        if density <= 0.0 or not isfinite(density):
            return float("-inf")
        return log(density)

    def density(self, params: PhysicalParams) -> float:
        p_mass = self.mass_density_given_dl(params.ML, params.DL)
        p_dl = self.distance.lens_pdf_given_source(params.DL, params.DS)
        p_ds = self.distance.source_pdf(params.DS)
        p_mu, p_phi = self.murel.densities(params.DL, params.DS, params.mu, params.phi)
        return p_mass * p_dl * p_ds * p_mu * p_phi

    def log_prior(self, params: PhysicalParams, *, include_event_rate: bool = True) -> float:
        logp = self.log_density(params)
        if not include_event_rate:
            return logp
        log_gamma = log_event_rate(params)
        if logp == float("-inf") or log_gamma == float("-inf"):
            return float("-inf")
        return logp + log_gamma

    def mass_density_given_dl(self, mass: float, dl_pc: float) -> float:
        p_mass_given_component = self.mass.density_given_component(mass)
        component_fraction = self.distance.component_fractions(dl_pc)
        return float(np.sum(p_mass_given_component * component_fraction))

    def component_fractions(self, dl_pc: float) -> dict[str, float]:
        fractions = self.distance.component_fractions(dl_pc)
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
    return float(np.trapz(y, x))


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


def _bracketing_values(values: np.ndarray, value: float) -> list[float]:
    if len(values) == 0 or value < values[0] or value > values[-1]:
        return []
    right = int(np.searchsorted(values, value, side="left"))
    if right < len(values) and values[right] == value:
        return [float(values[right])]
    left = right - 1
    out = []
    if 0 <= left < len(values):
        out.append(float(values[left]))
    if 0 <= right < len(values):
        out.append(float(values[right]))
    return out


def _wrap_phi(phi: float) -> float:
    while phi < -pi:
        phi += 2.0 * pi
    while phi > pi:
        phi -= 2.0 * pi
    return phi
