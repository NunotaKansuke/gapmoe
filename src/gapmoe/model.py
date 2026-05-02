from __future__ import annotations

import warnings
from math import cos, sin
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gapmoe.density import HistogramDensity
from gapmoe.physical import PhysicalParams
from gapmoe.pre_runner import PreRunResult, PreRunner, SourceSelection
from gapmoe.priors import GalacticPrior


class GalacticModel:
    """Compatibility-facing model built from PreRunner, density, and prior objects."""

    def __init__(
        self,
        ra_deg: Optional[float] = None,
        dec_deg: Optional[float] = None,
        *,
        density: Optional[HistogramDensity] = None,
        pre_run_result: Optional[PreRunResult] = None,
        mass_path: Optional[str | Path] = None,
        rho_path: Optional[str | Path] = None,
        murel_path: Optional[str | Path] = None,
        rhos_path: Optional[str | Path] = None,
        mu_path: Optional[str | Path] = None,
        genulens_root: Optional[str | Path] = None,
        output_dir: str | Path = ".",
        run_name: Optional[str] = None,
        source: Optional[SourceSelection] = None,
        include_event_rate: bool = True,
        require_source_selection: bool = True,
        auto_build: bool = False,
        **pre_runner_options: Any,
    ) -> None:
        self.pre_run_result = pre_run_result

        if rho_path is None and rhos_path is not None:
            rho_path = rhos_path
        if murel_path is None and mu_path is not None:
            murel_path = mu_path

        if density is None:
            if pre_run_result is not None:
                density = HistogramDensity.from_pre_run(
                    pre_run_result,
                    require_source_selection=require_source_selection,
                )
            elif mass_path is not None and rho_path is not None and murel_path is not None:
                density = HistogramDensity.from_paths(
                    mass_path,
                    rho_path,
                    murel_path,
                    require_source_selection=require_source_selection,
                )
            elif ra_deg is not None and dec_deg is not None:
                runner = PreRunner(genulens_root=genulens_root, output_dir=output_dir, auto_build=auto_build)
                self.pre_run_result = runner.run(
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    run_name=run_name,
                    source=source,
                    **pre_runner_options,
                )
                density = HistogramDensity.from_pre_run(
                    self.pre_run_result,
                    require_source_selection=require_source_selection,
                )
            else:
                raise ValueError(
                    "Provide density=..., pre_run_result=..., mass/rho/murel paths, "
                    "or ra_deg and dec_deg for PreRunner."
                )

        self.density = density
        self.prior = GalacticPrior(density, include_event_rate=include_event_rate)

    @classmethod
    def from_pre_run(cls, pre_run_result: PreRunResult, **kwargs: Any) -> "GalacticModel":
        return cls(pre_run_result=pre_run_result, **kwargs)

    @classmethod
    def from_paths(
        cls,
        mass_path: str | Path,
        rho_path: str | Path,
        murel_path: str | Path,
        **kwargs: Any,
    ) -> "GalacticModel":
        return cls(mass_path=mass_path, rho_path=rho_path, murel_path=murel_path, **kwargs)

    def set_data(self, fREM_range: Any = None) -> "GalacticModel":
        return self

    def set_data_rho(self) -> "GalacticModel":
        return self

    def set_data_mass(self, fREM_range: Any = None) -> "GalacticModel":
        return self

    def set_data_mu(self) -> "GalacticModel":
        return self

    def get_density_DL(self, DL_value: float, comp: str = "all") -> float:
        if comp != "all":
            return self._component_density(DL_value, comp)
        vals = self.density.distance.lens_density_by_component.sum(axis=1)
        norm = _trapz(vals, self.density.distance.distance_pc)
        if norm <= 0.0:
            return 0.0
        return float(np.interp(DL_value, self.density.distance.distance_pc, vals, left=0.0, right=0.0) / norm)

    def get_density_DL_given_DS(self, DL_value: float, DS_value: float) -> float:
        return self.density.distance.lens_pdf_given_source(DL_value, DS_value)

    def get_density_DS(self, DS_value: float, comp: str = "all") -> float:
        if comp != "all":
            raise ValueError("component-specific source density is not exposed by the new wrapper yet.")
        return self.density.distance.source_pdf(DS_value)

    def get_density_M(self, M_value: float, comp: str = "bulge") -> float:
        idx = self._component_index(comp)
        return float(self.density.mass.density_given_component(M_value)[idx])

    def get_density_M_given_DL(self, M_value: float, DL_value: float) -> float:
        return self.density.mass_density_given_dl(M_value, DL_value)

    def get_density_mu_given_DL_DS(
        self,
        DL_value: float,
        DS_value: float,
        mu_value: float,
        phi_value: float,
    ) -> tuple[float, float]:
        return self.density.murel.densities(DL_value, DS_value, mu_value, phi_value)

    def safe_log(self, x: float) -> float:
        if x <= 0.0:
            return float("-inf")
        return float(np.log(x))

    def get_joint_log_density(
        self,
        ML_value: float,
        DL_value: float,
        DS_value: float,
        mu_value: float,
        phi_value: float,
    ) -> float:
        return self.density.log_density(_physical_from_mu_phi(ML_value, DL_value, DS_value, mu_value, phi_value))

    def get_joint_log_density_given_DS(
        self,
        ML_value: float,
        DL_value: float,
        DS_value: float,
        mu_value: float,
        phi_value: float,
    ) -> float:
        p_mass = self.get_density_M_given_DL(ML_value, DL_value)
        p_dl = self.get_density_DL_given_DS(DL_value, DS_value)
        p_mu, p_phi = self.get_density_mu_given_DL_DS(DL_value, DS_value, mu_value, phi_value)
        return self.safe_log(p_mass) + self.safe_log(p_dl) + self.safe_log(p_mu) + self.safe_log(p_phi)

    def log_prob(self, params: PhysicalParams) -> float:
        return self.prior.log_prob(params)

    def calc_log_Gamma(self, ML: float, DL: float, DS: float, mu: float) -> float:
        from gapmoe.priors.event_rate import log_event_rate

        return log_event_rate(_physical_from_mu_phi(ML, DL, DS, mu, 0.0))

    def log_galactic_prior(
        self,
        ML_value: float,
        DL_value: float,
        DS_value: float,
        mu_value: float,
        phi_value: float,
    ) -> float:
        return self.prior.log_prob(_physical_from_mu_phi(ML_value, DL_value, DS_value, mu_value, phi_value))

    def log_galactic_prior_given_DS(
        self,
        ML_value: float,
        DL_value: float,
        DS_value: float,
        mu_value: float,
        phi_value: float,
    ) -> float:
        log_density = self.get_joint_log_density_given_DS(ML_value, DL_value, DS_value, mu_value, phi_value)
        log_gamma = self.calc_log_Gamma(ML_value, DL_value, DS_value, mu_value)
        if not np.isfinite(log_density) or not np.isfinite(log_gamma):
            return float("-inf")
        return float(log_density + log_gamma)

    def get_comp_fraction(self, DL_value: float) -> np.ndarray:
        return self.density.distance.component_fractions(DL_value)

    def component_fractions(self, DL_value: float) -> dict[str, float]:
        return self.density.component_fractions(DL_value)

    def _component_density(self, distance_pc: float, comp: str) -> float:
        idx = self._component_index(comp)
        vals = self.density.distance.lens_density_by_component[:, idx]
        norm = _trapz(vals, self.density.distance.distance_pc)
        if norm <= 0.0:
            return 0.0
        return float(np.interp(distance_pc, self.density.distance.distance_pc, vals, left=0.0, right=0.0) / norm)

    def _component_index(self, comp: str) -> int:
        if comp == "all":
            raise ValueError("'all' does not identify one mass component.")
        try:
            return self.density.component_names.index(comp)
        except ValueError as exc:
            names = ", ".join(self.density.component_names)
            raise ValueError(f"unknown component {comp!r}; expected one of: {names}") from exc


class gapmoe(GalacticModel):
    """Deprecated alias for old scripts. Use GalacticModel instead."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "gapmoe.gapmoe is deprecated; use gapmoe.GalacticModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _physical_from_mu_phi(ML: float, DL: float, DS: float, mu: float, phi: float) -> PhysicalParams:
    return PhysicalParams(ML=ML, DL=DL, DS=DS, mu_N=mu * cos(phi), mu_E=mu * sin(phi))


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.trapz(y, x))
