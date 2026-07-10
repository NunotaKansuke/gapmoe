"""Small public API for source-aware five-dimensional Galactic priors."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gapmoe.source_selection import (
    CmdCoordinates,
    CmdPriorTable,
    ColorCut,
    ExponentialDustModel,
    ExponentialDustOffsets,
    GenulensSourceModel,
    MagnitudeCut,
    SourceSelection,
)
from gapmoe.pre_runner import PreRunResult, PreRunner

from .source import EventPrior5D, SourceCmdPrior


Context = Mapping[str, Any] | None


@dataclass(frozen=True)
class IsochroneModel:
    """Source CMD model used by :class:`GalaxyModel`.

    ``magnitude_range`` and ``color_range`` define a hard source selection.
    Leaving either range as ``None`` leaves that coordinate unrestricted.
    Leaving both unset means that no CMD selection is applied.
    """

    reference_band: str
    color_bands: tuple[str, str]
    magnitude_range: tuple[float, float] | None = None
    color_range: tuple[float, float] | None = None
    table: CmdPriorTable | None = None

    def __post_init__(self) -> None:
        if len(self.color_bands) != 2:
            raise ValueError("color_bands must be a (blue_band, red_band) pair")
        for name, bounds in (("magnitude_range", self.magnitude_range), ("color_range", self.color_range)):
            if bounds is not None and len(bounds) != 2:
                raise ValueError(f"{name} must be a (minimum, maximum) pair")
            if bounds is not None and bounds[0] >= bounds[1]:
                raise ValueError(f"{name} must have minimum < maximum")

    @property
    def coordinates(self) -> CmdCoordinates:
        return CmdCoordinates(
            reference_band=self.reference_band,
            blue_band=self.color_bands[0],
            red_band=self.color_bands[1],
        )

    @property
    def selection(self) -> SourceSelection | None:
        cuts = []
        if self.magnitude_range is not None:
            cuts.append(MagnitudeCut(self.reference_band, *self.magnitude_range))
        if self.color_range is not None:
            cuts.append(ColorCut(*self.color_bands, *self.color_range))
        return SourceSelection(tuple(cuts)) if cuts else None

    def build(
        self,
        *,
        reference_edges: Sequence[float],
        color_edges: Sequence[float],
        smoothing_sigma_bins: float = 0.75,
        source_model: GenulensSourceModel | None = None,
    ) -> "IsochroneModel":
        """Build the intrinsic CMD table once from the isochrone population."""

        source_model = source_model or GenulensSourceModel()
        return replace(
            self,
            table=source_model.build_cmd_prior(
                self.coordinates,
                reference_edges=reference_edges,
                color_edges=color_edges,
                smoothing_sigma_bins=smoothing_sigma_bins,
            ),
        )


@dataclass(frozen=True)
class GalaxyModel:
    """Source-aware Galactic event prior with one ``log_density`` entry point.

    ``theta`` is always ``(ML, DL, DS, mu_N, mu_E)``.  ``cmd`` is either
    ``None`` or ``(reference magnitude, blue-minus-red colour)``.
    """

    density: Any
    isochrone: IsochroneModel
    l_deg: float
    b_deg: float
    extinction_at_rc: Mapping[str, float]
    dm_rc: float | None = None
    dust_scale_height_pc: float = 164.0
    include_event_rate: bool = True

    @classmethod
    def from_pre_run(
        cls,
        pre_run: Any,
        *,
        isochrone: IsochroneModel,
        extinction_at_rc: Mapping[str, float],
        dm_rc: float | None = None,
        dust_scale_height_pc: float = 164.0,
        include_event_rate: bool = True,
    ) -> "GalaxyModel":
        """Create a histogram-backed model using the pre-run sightline.

        ``PreRunner.run(l=..., b=...)`` is the sole place where users need to
        supply the Galactic coordinates for this path.
        """

        from gapmoe.density import HistogramDensity

        return cls(
            density=HistogramDensity.from_pre_run(pre_run),
            isochrone=isochrone,
            l_deg=float(pre_run.l_deg),
            b_deg=float(pre_run.b_deg),
            extinction_at_rc=extinction_at_rc,
            dm_rc=dm_rc,
            dust_scale_height_pc=dust_scale_height_pc,
            include_event_rate=include_event_rate,
        )

    def __post_init__(self) -> None:
        if self.isochrone.table is None:
            raise ValueError("isochrone must be built before constructing GalaxyModel")

        dust = ExponentialDustOffsets(
            l_deg=self.l_deg,
            b_deg=self.b_deg,
            extinction_at_reference=self.extinction_at_rc,
            dm_reference=self.dm_rc,
            dust_scale_height_pc=self.dust_scale_height_pc,
        )
        runtime_dust = ExponentialDustModel.from_exponential(dust, self.isochrone.coordinates)
        source_prior = SourceCmdPrior(
            density=self.density,
            cmd_prior=self.isochrone.table.evaluator(),
            offset_calculator=lambda ds_kpc, context: runtime_dust.offsets(ds_kpc),
        )
        conditional = EventPrior5D(self.density, source_prior, include_event_rate=self.include_event_rate)

        selected_density = self.density
        if self.isochrone.selection is not None:
            evidence = self.isochrone.table.evidence_for_selection(
                self.isochrone.selection,
                np.asarray(self.density.distance.distance_pc),
                offset_provider=dust,
            )
            selected_density = self.density.with_source_evidence(evidence)
        selected = EventPrior5D(selected_density, source_prior, include_event_rate=self.include_event_rate)

        object.__setattr__(self, "_conditional_prior", conditional)
        object.__setattr__(self, "_selected_prior", selected)

    def log_density(self, theta: Any, cmd: Any | None = None, *, context: Context = None):
        """Evaluate the event density for a five-vector and optional CMD pair."""

        ml, dl, ds, mu_n, mu_e = theta[:5]
        if cmd is None:
            return self._selected_prior.log_density(ml, dl, ds, mu_n, mu_e, context=context)
        return self._conditional_prior.log_density(
            ml,
            dl,
            ds,
            mu_n,
            mu_e,
            reference_magnitude=cmd[0],
            color=cmd[1],
            context=context,
        )


class Model:
    """High-level gapmoe interface for one Galactic line of sight.

    The public workflow is ``set(...)``, ``prepare()``, ``isochrone(...)``,
    and ``galactic_model(...)``.  Pre-gapmoe artifacts and dust conversion are
    intentionally kept behind this interface.
    """

    _SETTINGS = {"l", "b", "extinction", "dm_rc", "dust_scale_height_pc", "ai_rc", "evi_rc"}

    def __init__(
        self,
        *,
        genulens_root: str | Path | None = None,
        auto_build: bool = False,
        backend: str = "auto",
    ) -> None:
        self._runner = PreRunner(
            genulens_root=genulens_root,
            output_dir=".",
            auto_build=auto_build,
            backend=backend,
        )
        self.directory: Path | None = None
        self._settings: dict[str, Any] = {"dust_scale_height_pc": 164.0}
        self._explicit_settings: set[str] = set()
        self._prepare_options: dict[str, Any] = {}
        self._prepared: PreRunResult | None = None

    def set(self, **settings: Any) -> "Model":
        """Set sightline and extinction values, invalidating prepared tables."""

        unknown = set(settings) - self._SETTINGS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown model setting(s): {names}")
        if "extinction" in settings:
            extinction = settings["extinction"]
            if not isinstance(extinction, Mapping):
                raise TypeError("extinction must map band names to RC extinctions")
            settings["extinction"] = {str(band): float(value) for band, value in extinction.items()}
        for name in ("l", "b", "dm_rc", "dust_scale_height_pc", "ai_rc", "evi_rc"):
            if name in settings and settings[name] is not None:
                settings[name] = float(settings[name])
        if "dust_scale_height_pc" in settings and settings["dust_scale_height_pc"] <= 0.0:
            raise ValueError("dust_scale_height_pc must be positive")
        if "extinction" in settings and ("ai_rc" in settings or "evi_rc" in settings):
            raise ValueError("use either extinction or ai_rc/evi_rc, not both")

        precompute_changed = any(
            name in settings and settings[name] != self._settings.get(name) for name in ("l", "b")
        )
        self._settings.update(settings)
        self._explicit_settings.update(settings)
        if precompute_changed:
            self._prepared = None
        return self

    def prepare(self, directory: str | Path, *, force: bool = False, **options: Any) -> "Model":
        """Create or reuse raw pre-gapmoe artifacts for the configured sightline."""

        self.directory = Path(directory).expanduser().resolve()
        if not force:
            cached = self._load_prepared_directory(self.directory)
            if cached is not None:
                self._prepared = cached
                cached_settings = self._load_settings(self.directory)
                for name in ("l", "b"):
                    cached_value = cached_settings.get(name, getattr(cached, f"{name}_deg"))
                    if name in self._explicit_settings and self._settings[name] != cached_value:
                        raise ValueError(
                            f"prepared directory sightline disagrees on {name}; use a new directory or force=True"
                        )
                self._settings.update(
                    {name: value for name, value in cached_settings.items() if name not in self._explicit_settings}
                )
                if not options or options == self._prepare_options:
                    return self
        if "l" not in self._settings or "b" not in self._settings:
            raise ValueError("set l and b before prepare()")
        forbidden = {"source_model", "cmd_prior", "l", "b", "l_deg", "b_deg", "run_name"} & set(options)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise TypeError(f"prepare() manages {names}; configure source CMD through isochrone()")

        if not force and self._prepared is not None and options == self._prepare_options:
            return self
        self._runner.output_dir = self.directory.parent
        self._prepared = self._runner.run(
            l=self._settings["l"],
            b=self._settings["b"],
            run_name=self.directory.name,
            **options,
        )
        self._prepare_options = dict(options)
        self._write_settings()
        return self

    def resume(self, directory: str | Path) -> "Model":
        """Open an existing prepared event directory without rerunning genulens."""

        self.directory = Path(directory).expanduser().resolve()
        prepared = self._load_prepared_directory(self.directory)
        if prepared is None:
            raise FileNotFoundError(f"no complete gapmoe artifacts in {self.directory}")
        self._prepared = prepared
        self._settings.update(self._load_settings(self.directory))
        self._explicit_settings.clear()
        return self

    def isochrone(
        self,
        *,
        reference_band: str,
        color_bands: tuple[str, str],
        magnitude_range: tuple[float, float] | None = None,
        color_range: tuple[float, float] | None = None,
        reference_edges: Sequence[float] | None = None,
        color_edges: Sequence[float] | None = None,
        smoothing_sigma_bins: float = 0.75,
    ) -> IsochroneModel:
        """Build an isochrone CMD model with optional hard source cuts."""

        reference_edges = np.asarray(
            np.linspace(-8.0, 20.0, 561) if reference_edges is None else reference_edges,
            dtype=float,
        )
        color_edges = np.asarray(
            np.linspace(-2.0, 8.0, 201) if color_edges is None else color_edges,
            dtype=float,
        )
        model = IsochroneModel(
            reference_band=reference_band,
            color_bands=color_bands,
            magnitude_range=magnitude_range,
            color_range=color_range,
        )
        cached = self._load_isochrone(model, reference_edges, color_edges, smoothing_sigma_bins)
        if cached is not None:
            return cached
        built = model.build(
            reference_edges=reference_edges,
            color_edges=color_edges,
            smoothing_sigma_bins=smoothing_sigma_bins,
        )
        self._save_isochrone(built, smoothing_sigma_bins)
        return built

    def galactic_model(self, isochrone: IsochroneModel, *, include_event_rate: bool = True) -> GalaxyModel:
        """Return the five-dimensional event prior for an isochrone model."""

        if self._prepared is None:
            raise RuntimeError("call prepare() before galactic_model()")
        return GalaxyModel.from_pre_run(
            self._prepared,
            isochrone=isochrone,
            extinction_at_rc=self._extinction_for(isochrone),
            dm_rc=self._settings.get("dm_rc"),
            dust_scale_height_pc=self._settings["dust_scale_height_pc"],
            include_event_rate=include_event_rate,
        )

    def _extinction_for(self, isochrone: IsochroneModel) -> Mapping[str, float]:
        extinction = self._settings.get("extinction")
        if extinction is not None:
            missing = set(isochrone.coordinates.bands) - set(extinction)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"extinction is missing RC values for CMD band(s): {names}")
            return extinction
        ai_rc = self._settings.get("ai_rc")
        evi_rc = self._settings.get("evi_rc")
        if ai_rc is None and evi_rc is None:
            return {}
        if ai_rc is None:
            raise ValueError("evi_rc requires ai_rc")
        blue, red = isochrone.color_bands
        if isochrone.reference_band != red:
            raise ValueError("ai_rc/evi_rc shorthand requires the reference band to be the red color band")
        return {red: ai_rc, blue: ai_rc + (0.0 if evi_rc is None else evi_rc)}

    def _load_prepared_directory(self, directory: Path) -> PreRunResult | None:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
            paths = {name: Path(manifest[f"{name}_path"]) for name in ("mass", "rho", "murel")}
        except (KeyError, OSError, json.JSONDecodeError):
            return None
        if not all(path.is_file() for path in paths.values()):
            return None
        source_path = manifest.get("source_evidence_path")
        cmd_path = manifest.get("cmd_prior_path")
        return PreRunResult(
            ra_deg=manifest.get("ra_deg"),
            dec_deg=manifest.get("dec_deg"),
            l_deg=float(manifest["l_deg"]),
            b_deg=float(manifest["b_deg"]),
            output_dir=directory,
            mass_path=paths["mass"],
            rho_path=paths["rho"],
            murel_path=paths["murel"],
            manifest_path=manifest_path,
            source_evidence_path=Path(source_path) if source_path else None,
            cmd_prior_path=Path(cmd_path) if cmd_path else None,
            commands=manifest.get("commands", {}),
        )

    def _load_settings(self, directory: Path) -> dict[str, Any]:
        path = directory / "gapmoe.json"
        if not path.is_file():
            return {"l": self._prepared.l_deg, "b": self._prepared.b_deg} if self._prepared else {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"l": self._prepared.l_deg, "b": self._prepared.b_deg} if self._prepared else {}
        self._prepare_options = dict(payload.get("prepare_options", {}))
        return dict(payload.get("settings", {}))

    def _write_settings(self) -> None:
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "gapmoe-model-v1",
            "l_deg": self._settings["l"],
            "b_deg": self._settings["b"],
            "settings": self._settings,
            "prepare_options": self._prepare_options,
        }
        (self.directory / "gapmoe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _load_isochrone(
        self,
        model: IsochroneModel,
        reference_edges: np.ndarray,
        color_edges: np.ndarray,
        smoothing_sigma_bins: float,
    ) -> IsochroneModel | None:
        if self.directory is None:
            return None
        table_path = self.directory / "cmd_prior.npz"
        metadata_path = self.directory / "isochrone.json"
        if not table_path.is_file() or not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text())
            table = CmdPriorTable.load_npz(table_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        expected = self._isochrone_metadata(model, smoothing_sigma_bins)
        if metadata != expected:
            return None
        if not np.array_equal(table.reference_edges, reference_edges) or not np.array_equal(table.color_edges, color_edges):
            return None
        return replace(model, table=table)

    def _save_isochrone(self, model: IsochroneModel, smoothing_sigma_bins: float) -> None:
        if self.directory is None or model.table is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        model.table.save_npz(self.directory / "cmd_prior.npz")
        (self.directory / "isochrone.json").write_text(
            json.dumps(self._isochrone_metadata(model, smoothing_sigma_bins), indent=2, sort_keys=True) + "\n"
        )

    @staticmethod
    def _isochrone_metadata(model: IsochroneModel, smoothing_sigma_bins: float) -> dict[str, Any]:
        return {
            "reference_band": model.reference_band,
            "color_bands": list(model.color_bands),
            "magnitude_range": list(model.magnitude_range) if model.magnitude_range is not None else None,
            "color_range": list(model.color_range) if model.color_range is not None else None,
            "smoothing_sigma_bins": smoothing_sigma_bins,
        }
