from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


ComponentIndex = int
BandOffsets = Mapping[str, float]


@dataclass(frozen=True)
class AgeMetallicityPoint:
    """One weighted age-metallicity node for an isochrone source population."""

    log_age: float
    metallicity_mh: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.log_age) or not np.isfinite(self.metallicity_mh) or self.weight < 0.0:
            raise ValueError("age-metallicity points require finite values and non-negative weights")


@dataclass(frozen=True)
class SourcePopulation:
    """Optional overrides for genulens' default source-population prior.

    Leaving both fields unset reproduces genulens' component-dependent
    age-metallicity prior and its default broken-power-law IMF.
    """

    imf: Mapping[str, float] | None = None
    age_metallicity_by_component: Mapping[int, Sequence[AgeMetallicityPoint]] | None = None

    def __post_init__(self) -> None:
        if self.imf is not None:
            unknown = set(self.imf) - {
                "m0", "m1", "m2", "m3", "mbr", "ml", "mu", "alpha0", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5"
            }
            if unknown:
                raise ValueError(f"unknown IMF parameter(s): {', '.join(sorted(unknown))}")
        if self.age_metallicity_by_component is not None:
            for component, points in self.age_metallicity_by_component.items():
                if int(component) < 0 or not points or sum(point.weight for point in points) <= 0.0:
                    raise ValueError("each overridden source component needs positive total population weight")

    def metadata(self) -> dict[str, object]:
        return {
            "imf": None if self.imf is None else dict(self.imf),
            "age_metallicity_by_component": (
                None
                if self.age_metallicity_by_component is None
                else {
                    str(component): [
                        {"log_age": point.log_age, "metallicity_mh": point.metallicity_mh, "weight": point.weight}
                        for point in points
                    ]
                    for component, points in self.age_metallicity_by_component.items()
                }
            ),
        }


def angular_radius_microarcsec(radius_rsun: np.ndarray | float, distance_pc: np.ndarray | float) -> np.ndarray:
    """Return stellar angular radius in microarcsec."""

    radius = np.asarray(radius_rsun, dtype=float)
    distance = np.asarray(distance_pc, dtype=float)
    solar_radius_au = 0.004650467260962157
    return radius * solar_radius_au / distance * 1.0e6


@dataclass(frozen=True)
class SourceObservables:
    """Isochrone-derived source observables at one distance.

    Magnitudes stored in ``absolute_magnitudes`` are intrinsic table values.
    Apparent selections add ``magnitude_offsets`` band by band, usually distance
    modulus plus extinction.
    """

    absolute_magnitudes: Mapping[str, np.ndarray]
    radius_rsun: np.ndarray
    distance_pc: float
    magnitude_offsets: BandOffsets = field(default_factory=dict)

    def magnitude(self, band: str, *, apparent: bool = True) -> np.ndarray:
        if band not in self.absolute_magnitudes:
            raise KeyError(f"source observable band is not available: {band}")
        values = np.asarray(self.absolute_magnitudes[band], dtype=float)
        if apparent:
            values = values + float(self.magnitude_offsets.get(band, 0.0))
        return values

    @property
    def theta_star_microarcsec(self) -> np.ndarray:
        return angular_radius_microarcsec(self.radius_rsun, self.distance_pc)


class SourceCut(Protocol):
    def mask(self, source: SourceObservables) -> np.ndarray:
        ...


class SourceDataModel(Protocol):
    """A source-data factor, evaluated on isochrone observables."""

    def log_weight(self, source: SourceObservables) -> np.ndarray:
        ...


@dataclass(frozen=True)
class MagnitudeCut:
    band: str
    minimum: float
    maximum: float
    apparent: bool = True

    def mask(self, source: SourceObservables) -> np.ndarray:
        mag = source.magnitude(self.band, apparent=self.apparent)
        return (mag >= self.minimum) & (mag <= self.maximum)


@dataclass(frozen=True)
class ColorCut:
    blue_band: str
    red_band: str
    minimum: float
    maximum: float
    apparent: bool = True

    def mask(self, source: SourceObservables) -> np.ndarray:
        color = source.magnitude(self.blue_band, apparent=self.apparent) - source.magnitude(
            self.red_band,
            apparent=self.apparent,
        )
        return (color >= self.minimum) & (color <= self.maximum)


@dataclass(frozen=True)
class SourceSelection:
    cuts: tuple[SourceCut, ...] = ()
    label: str | None = None

    def mask(self, source: SourceObservables) -> np.ndarray:
        if not self.cuts:
            first = next(iter(source.absolute_magnitudes.values()), source.radius_rsun)
            return np.ones(np.asarray(first).shape, dtype=bool)
        out: np.ndarray | None = None
        for cut in self.cuts:
            cut_mask = np.asarray(cut.mask(source), dtype=bool)
            out = cut_mask if out is None else out & cut_mask
        if out is None:
            first = next(iter(source.absolute_magnitudes.values()), source.radius_rsun)
            return np.ones(np.asarray(first).shape, dtype=bool)
        return out

    def log_weight(self, source: SourceObservables) -> np.ndarray:
        return np.where(self.mask(source), 0.0, -np.inf)


@dataclass(frozen=True)
class ExponentialDustOffsets:
    """Distance modulus plus an exponential extinction screen.

    ``extinction_at_reference`` gives extinction in each band at the red-clump
    reference distance. This is the same functional form used by genulens'
    manual ``AIrc``/``EVIrc`` source-selection path.
    """

    l_deg: float
    b_deg: float
    extinction_at_reference: BandOffsets
    dm_reference: float | None = None
    dust_scale_height_pc: float = 164.0

    def __call__(self, component: ComponentIndex, distance_pc: float) -> BandOffsets:
        del component
        if distance_pc <= 0.0:
            raise ValueError("distance_pc must be positive")
        dm_reference = self._dm_reference()
        reference_distance = 10.0 ** (0.2 * dm_reference) * 10.0
        hscale = self.dust_scale_height_pc / (abs(np.sin(np.deg2rad(self.b_deg))) + 1.0e-4)
        denominator = 1.0 - np.exp(-reference_distance / hscale)
        fraction = (1.0 - np.exp(-distance_pc / hscale)) / denominator
        distance_modulus = 5.0 * np.log10(distance_pc) - 5.0
        return {
            band: float(distance_modulus + extinction * fraction)
            for band, extinction in self.extinction_at_reference.items()
        }

    def _dm_reference(self) -> float:
        if self.dm_reference is not None:
            return self.dm_reference
        return 14.3955 - 0.0239 * self.l_deg + 0.0122 * abs(self.b_deg) + 0.128


@dataclass(frozen=True)
class ExponentialDustModel:
    """JAX-evaluable counterpart of ``ExponentialDustOffsets`` for one CMD chart."""

    l_deg: float
    b_deg: float
    extinction_reference: tuple[float, float, float]
    dm_reference: float | None = None
    dust_scale_height_pc: float = 164.0

    @classmethod
    def from_exponential(
        cls,
        dust: ExponentialDustOffsets,
        coordinates: "CmdCoordinates",
    ) -> "ExponentialDustModel":
        extinction = dust.extinction_at_reference
        return cls(
            l_deg=dust.l_deg,
            b_deg=dust.b_deg,
            extinction_reference=(
                float(extinction.get(coordinates.reference_band, 0.0)),
                float(extinction.get(coordinates.blue_band, 0.0)),
                float(extinction.get(coordinates.red_band, 0.0)),
            ),
            dm_reference=dust.dm_reference,
            dust_scale_height_pc=dust.dust_scale_height_pc,
        )

    def offsets(self, ds_kpc):
        """Return JAX offsets ordered as ``(reference, blue, red)``."""

        import jax.numpy as jnp

        distance_pc = jnp.asarray(ds_kpc) * 1000.0
        dm_reference = self.dm_reference
        if dm_reference is None:
            dm_reference = 14.3955 - 0.0239 * self.l_deg + 0.0122 * abs(self.b_deg) + 0.128
        reference_distance = 10.0 ** (0.2 * dm_reference) * 10.0
        hscale = self.dust_scale_height_pc / (abs(np.sin(np.deg2rad(self.b_deg))) + 1.0e-4)
        fraction = (1.0 - jnp.exp(-distance_pc / hscale)) / (1.0 - np.exp(-reference_distance / hscale))
        distance_modulus = 5.0 * jnp.log10(distance_pc) - 5.0
        return distance_modulus + jnp.asarray(self.extinction_reference) * fraction


@dataclass(frozen=True)
class GenulensSourceModel:
    """Forward-source configuration used to make source-data evidence grids."""

    source_data: SourceDataModel = field(default_factory=SourceSelection)
    bands: tuple[str, ...] = ()
    population: SourcePopulation | None = None
    isochrone_table_path: str | None = None
    offset_provider: OffsetProvider | None = None
    min_initial_mass_msun: float = 0.09
    # None delegates the upper limit to genulens, which intersects the IMF
    # with the selected age/metallicity isochrone support.  A fixed 1 Msun
    # cap removes the turn-off and giant branch of old populations.
    max_initial_mass_msun: float | None = None
    # CMD likelihoods for a precise source can be dominated by short-lived
    # post-main-sequence phases.  A random IMF draw represents those phases
    # with one or zero rows, then histogram smoothing can turn that accident
    # into a spurious distance mode.  Use equal-IMF-probability quadrature
    # points instead.  This is intentionally much denser than the legacy
    # 4096 Monte-Carlo draws.
    cmd_imf_quadrature_points: int = 8192
    samples_per_population_point: int = 4096

    def build_evidence_grid(
        self,
        distance_pc: Sequence[float],
        *,
        cmd_prior: "CmdPriorTable | None" = None,
    ) -> "SourceEvidenceGrid":
        distances = np.asarray(distance_pc, dtype=float)
        if cmd_prior is not None:
            if not isinstance(self.source_data, SourceSelection):
                raise TypeError("a CMD table can produce hard-selection evidence only from SourceSelection")
            if self.offset_provider is None and _uses_apparent_photometry(self.source_data):
                raise ValueError(
                    "apparent source selection requires offset_provider with distance modulus and extinction"
                )
            return cmd_prior.evidence_for_selection(
                self.source_data,
                distances,
                offset_provider=self.offset_provider or (lambda component, distance: {}),
                component_indices=range(11),
            )
        if isinstance(self.source_data, SourceSelection) and not self.source_data.cuts:
            return SourceEvidenceGrid.unit_evidence(distances, n_components=11)
        if self.offset_provider is None and _uses_apparent_photometry(self.source_data):
            raise ValueError(
                "apparent source data requires offset_provider with distance modulus and extinction"
            )
        try:
            import genulens
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("an active forward source selection requires the genulens Python package") from exc
        generator = self._load_generator(genulens)
        return GenulensSourceEvidenceBuilder(
            generator=generator,
            genulens=genulens,
            source_data=self.source_data,
            offset_provider=self.offset_provider,
            min_initial_mass_msun=self.min_initial_mass_msun,
            max_initial_mass_msun=self.max_initial_mass_msun,
            samples_per_population_point=self.samples_per_population_point,
        ).build(distances)

    def build_cmd_prior(
        self,
        coordinates: "CmdCoordinates",
        *,
        reference_edges: Sequence[float],
        color_edges: Sequence[float],
        smoothing_sigma_bins: float = 0.75,
        smoothing: "CmdSmoothing | None" = None,
    ) -> "CmdPriorTable":
        """Build an intrinsic component-conditional CMD prior from genulens.

        The returned table is independent of line of sight. Convert it to
        apparent photometry only at evaluation time using the current distance
        modulus and extinction offsets.
        """

        try:
            import genulens
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("building a CMD prior requires the genulens Python package") from exc
        generator = self._load_generator(genulens)
        return GenulensCmdPriorBuilder(
            generator=generator,
            genulens=genulens,
            population_points_provider=self._population_points_provider(),
            min_initial_mass_msun=self.min_initial_mass_msun,
            max_initial_mass_msun=self.max_initial_mass_msun,
            imf_quadrature_points=self.cmd_imf_quadrature_points,
            samples_per_population_point=self.samples_per_population_point,
        ).build(
            coordinates,
            reference_edges=reference_edges,
            color_edges=color_edges,
            smoothing_sigma_bins=smoothing_sigma_bins,
            smoothing=smoothing,
        )

    def _load_generator(self, genulens: Any) -> Any:
        imf = None
        if self.population is not None and self.population.imf is not None:
            imf = genulens.IMFParameters()
            for name, value in self.population.imf.items():
                setattr(imf, name, float(value))
        if self.isochrone_table_path is not None:
            return genulens.ForwardSourceGenerator.load_table(self.isochrone_table_path, imf) if imf is not None else genulens.ForwardSourceGenerator.load_table(self.isochrone_table_path)
        if self.bands:
            return (
                genulens.ForwardSourceGenerator.load_default_for_bands(list(self.bands), imf)
                if imf is not None else genulens.ForwardSourceGenerator.load_default_for_bands(list(self.bands))
            )
        raise ValueError("bands must be specified when using a default isochrone table")

    def _population_points_provider(self) -> PopulationPointProvider | None:
        if self.population is None or self.population.age_metallicity_by_component is None:
            return None

        overrides = self.population.age_metallicity_by_component

        def provider(component: ComponentIndex) -> Sequence[AgeMetallicityPoint]:
            return overrides.get(component, ())

        return provider


def _uses_apparent_photometry(source_data: SourceDataModel) -> bool:
    if isinstance(source_data, SourceSelection):
        return any(
            isinstance(cut, (MagnitudeCut, ColorCut)) and cut.apparent
            for cut in source_data.cuts
        )
    return False


@dataclass(frozen=True)
class IsochroneSampleGrid:
    """Weighted isochrone samples for one Galactic source component.

    This class is deliberately agnostic about how the samples were produced.
    They may come from genulens' forward-source tables, PARSEC/MIST tables, or a
    user supplied population prior. ``weights`` should already include the IMF
    and age/metallicity population prior.
    """

    absolute_magnitudes: Mapping[str, np.ndarray]
    radius_rsun: np.ndarray
    weights: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        radius = np.asarray(self.radius_rsun, dtype=float)
        if weights.ndim != 1 or radius.shape != weights.shape:
            raise ValueError("weights and radius_rsun must be one-dimensional arrays with the same shape")
        for band, values in self.absolute_magnitudes.items():
            if np.asarray(values).shape != weights.shape:
                raise ValueError(f"absolute magnitude band {band!r} has shape inconsistent with weights")

    def selection_probability(
        self,
        selection: SourceSelection,
        *,
        distance_pc: float,
        magnitude_offsets: BandOffsets | None = None,
    ) -> float:
        weights = np.asarray(self.weights, dtype=float)
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0
        source = SourceObservables(
            absolute_magnitudes=self.absolute_magnitudes,
            radius_rsun=self.radius_rsun,
            distance_pc=distance_pc,
            magnitude_offsets=magnitude_offsets or {},
        )
        selected = selection.mask(source)
        return float(np.sum(weights[selected]) / total)

    def evidence(
        self,
        source_data: SourceDataModel,
        *,
        distance_pc: float,
        magnitude_offsets: BandOffsets | None = None,
    ) -> float:
        weights = np.asarray(self.weights, dtype=float)
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0
        source = SourceObservables(
            absolute_magnitudes=self.absolute_magnitudes,
            radius_rsun=self.radius_rsun,
            distance_pc=distance_pc,
            magnitude_offsets=magnitude_offsets or {},
        )
        log_weight = np.asarray(source_data.log_weight(source), dtype=float)
        finite = np.isfinite(log_weight) & (weights > 0.0)
        if not np.any(finite):
            return 0.0
        shift = float(np.max(log_weight[finite]))
        return float(np.exp(shift) * np.sum(weights[finite] * np.exp(log_weight[finite] - shift)) / total)

    def theta_posterior(
        self,
        source_data: SourceDataModel,
        *,
        distance_pc: float,
        magnitude_offsets: BandOffsets | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = SourceObservables(
            absolute_magnitudes=self.absolute_magnitudes,
            radius_rsun=self.radius_rsun,
            distance_pc=distance_pc,
            magnitude_offsets=magnitude_offsets or {},
        )
        log_weight = np.asarray(source_data.log_weight(source), dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        finite = np.isfinite(log_weight) & (weights > 0.0)
        posterior = np.zeros_like(weights)
        if np.any(finite):
            shift = float(np.max(log_weight[finite]))
            posterior[finite] = weights[finite] * np.exp(log_weight[finite] - shift)
            posterior /= posterior.sum()
        return source.theta_star_microarcsec, posterior

    def theta_distribution(
        self,
        selection: SourceSelection,
        *,
        distance_pc: float,
        theta_edges_microarcsec: np.ndarray,
        magnitude_offsets: BandOffsets | None = None,
    ) -> np.ndarray:
        weights = np.asarray(self.weights, dtype=float)
        source = SourceObservables(
            absolute_magnitudes=self.absolute_magnitudes,
            radius_rsun=self.radius_rsun,
            distance_pc=distance_pc,
            magnitude_offsets=magnitude_offsets or {},
        )
        selected = selection.mask(source)
        hist, _ = np.histogram(
            source.theta_star_microarcsec[selected],
            bins=np.asarray(theta_edges_microarcsec, dtype=float),
            weights=weights[selected],
            density=False,
        )
        norm = np.sum(hist)
        if norm <= 0.0:
            return np.zeros_like(hist, dtype=float)
        widths = np.diff(theta_edges_microarcsec)
        return hist / norm / widths


@dataclass(frozen=True)
class CmdCoordinates:
    """A two-dimensional CMD chart: one absolute magnitude and one colour."""

    reference_band: str
    blue_band: str
    red_band: str

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.reference_band, self.blue_band, self.red_band)))

    def absolute_values(self, absolute_magnitudes: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        reference = np.asarray(absolute_magnitudes[self.reference_band], dtype=float)
        color = np.asarray(absolute_magnitudes[self.blue_band], dtype=float) - np.asarray(
            absolute_magnitudes[self.red_band], dtype=float
        )
        return reference, color

    def apparent_to_absolute(
        self,
        reference_magnitude: float,
        color: float,
        magnitude_offsets: BandOffsets,
    ) -> tuple[float, float]:
        reference_offset = float(magnitude_offsets.get(self.reference_band, 0.0))
        color_offset = float(magnitude_offsets.get(self.blue_band, 0.0)) - float(
            magnitude_offsets.get(self.red_band, 0.0)
        )
        return reference_magnitude - reference_offset, color - color_offset

    @staticmethod
    def magnitude_from_flux(flux: float, zero_point: float) -> float:
        if flux <= 0.0:
            return float("inf")
        return float(zero_point - 2.5 * np.log10(flux))

    def apparent_from_fluxes(
        self,
        flux_blue: float,
        flux_red: float,
        *,
        zero_point_blue: float,
        zero_point_red: float,
    ) -> tuple[float, float]:
        if self.reference_band != self.red_band:
            raise ValueError(
                "two-flux conversion requires reference_band == red_band; evaluate the CMD prior in magnitudes instead"
            )
        blue = self.magnitude_from_flux(flux_blue, zero_point_blue)
        red = self.magnitude_from_flux(flux_red, zero_point_red)
        return red, blue - red

    @staticmethod
    def log_flux_jacobian(flux_blue: float, flux_red: float) -> float:
        """Log |d(m_red, m_blue-m_red) / d(F_blue, F_red)|."""

        if flux_blue <= 0.0 or flux_red <= 0.0:
            return float("-inf")
        magnitude_factor = 2.5 / np.log(10.0)
        return float(2.0 * np.log(magnitude_factor) - np.log(flux_blue) - np.log(flux_red))


@dataclass(frozen=True)
class CmdSmoothing:
    """Phenomenological intrinsic CMD width, in magnitudes rather than bins."""

    reference_sigma_mag: float = 0.0
    color_sigma_mag: float = 0.0

    def __post_init__(self) -> None:
        if self.reference_sigma_mag < 0.0 or self.color_sigma_mag < 0.0:
            raise ValueError("CMD smoothing widths must be non-negative")


@dataclass(frozen=True)
class CmdPriorTable:
    """Component-conditional intrinsic CMD density table.

    The table is a density in ``(M_reference, M_blue - M_red)``. It is built
    from the source-population prior, not from photometric measurement errors.
    At an MCMC step, apparent values are shifted back to intrinsic coordinates
    using the current source distance and extinction, then evaluated by
    bilinear interpolation.
    """

    coordinates: CmdCoordinates
    reference_edges: np.ndarray
    color_edges: np.ndarray
    density_by_component: np.ndarray
    log_radius_moment_by_component: np.ndarray | None = None
    log_radius_square_moment_by_component: np.ndarray | None = None
    component_indices: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reference_edges = np.asarray(self.reference_edges, dtype=float)
        color_edges = np.asarray(self.color_edges, dtype=float)
        density = np.asarray(self.density_by_component, dtype=float)
        if reference_edges.ndim != 1 or color_edges.ndim != 1:
            raise ValueError("CMD edges must be one-dimensional")
        if len(reference_edges) < 2 or len(color_edges) < 2:
            raise ValueError("CMD edges must each contain at least two values")
        if np.any(np.diff(reference_edges) <= 0.0) or np.any(np.diff(color_edges) <= 0.0):
            raise ValueError("CMD edges must be strictly increasing")
        expected_shape = (density.shape[0], len(reference_edges) - 1, len(color_edges) - 1)
        if density.ndim != 3 or density.shape != expected_shape:
            raise ValueError("density_by_component must have shape (component, reference_bin, color_bin)")
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("CMD density must be finite and non-negative")
        for name, moment in (
            ("log_radius_moment_by_component", self.log_radius_moment_by_component),
            ("log_radius_square_moment_by_component", self.log_radius_square_moment_by_component),
        ):
            if moment is not None:
                values = np.asarray(moment, dtype=float)
                if values.shape != density.shape:
                    raise ValueError(f"{name} must have the same shape as density_by_component")
                if np.any(~np.isfinite(values)):
                    raise ValueError(f"{name} must be finite")

    @classmethod
    def from_isochrone_samples(
        cls,
        samples_by_component: Mapping[int, IsochroneSampleGrid],
        coordinates: CmdCoordinates,
        *,
        reference_edges: Sequence[float],
        color_edges: Sequence[float],
        smoothing_sigma_bins: float = 0.75,
        smoothing: CmdSmoothing | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "CmdPriorTable":
        if smoothing_sigma_bins < 0.0:
            raise ValueError("smoothing_sigma_bins must be non-negative")
        reference_edges_array = np.asarray(reference_edges, dtype=float)
        color_edges_array = np.asarray(color_edges, dtype=float)
        components = np.asarray(sorted(samples_by_component), dtype=int)
        density = np.zeros(
            (len(components), len(reference_edges_array) - 1, len(color_edges_array) - 1),
            dtype=float,
        )
        log_radius_moment = np.zeros_like(density)
        log_radius_square_moment = np.zeros_like(density)
        bin_areas = np.diff(reference_edges_array)[:, None] * np.diff(color_edges_array)[None, :]
        for column, component in enumerate(components):
            samples = samples_by_component[int(component)]
            reference, color = coordinates.absolute_values(samples.absolute_magnitudes)
            weights = np.asarray(samples.weights, dtype=float)
            counts, _, _ = np.histogram2d(
                reference,
                color,
                bins=(reference_edges_array, color_edges_array),
                weights=weights,
            )
            log_radius = np.log(np.asarray(samples.radius_rsun, dtype=float))
            first_moment, _, _ = np.histogram2d(
                reference, color, bins=(reference_edges_array, color_edges_array), weights=weights * log_radius
            )
            second_moment, _, _ = np.histogram2d(
                reference, color, bins=(reference_edges_array, color_edges_array), weights=weights * log_radius**2
            )
            if smoothing is None:
                sigma_reference = sigma_color = smoothing_sigma_bins
            else:
                sigma_reference = smoothing.reference_sigma_mag / np.median(np.diff(reference_edges_array))
                sigma_color = smoothing.color_sigma_mag / np.median(np.diff(color_edges_array))
            if sigma_reference > 0.0 or sigma_color > 0.0:
                counts = _gaussian_smooth_2d(counts, sigma_reference, sigma_color)
                first_moment = _gaussian_smooth_2d(first_moment, sigma_reference, sigma_color)
                second_moment = _gaussian_smooth_2d(second_moment, sigma_reference, sigma_color)
            normalisation = float(np.sum(counts))
            if normalisation > 0.0:
                density[column] = counts / normalisation / bin_areas
                log_radius_moment[column] = first_moment / normalisation / bin_areas
                log_radius_square_moment[column] = second_moment / normalisation / bin_areas
        return cls(
            coordinates=coordinates,
            reference_edges=reference_edges_array,
            color_edges=color_edges_array,
            density_by_component=density,
            log_radius_moment_by_component=log_radius_moment,
            log_radius_square_moment_by_component=log_radius_square_moment,
            component_indices=components,
            metadata={
                **dict(metadata or {}),
                "smoothing": (
                    {"reference_sigma_mag": smoothing.reference_sigma_mag, "color_sigma_mag": smoothing.color_sigma_mag}
                    if smoothing is not None
                    else {"sigma_bins": smoothing_sigma_bins}
                ),
            },
        )

    def density(
        self,
        component: int,
        reference_magnitude: float,
        color: float,
        *,
        distance_pc: float,
        magnitude_offsets: BandOffsets,
    ) -> float:
        absolute_reference, absolute_color = self.coordinates.apparent_to_absolute(
            reference_magnitude,
            color,
            magnitude_offsets,
        )
        del distance_pc  # Kept in the public signature to make the conditioning explicit.
        return _bilinear_density(
            self._density_for_component(component),
            self.reference_edges,
            self.color_edges,
            absolute_reference,
            absolute_color,
        )

    def density_from_fluxes(
        self,
        component: int,
        flux_blue: float,
        flux_red: float,
        *,
        zero_point_blue: float,
        zero_point_red: float,
        distance_pc: float,
        magnitude_offsets: BandOffsets,
        include_flux_jacobian: bool = True,
    ) -> float:
        reference, color = self.coordinates.apparent_from_fluxes(
            flux_blue,
            flux_red,
            zero_point_blue=zero_point_blue,
            zero_point_red=zero_point_red,
        )
        value = self.density(
            component,
            reference,
            color,
            distance_pc=distance_pc,
            magnitude_offsets=magnitude_offsets,
        )
        if not include_flux_jacobian:
            return value
        log_jacobian = self.coordinates.log_flux_jacobian(flux_blue, flux_red)
        return float(value * np.exp(log_jacobian)) if np.isfinite(log_jacobian) else 0.0

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            reference_band=np.asarray(self.coordinates.reference_band),
            blue_band=np.asarray(self.coordinates.blue_band),
            red_band=np.asarray(self.coordinates.red_band),
            reference_edges=np.asarray(self.reference_edges, dtype=float),
            color_edges=np.asarray(self.color_edges, dtype=float),
            density_by_component=np.asarray(self.density_by_component, dtype=float),
            log_radius_moment_by_component=(
                np.asarray(self.log_radius_moment_by_component, dtype=float)
                if self.log_radius_moment_by_component is not None else np.asarray([])
            ),
            log_radius_square_moment_by_component=(
                np.asarray(self.log_radius_square_moment_by_component, dtype=float)
                if self.log_radius_square_moment_by_component is not None else np.asarray([])
            ),
            component_indices=np.asarray(
                self.component_indices
                if self.component_indices is not None
                else np.arange(self.density_by_component.shape[0]),
                dtype=int,
            ),
            metadata=np.asarray(dict(self.metadata), dtype=object),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "CmdPriorTable":
        data = np.load(path, allow_pickle=True)
        metadata_raw = data.get("metadata")
        metadata = dict(metadata_raw.item()) if metadata_raw is not None and metadata_raw.shape == () else {}
        return cls(
            coordinates=CmdCoordinates(
                reference_band=str(data["reference_band"].item()),
                blue_band=str(data["blue_band"].item()),
                red_band=str(data["red_band"].item()),
            ),
            reference_edges=data["reference_edges"],
            color_edges=data["color_edges"],
            density_by_component=data["density_by_component"],
            log_radius_moment_by_component=(
                data["log_radius_moment_by_component"]
                if "log_radius_moment_by_component" in data and data["log_radius_moment_by_component"].size else None
            ),
            log_radius_square_moment_by_component=(
                data["log_radius_square_moment_by_component"]
                if "log_radius_square_moment_by_component" in data and data["log_radius_square_moment_by_component"].size else None
            ),
            component_indices=data.get("component_indices"),
            metadata=metadata,
        )

    def evaluator(
        self,
        *,
        interpolation: str = "bilinear",
        log_cubic_floor_relative: float = 1.0e-12,
        log_cubic_padding_cells: int = 3,
    ):
        """Return the canonical histogram-backend representation of this CMD table.

        ``log_cubic`` is a finite, C1 interpolation of log CMD density for
        gradient-based inference; ``bilinear`` preserves legacy behaviour.
        """

        from gapmoe.density.histogram_backend import CmdPriorEvaluator

        return CmdPriorEvaluator.from_table(
            self,
            interpolation=interpolation,
            log_cubic_floor_relative=log_cubic_floor_relative,
            log_cubic_padding_cells=log_cubic_padding_cells,
        )

    def evidence_for_selection(
        self,
        selection: SourceSelection,
        distance_pc: Sequence[float],
        *,
        offset_provider: OffsetProvider,
        component_indices: Sequence[int] | None = None,
    ) -> "SourceEvidenceGrid":
        """Integrate this CMD density over an apparent magnitude-colour cut.

        This is the hard-selection counterpart of direct CMD density
        evaluation. Only cuts represented by this table's reference magnitude
        and colour are accepted, which keeps both paths on exactly the same
        source-population model.
        """

        reference_bounds = [-np.inf, np.inf]
        color_bounds = [-np.inf, np.inf]
        for cut in selection.cuts:
            if isinstance(cut, MagnitudeCut) and cut.apparent and cut.band == self.coordinates.reference_band:
                reference_bounds[0] = max(reference_bounds[0], cut.minimum)
                reference_bounds[1] = min(reference_bounds[1], cut.maximum)
            elif (
                isinstance(cut, ColorCut)
                and cut.apparent
                and cut.blue_band == self.coordinates.blue_band
                and cut.red_band == self.coordinates.red_band
            ):
                color_bounds[0] = max(color_bounds[0], cut.minimum)
                color_bounds[1] = min(color_bounds[1], cut.maximum)
            else:
                raise ValueError("selection cannot be represented by this CMD table")
        components = (
            np.asarray(self.component_indices, dtype=int)
            if component_indices is None and self.component_indices is not None
            else np.asarray(component_indices if component_indices is not None else np.arange(self.density_by_component.shape[0]), dtype=int)
        )
        distances = np.asarray(distance_pc, dtype=float)
        evidence = np.zeros((len(distances), len(components)), dtype=float)
        for i, distance in enumerate(distances):
            for j, component in enumerate(components):
                offsets = offset_provider(int(component), float(distance))
                reference_offset = float(offsets.get(self.coordinates.reference_band, 0.0))
                color_offset = float(offsets.get(self.coordinates.blue_band, 0.0)) - float(
                    offsets.get(self.coordinates.red_band, 0.0)
                )
                evidence[i, j] = self._integrate_rectangle(
                    int(component),
                    reference_bounds[0] - reference_offset,
                    reference_bounds[1] - reference_offset,
                    color_bounds[0] - color_offset,
                    color_bounds[1] - color_offset,
                )
        return SourceEvidenceGrid(
            distance_pc=distances,
            evidence_by_component=evidence,
            component_indices=components,
            metadata={"backend": "CmdPriorTable", "selection": repr(selection)},
        )

    def _integrate_rectangle(
        self,
        component: int,
        reference_minimum: float,
        reference_maximum: float,
        color_minimum: float,
        color_maximum: float,
    ) -> float:
        if reference_maximum <= reference_minimum or color_maximum <= color_minimum:
            return 0.0
        reference_overlap = np.maximum(
            0.0,
            np.minimum(self.reference_edges[1:], reference_maximum)
            - np.maximum(self.reference_edges[:-1], reference_minimum),
        )
        color_overlap = np.maximum(
            0.0,
            np.minimum(self.color_edges[1:], color_maximum) - np.maximum(self.color_edges[:-1], color_minimum),
        )
        return float(np.sum(self._density_for_component(component) * reference_overlap[:, None] * color_overlap[None, :]))

    def _density_for_component(self, component: int) -> np.ndarray:
        if self.component_indices is None:
            if 0 <= component < self.density_by_component.shape[0]:
                return self.density_by_component[component]
        else:
            matches = np.flatnonzero(np.asarray(self.component_indices) == component)
            if matches.size:
                return self.density_by_component[int(matches[0])]
        raise KeyError(f"CMD prior table does not contain component {component}")


def _gaussian_smooth_2d(values: np.ndarray, sigma_reference: float, sigma_color: float) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    for axis, sigma_bins in enumerate((sigma_reference, sigma_color)):
        if sigma_bins <= 0.0:
            continue
        radius = max(1, int(np.ceil(4.0 * sigma_bins)))
        positions = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (positions / sigma_bins) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(out, [(radius, radius) if index == axis else (0, 0) for index in range(2)])
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)
    return out


def _bilinear_density(
    density: np.ndarray,
    reference_edges: np.ndarray,
    color_edges: np.ndarray,
    reference: float,
    color: float,
) -> float:
    reference_centers = 0.5 * (reference_edges[:-1] + reference_edges[1:])
    color_centers = 0.5 * (color_edges[:-1] + color_edges[1:])
    if reference < reference_centers[0] or reference > reference_centers[-1]:
        return 0.0
    if color < color_centers[0] or color > color_centers[-1]:
        return 0.0
    i1 = int(np.searchsorted(reference_centers, reference, side="right"))
    j1 = int(np.searchsorted(color_centers, color, side="right"))
    i0, i1 = max(0, i1 - 1), min(len(reference_centers) - 1, i1)
    j0, j1 = max(0, j1 - 1), min(len(color_centers) - 1, j1)
    if i0 == i1 and j0 == j1:
        return float(density[i0, j0])
    tx = 0.0 if i0 == i1 else (reference - reference_centers[i0]) / (reference_centers[i1] - reference_centers[i0])
    ty = 0.0 if j0 == j1 else (color - color_centers[j0]) / (color_centers[j1] - color_centers[j0])
    return float(
        (1.0 - tx) * (1.0 - ty) * density[i0, j0]
        + tx * (1.0 - ty) * density[i1, j0]
        + (1.0 - tx) * ty * density[i0, j1]
        + tx * ty * density[i1, j1]
    )


def genulens_population_component(component: ComponentIndex) -> ComponentIndex:
    """Map a density component onto genulens' default source population."""

    # genulens uses the thick-disk stellar population for halo sources.
    return 7 if component == 10 else component


@dataclass
class GenulensCmdPriorBuilder:
    """Sample genulens' age/metallicity/IMF source population into a CMD table."""

    generator: Any
    genulens: Any
    population_points_provider: PopulationPointProvider | None = None
    population_component_mapper: PopulationComponentMapper = genulens_population_component
    min_initial_mass_msun: float = 0.09
    max_initial_mass_msun: float | None = None
    imf_quadrature_points: int = 8192
    samples_per_population_point: int = 4096

    def build(
        self,
        coordinates: CmdCoordinates,
        *,
        reference_edges: Sequence[float],
        color_edges: Sequence[float],
        smoothing_sigma_bins: float = 0.75,
        smoothing: CmdSmoothing | None = None,
        component_indices: Sequence[ComponentIndex] = tuple(range(11)),
    ) -> CmdPriorTable:
        if self.imf_quadrature_points <= 0:
            raise ValueError("imf_quadrature_points must be positive")
        samples = {
            int(component): self._sample_component(int(component), coordinates, index)
            for index, component in enumerate(component_indices)
        }
        return CmdPriorTable.from_isochrone_samples(
            samples,
            coordinates,
            reference_edges=reference_edges,
            color_edges=color_edges,
            smoothing_sigma_bins=smoothing_sigma_bins,
            smoothing=smoothing,
            metadata={
                "backend": "genulens.ForwardSourceGenerator",
                "cmd_sampling": "deterministic_equal_imf_probability_quadrature",
                "imf_quadrature_points": self.imf_quadrature_points,
                "samples_per_population_point": self.samples_per_population_point,
                "min_initial_mass_msun": self.min_initial_mass_msun,
                "max_initial_mass_msun": self.max_initial_mass_msun,
            },
        )

    def _sample_component(
        self,
        component: ComponentIndex,
        coordinates: CmdCoordinates,
        component_position: int,
    ) -> IsochroneSampleGrid:
        points = self._population_points(component)
        absolute_magnitudes: dict[str, list[np.ndarray]] = {band: [] for band in coordinates.bands}
        radii: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for point_index, point in enumerate(points):
            population_weight = float(point.weight)
            if population_weight <= 0.0:
                continue
            query = self._query(component, point)
            quadrature = getattr(self.generator, "imf_quadrature", None)
            if quadrature is None:
                raise RuntimeError(
                    "The installed genulens binding lacks ForwardSourceGenerator.imf_quadrature. "
                    "Build genulens with the source-CMD IMF-quadrature API; random source draws are "
                    "not valid for a precision CMD likelihood."
                )
            result = quadrature(query, self.imf_quadrature_points)
            observables = GenulensSourceEvidenceBuilder._samples_from_result(result, 10.0, {})
            for band in coordinates.bands:
                try:
                    absolute_magnitudes[band].append(np.asarray(observables.absolute_magnitudes[band], dtype=float))
                except KeyError as exc:
                    raise ValueError(f"genulens source table is missing CMD band {band!r}") from exc
            radii.append(np.asarray(observables.radius_rsun, dtype=float))
            n_rows = len(observables.radius_rsun)
            # imf_quadrature uses equal-probability bins of the IMF CDF, so
            # every returned point has the same quadrature weight.
            weights.append(np.full(n_rows, population_weight / n_rows, dtype=float))
        if not weights:
            raise ValueError(f"source population has no positive-weight points for component {component}")
        return IsochroneSampleGrid(
            absolute_magnitudes={band: np.concatenate(values) for band, values in absolute_magnitudes.items()},
            radius_rsun=np.concatenate(radii),
            weights=np.concatenate(weights),
        )

    def _population_points(self, component: ComponentIndex) -> Sequence[Any]:
        prior_component = self.population_component_mapper(component)
        if self.population_points_provider is not None:
            points = self.population_points_provider(prior_component)
            if points:
                return points
        return self.genulens.SourcePopulationPrior.points_for_component(prior_component)

    def _query(self, component: ComponentIndex, point: Any) -> Any:
        query = self.genulens.ForwardSourceQuery()
        query.component_index = self.population_component_mapper(component)
        query.distance_pc = 10.0
        query.min_initial_mass_msun = self.min_initial_mass_msun
        if self.max_initial_mass_msun is not None:
            query.max_initial_mass_msun = self.max_initial_mass_msun
        query.use_default_log_age = False
        query.log_age = point.log_age
        query.use_default_metallicity = False
        query.metallicity_mh = point.metallicity_mh
        return query


OffsetProvider = Callable[[ComponentIndex, float], BandOffsets]
PopulationPointProvider = Callable[[ComponentIndex], Sequence[Any]]
PopulationComponentMapper = Callable[[ComponentIndex], ComponentIndex]


@dataclass
class GenulensSourceEvidenceBuilder:
    """Build source-data evidence grids from genulens' forward-source generator.

    Magnitude-only selections use genulens' exact IMF/isochrone interval
    integration.  Selections involving colour or angular source radius use
    draws from the same population model, because those predicates are not yet
    represented by genulens' ``MagnitudeSelection`` API.

    ``offset_provider`` should return apparent-magnitude offsets (distance
    modulus plus extinction) for a component and distance.  Its separation
    from this builder is intentional: it lets a caller share exactly the dust
    prescription used to produce a line-of-sight density table.
    """

    generator: Any
    genulens: Any
    source_data: SourceDataModel
    offset_provider: OffsetProvider | None = None
    population_points_provider: PopulationPointProvider | None = None
    population_component_mapper: PopulationComponentMapper = genulens_population_component
    min_initial_mass_msun: float = 0.09
    max_initial_mass_msun: float | None = None
    samples_per_population_point: int = 4096

    @classmethod
    def from_genulens(
        cls,
        generator: Any,
        source_data: SourceDataModel,
        **kwargs: Any,
    ) -> "GenulensSourceEvidenceBuilder":
        """Create a builder using the installed optional ``genulens`` binding."""

        try:
            import genulens
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "building source evidence requires the genulens Python package"
            ) from exc
        return cls(generator=generator, genulens=genulens, source_data=source_data, **kwargs)

    def build(
        self,
        distance_pc: Sequence[float],
        *,
        component_indices: Sequence[ComponentIndex] = tuple(range(11)),
    ) -> "SourceEvidenceGrid":
        distances = np.asarray(distance_pc, dtype=float)
        if distances.ndim != 1 or np.any(distances <= 0.0):
            raise ValueError("distance_pc must be a one-dimensional positive grid")
        components = np.asarray(component_indices, dtype=int)
        if components.ndim != 1:
            raise ValueError("component_indices must be one-dimensional")
        if self.samples_per_population_point <= 0:
            raise ValueError("samples_per_population_point must be positive")

        evidence = np.zeros((len(distances), len(components)), dtype=float)
        exact = self._supports_exact_magnitude_selection()
        for j, component in enumerate(components):
            for i, distance in enumerate(distances):
                if exact:
                    evidence[i, j] = self._exact_evidence(int(component), float(distance))
                else:
                    evidence[i, j] = self._sampled_evidence(int(component), float(distance), i, j)

        return SourceEvidenceGrid(
            distance_pc=distances,
            evidence_by_component=np.maximum(evidence, 0.0),
            component_indices=components,
            metadata={
                "backend": "genulens.ForwardSourceGenerator",
                "method": "isochrone_interval" if exact else "forward_source_monte_carlo",
                "source_data": repr(self.source_data),
                "min_initial_mass_msun": self.min_initial_mass_msun,
                "max_initial_mass_msun": self.max_initial_mass_msun,
                "samples_per_population_point": self.samples_per_population_point if not exact else None,
            },
        )

    def _supports_exact_magnitude_selection(self) -> bool:
        return isinstance(self.source_data, SourceSelection) and all(
            isinstance(cut, MagnitudeCut) for cut in self.source_data.cuts
        )

    def _population_points(self, component: ComponentIndex) -> Sequence[Any]:
        prior_component = self.population_component_mapper(component)
        if self.population_points_provider is not None:
            return self.population_points_provider(prior_component)
        return self.genulens.SourcePopulationPrior.points_for_component(prior_component)

    def _query(self, component: ComponentIndex, distance_pc: float, point: Any) -> Any:
        query = self.genulens.ForwardSourceQuery()
        query.component_index = self.population_component_mapper(component)
        query.distance_pc = distance_pc
        query.min_initial_mass_msun = self.min_initial_mass_msun
        if self.max_initial_mass_msun is not None:
            query.max_initial_mass_msun = self.max_initial_mass_msun
        query.use_default_log_age = False
        query.log_age = point.log_age
        query.use_default_metallicity = False
        query.metallicity_mh = point.metallicity_mh
        return query

    def _magnitude_selections(self, component: ComponentIndex, distance_pc: float) -> list[Any]:
        offsets = self.offset_provider(component, distance_pc) if self.offset_provider else {}
        selections = []
        assert isinstance(self.source_data, SourceSelection)
        for cut in self.source_data.cuts:
            assert isinstance(cut, MagnitudeCut)
            item = self.genulens.MagnitudeSelection()
            item.band = cut.band
            item.min_magnitude = cut.minimum
            item.max_magnitude = cut.maximum
            item.magnitude_offset = float(offsets.get(cut.band, 0.0)) if cut.apparent else 0.0
            selections.append(item)
        return selections

    def _exact_evidence(self, component: ComponentIndex, distance_pc: float) -> float:
        points = self._population_points(component)
        if not points:
            return 0.0
        weighted_probability = 0.0
        total_weight = 0.0
        selections = self._magnitude_selections(component, distance_pc)
        for point in points:
            weight = float(point.weight)
            if weight <= 0.0:
                continue
            query = self._query(component, distance_pc, point)
            query.magnitude_selections = selections
            weighted_probability += weight * float(self.generator.selection_probability(query))
            total_weight += weight
        return weighted_probability / total_weight if total_weight > 0.0 else 0.0

    def _sampled_evidence(
        self,
        component: ComponentIndex,
        distance_pc: float,
        distance_index: int,
        component_index: int,
    ) -> float:
        points = self._population_points(component)
        if not points:
            return 0.0
        offsets = self.offset_provider(component, distance_pc) if self.offset_provider else {}
        weighted_probability = 0.0
        total_weight = 0.0
        for point_index, point in enumerate(points):
            weight = float(point.weight)
            if weight <= 0.0:
                continue
            query = self._query(component, distance_pc, point)
            # This seed only controls the fallback estimator. Exact tables do
            # not depend on it, and each grid cell gets an independent stream.
            seed = 1 + distance_index + 1009 * component_index + 65537 * point_index
            result = self.generator.sample_many(query, self.samples_per_population_point, seed)
            samples = self._samples_from_result(result, distance_pc, offsets)
            log_weight = np.asarray(self.source_data.log_weight(samples), dtype=float)
            finite = np.isfinite(log_weight)
            if np.any(finite):
                shift = float(np.max(log_weight[finite]))
                mean_weight = np.exp(shift) * np.mean(np.exp(log_weight[finite] - shift))
                weighted_probability += weight * float(mean_weight) * float(np.mean(finite))
            total_weight += weight
        return weighted_probability / total_weight if total_weight > 0.0 else 0.0

    @staticmethod
    def _samples_from_result(result: Any, distance_pc: float, offsets: BandOffsets) -> SourceObservables:
        columns = list(result.columns)
        rows = np.asarray(result.to_numpy(), dtype=float)
        if rows.ndim != 2:
            raise ValueError("genulens ForwardSourceResult.to_numpy() must return a two-dimensional array")
        try:
            radius = rows[:, columns.index("R_S")]
        except ValueError as exc:
            raise ValueError("genulens ForwardSourceResult is missing R_S") from exc
        magnitudes = {
            name[2:-2]: rows[:, index]
            for index, name in enumerate(columns)
            if name.startswith("M_") and name.endswith("_S")
        }
        return SourceObservables(
            absolute_magnitudes=magnitudes,
            radius_rsun=radius,
            distance_pc=distance_pc,
            magnitude_offsets=offsets,
        )


@dataclass(frozen=True)
class SourceEvidenceGrid:
    """Distance-grid representation of ``p(source data | component, distance)``.

    A hard survey selection produces values in [0, 1]. Apparent photometry
    likelihoods may instead have density-valued evidence, so this class only
    requires non-negative weights.
    """

    distance_pc: np.ndarray
    evidence_by_component: np.ndarray
    component_indices: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        distance = np.asarray(self.distance_pc, dtype=float)
        evidence = np.asarray(self.evidence_by_component, dtype=float)
        if distance.ndim != 1:
            raise ValueError("distance_pc must be one-dimensional")
        if distance.size == 0 or np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
            raise ValueError("distance_pc must contain finite positive values")
        if np.any(np.diff(distance) <= 0.0):
            raise ValueError("distance_pc must be strictly increasing")
        if evidence.ndim != 2 or evidence.shape[0] != distance.shape[0]:
            raise ValueError("evidence_by_component must have shape (n_distance, n_component)")
        if np.any(~np.isfinite(evidence)) or np.any(evidence < 0.0):
            raise ValueError("source evidence must be finite and non-negative")

    @classmethod
    def from_source_samples(
        cls,
        samples_by_component: Mapping[int, IsochroneSampleGrid],
        distance_pc: Sequence[float],
        source_data: SourceDataModel,
        *,
        offset_provider: OffsetProvider | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "SourceEvidenceGrid":
        distances = np.asarray(distance_pc, dtype=float)
        components = np.asarray(sorted(samples_by_component), dtype=int)
        evidence = np.zeros((len(distances), len(components)), dtype=float)
        for j, component in enumerate(components):
            samples = samples_by_component[int(component)]
            for i, distance in enumerate(distances):
                offsets = offset_provider(int(component), float(distance)) if offset_provider else None
                evidence[i, j] = samples.evidence(
                    source_data,
                    distance_pc=float(distance),
                    magnitude_offsets=offsets,
                )
        return cls(
            distance_pc=distances,
            evidence_by_component=evidence,
            component_indices=components,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def unit_evidence(cls, distance_pc: Sequence[float], n_components: int) -> "SourceEvidenceGrid":
        return cls(
            distance_pc=np.asarray(distance_pc, dtype=float),
            evidence_by_component=np.ones((len(distance_pc), n_components), dtype=float),
            component_indices=np.arange(n_components),
        )

    def evidence(self, component: int, distance_kpc: float) -> float:
        component_index = self._column_for_component(component)
        return float(
            np.interp(
                distance_kpc * 1000.0,
                self.distance_pc,
                self.evidence_by_component[:, component_index],
                left=0.0,
                right=0.0,
            )
        )

    def evidence_on(self, distance_pc: np.ndarray, n_components: int) -> np.ndarray:
        out = np.zeros((len(distance_pc), n_components), dtype=float)
        for component in range(n_components):
            column = self._column_for_component(component, missing_ok=True)
            if column is None:
                continue
            out[:, component] = np.interp(
                distance_pc,
                self.distance_pc,
                self.evidence_by_component[:, column],
                left=0.0,
                right=0.0,
            )
        return out

    def save_npz(self, path: str | Path) -> None:
        metadata = np.asarray(dict(self.metadata), dtype=object)
        np.savez(
            path,
            distance_pc=np.asarray(self.distance_pc, dtype=float),
            evidence_by_component=np.asarray(self.evidence_by_component, dtype=float),
            component_indices=np.asarray(
                self.component_indices
                if self.component_indices is not None
                else np.arange(self.evidence_by_component.shape[1]),
                dtype=int,
            ),
            metadata=metadata,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SourceEvidenceGrid":
        data = np.load(path, allow_pickle=True)
        metadata_raw = data.get("metadata")
        metadata = dict(metadata_raw.item()) if metadata_raw is not None and metadata_raw.shape == () else {}
        return cls(
            distance_pc=data["distance_pc"],
            evidence_by_component=data["evidence_by_component"],
            component_indices=data.get("component_indices"),
            metadata=metadata,
        )

    def _column_for_component(self, component: int, *, missing_ok: bool = False) -> int | None:
        if self.component_indices is None:
            if 0 <= component < self.evidence_by_component.shape[1]:
                return int(component)
        else:
            matches = np.flatnonzero(np.asarray(self.component_indices) == component)
            if matches.size:
                return int(matches[0])
        if missing_ok:
            return None
        raise KeyError(f"source evidence does not contain component {component}")


@dataclass(frozen=True)
class ConditionedSourceDensity:
    """Source density after applying source-data evidence."""

    distance_pc: np.ndarray
    source_density_by_component: np.ndarray
    source_density: np.ndarray
    source_norm: float

    @classmethod
    def from_base_density(
        cls,
        distance_pc: Sequence[float],
        source_density_by_component: np.ndarray,
        evidence: SourceEvidenceGrid,
    ) -> "ConditionedSourceDensity":
        distance = np.asarray(distance_pc, dtype=float)
        base = np.asarray(source_density_by_component, dtype=float)
        if base.shape[0] != distance.shape[0]:
            raise ValueError("source_density_by_component must align with distance_pc")
        weights = evidence.evidence_on(distance, base.shape[1])
        selected = base * weights
        total = selected.sum(axis=1)
        return cls(
            distance_pc=distance,
            source_density_by_component=selected,
            source_density=total,
            source_norm=float(np.trapezoid(total, distance)),
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

    def component_weights(self, ds_kpc: float) -> np.ndarray:
        vals = np.array(
            [
                np.interp(ds_kpc * 1000.0, self.distance_pc, self.source_density_by_component[:, i], left=0.0, right=0.0)
                for i in range(self.source_density_by_component.shape[1])
            ]
        )
        total = vals.sum()
        if total <= 0.0:
            return np.zeros_like(vals)
        return vals / total
