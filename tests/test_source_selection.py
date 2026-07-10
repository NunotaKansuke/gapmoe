from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gapmoe.density import HistogramDensity
from gapmoe.density.histogram_tables import MurelHistogram
from gapmoe.source_selection import (
    ColorCut,
    CmdCoordinates,
    CmdPriorTable,
    GenulensSourceEvidenceBuilder,
    GenulensSourceModel,
    GenulensCmdPriorBuilder,
    ExponentialDustOffsets,
    IsochroneSampleGrid,
    MagnitudeCut,
    MagnitudeMeasurement,
    ConditionedSourceDensity,
    SourceEvidenceGrid,
    SourcePhotometry,
    SourceSelection,
    ExponentialDustModel,
    angular_radius_microarcsec,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "small_source_default"


class _FakeForwardSourceQuery:
    def __init__(self) -> None:
        self.component_index = -1
        self.distance_pc = 0.0
        self.min_initial_mass_msun = 0.0
        self.max_initial_mass_msun = 0.0
        self.use_default_log_age = True
        self.log_age = 0.0
        self.use_default_metallicity = True
        self.metallicity_mh = 0.0
        self.magnitude_selections = []


class _FakeMagnitudeSelection:
    def __init__(self) -> None:
        self.band = ""
        self.min_magnitude = 0.0
        self.max_magnitude = 0.0
        self.magnitude_offset = 0.0


class _FakeGenerator:
    def __init__(self) -> None:
        self.queries = []

    def selection_probability(self, query):
        self.queries.append(query)
        return 0.2 + 0.1 * query.log_age

    def sample_many(self, query, n_sources, seed):
        self.queries.append(query)
        rows = np.tile(np.asarray([query.component_index, query.distance_pc, 0, 0, 0, 0, 2.0, 0, 0, 0, 3.0, 4.0]), (n_sources, 1))
        rows[::2, 6] = 0.1
        rows[::2, 11] = 5.0
        return SimpleNamespace(
            columns=["iS", "D_S", "logage_S", "MH_S", "M_S_ini", "M_S", "R_S", "teff_S", "logg_S", "theta_S", "M_Imag_S", "M_Vmag_S"],
            to_numpy=lambda: rows,
        )


def _fake_genulens(points):
    return SimpleNamespace(
        ForwardSourceQuery=_FakeForwardSourceQuery,
        MagnitudeSelection=_FakeMagnitudeSelection,
        SourcePopulationPrior=SimpleNamespace(points_for_component=lambda component: points[component]),
    )


def test_isochrone_sample_grid_supports_magnitude_and_color_cuts() -> None:
    samples = IsochroneSampleGrid(
        absolute_magnitudes={
            "Imag": np.asarray([3.0, 5.0, 8.0]),
            "Vmag": np.asarray([4.0, 6.5, 10.5]),
        },
        radius_rsun=np.asarray([8.0, 1.0, 0.3]),
        weights=np.asarray([1.0, 2.0, 1.0]),
    )
    selection = SourceSelection(
        cuts=(
            MagnitudeCut("Imag", 19.0, 20.0),
            ColorCut("Vmag", "Imag", 0.5, 2.0),
        )
    )
    distance_pc = 8_000.0
    distance_modulus = 5.0 * np.log10(distance_pc) - 5.0

    probability = samples.selection_probability(
        selection,
        distance_pc=distance_pc,
        magnitude_offsets={"Imag": distance_modulus, "Vmag": distance_modulus},
    )

    assert probability == pytest.approx(0.5)
    assert angular_radius_microarcsec(1.0, distance_pc) == pytest.approx(0.5813084076202696)


def test_photometry_likelihood_produces_a_theta_star_posterior() -> None:
    samples = IsochroneSampleGrid(
        absolute_magnitudes={"Imag": np.asarray([3.0, 5.0, 8.0])},
        radius_rsun=np.asarray([8.0, 1.0, 0.3]),
        weights=np.asarray([1.0, 2.0, 1.0]),
    )
    distance_pc = 8_000.0
    distance_modulus = 5.0 * np.log10(distance_pc) - 5.0
    photometry = SourcePhotometry(magnitudes=(MagnitudeMeasurement("Imag", 5.0 + distance_modulus, 0.03),))

    evidence = samples.evidence(photometry, distance_pc=distance_pc, magnitude_offsets={"Imag": distance_modulus})
    theta, posterior = samples.theta_posterior(
        photometry,
        distance_pc=distance_pc,
        magnitude_offsets={"Imag": distance_modulus},
    )

    assert evidence > 0.0
    assert posterior[1] > 0.999
    assert np.sum(theta * posterior) == pytest.approx(angular_radius_microarcsec(1.0, distance_pc))


def test_cmd_prior_table_evaluates_apparent_magnitude_color_and_flux_density() -> None:
    samples = IsochroneSampleGrid(
        absolute_magnitudes={
            "Imag": np.asarray([1.0, 3.0]),
            "Vmag": np.asarray([1.5, 4.0]),
        },
        radius_rsun=np.asarray([1.0, 1.0]),
        weights=np.asarray([1.0, 3.0]),
    )
    table = CmdPriorTable.from_isochrone_samples(
        {8: samples},
        CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=np.asarray([0.0, 2.0, 4.0]),
        color_edges=np.asarray([0.0, 1.0, 2.0]),
        smoothing_sigma_bins=0.0,
    )

    area = 2.0
    assert np.sum(table.density_by_component[0] * area) == pytest.approx(1.0)

    offsets = {"Imag": 14.5, "Vmag": 15.0}
    density = table.density(8, 15.5, 1.0, distance_pc=8_000.0, magnitude_offsets=offsets)
    assert density == pytest.approx(0.125)

    flux_blue = 10.0 ** ((25.0 - 16.5) / 2.5)
    flux_red = 10.0 ** ((25.0 - 15.5) / 2.5)
    flux_density = table.density_from_fluxes(
        8,
        flux_blue,
        flux_red,
        zero_point_blue=25.0,
        zero_point_red=25.0,
        distance_pc=8_000.0,
        magnitude_offsets=offsets,
    )
    expected_jacobian = np.exp(table.coordinates.log_flux_jacobian(flux_blue, flux_red))
    assert flux_density == pytest.approx(density * expected_jacobian)

    evidence = table.evidence_for_selection(
        SourceSelection(cuts=(MagnitudeCut("Imag", 15.0, 16.0), ColorCut("Vmag", "Imag", 0.75, 1.25))),
        [8_000.0],
        offset_provider=lambda component, distance: offsets,
        component_indices=[8],
    )
    assert evidence.evidence_by_component[0, 0] == pytest.approx(0.0625)


def test_apparent_source_photometry_requires_distance_and_extinction_offsets() -> None:
    model = GenulensSourceModel(
        source_data=SourcePhotometry(magnitudes=(MagnitudeMeasurement("Imag", 19.0, 0.03),)),
    )

    with pytest.raises(ValueError, match="offset_provider"):
        model.build_evidence_grid([8_000.0])


def test_jax_exponential_dust_offsets_match_numpy_offsets() -> None:
    dust = ExponentialDustOffsets(
        l_deg=1.0,
        b_deg=-3.9,
        extinction_at_reference={"Imag": 1.2, "Vmag": 2.0},
    )
    coordinates = CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag")
    jax_dust = ExponentialDustModel.from_exponential(dust, coordinates)

    expected = dust(8, 8_000.0)
    actual = np.asarray(jax_dust.offsets(8.0))
    assert actual == pytest.approx([expected["Imag"], expected["Vmag"], expected["Imag"]])


def test_selection_probability_table_can_be_saved_loaded_and_interpolated(tmp_path: Path) -> None:
    table = SourceEvidenceGrid(
        distance_pc=np.asarray([1000.0, 2000.0, 3000.0]),
        evidence_by_component=np.asarray(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [1.0, 0.0],
            ]
        ),
        component_indices=np.asarray([2, 5]),
        metadata={"label": "test"},
    )

    path = tmp_path / "selection.npz"
    table.save_npz(path)
    loaded = SourceEvidenceGrid.load_npz(path)

    assert loaded.metadata == {"label": "test"}
    assert loaded.evidence(2, 2.5) == pytest.approx(0.75)
    assert loaded.evidence(5, 2.5) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("distance_pc", "evidence"),
    [
        (np.asarray([2_000.0, 1_000.0]), np.ones((2, 1))),
        (np.asarray([1_000.0, 2_000.0]), np.asarray([[1.0], [np.nan]])),
        (np.asarray([1_000.0, 2_000.0]), np.asarray([[1.0], [-0.1]])),
    ],
)
def test_source_evidence_grid_rejects_invalid_event_prior_weights(
    distance_pc: np.ndarray,
    evidence: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        SourceEvidenceGrid(distance_pc=distance_pc, evidence_by_component=evidence)


def test_genulens_builder_uses_exact_isochrone_probability_for_magnitude_cuts() -> None:
    points = {
        0: [
            SimpleNamespace(log_age=1.0, metallicity_mh=-0.2, weight=0.25),
            SimpleNamespace(log_age=3.0, metallicity_mh=0.1, weight=0.75),
        ]
    }
    generator = _FakeGenerator()
    table = GenulensSourceEvidenceBuilder(
        generator=generator,
        genulens=_fake_genulens(points),
        source_data=SourceSelection(cuts=(MagnitudeCut("Imag", 19.0, 21.0),)),
        offset_provider=lambda component, distance: {"Imag": 14.5},
    ).build([8_000.0], component_indices=[0])

    assert table.evidence_by_component[0, 0] == pytest.approx(0.45)
    assert table.metadata["method"] == "isochrone_interval"
    assert [query.magnitude_selections[0].magnitude_offset for query in generator.queries] == [14.5, 14.5]


def test_genulens_cmd_builder_creates_a_component_cmd_prior(tmp_path: Path) -> None:
    points = {0: [SimpleNamespace(log_age=1.0, metallicity_mh=-0.2, weight=1.0)]}
    table = GenulensCmdPriorBuilder(
        generator=_FakeGenerator(),
        genulens=_fake_genulens(points),
        samples_per_population_point=10,
    ).build(
        CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=[2.0, 3.5, 5.0],
        color_edges=[0.0, 1.5, 3.0],
        smoothing_sigma_bins=0.0,
        component_indices=[0],
    )

    path = tmp_path / "cmd_prior.npz"
    table.save_npz(path)
    loaded = CmdPriorTable.load_npz(path)

    assert table.density_by_component.shape == (1, 2, 2)
    assert np.sum(table.density_by_component[0] * 2.25) == pytest.approx(1.0)
    assert loaded.coordinates == table.coordinates
    assert np.allclose(loaded.density_by_component, table.density_by_component)


def test_genulens_builder_samples_color_selection_when_needed() -> None:
    points = {0: [SimpleNamespace(log_age=1.0, metallicity_mh=-0.2, weight=1.0)]}
    table = GenulensSourceEvidenceBuilder(
        generator=_FakeGenerator(),
        genulens=_fake_genulens(points),
        source_data=SourceSelection(cuts=(ColorCut("Vmag", "Imag", 0.5, 1.5),)),
        samples_per_population_point=10,
    ).build([8_000.0], component_indices=[0])

    assert table.evidence_by_component[0, 0] == pytest.approx(0.5)
    assert table.metadata["method"] == "forward_source_monte_carlo"


def test_genulens_builder_factory_imports_the_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    points = {0: [SimpleNamespace(log_age=1.0, metallicity_mh=-0.2, weight=1.0)]}
    module = _fake_genulens(points)
    monkeypatch.setitem(sys.modules, "genulens", module)

    builder = GenulensSourceEvidenceBuilder.from_genulens(
        _FakeGenerator(),
        SourceSelection(cuts=(MagnitudeCut("Imag", 19.0, 21.0),)),
    )

    assert builder.genulens is module


def test_selected_source_density_reweights_component_distance_density() -> None:
    distance_pc = np.asarray([1000.0, 2000.0, 3000.0])
    source_density_by_component = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    selection = SourceEvidenceGrid(
        distance_pc=distance_pc,
        evidence_by_component=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )

    selected = ConditionedSourceDensity.from_base_density(
        distance_pc,
        source_density_by_component,
        selection,
    )

    assert selected.source_density.tolist() == [0.0, 1.0, 2.0]
    assert selected.source_pdf(3.0) > selected.source_pdf(2.0)
    assert selected.component_weights(3.0).tolist() == [0.5, 0.5]


def test_histogram_density_can_apply_external_source_selection_table() -> None:
    density = HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    selection = SourceEvidenceGrid(
        distance_pc=density.distance.distance_pc,
        evidence_by_component=np.tile(
            (density.distance.distance_pc >= 500.0).astype(float)[:, None],
            (1, density.distance.source_density_by_component.shape[1]),
        ),
    )

    selected = density.with_source_evidence(selection)

    assert selected.distance.source_pdf(0.2) == 0.0
    assert np.all(selected.distance.source_density_by_component[0] == 0.0)
    assert selected.distance.source_norm > 0.0
    assert np.isfinite(selected.log_density(0.3, 0.26, 0.6, 5.0, 2.0))


def test_histogram_default_source_density_uses_forward_geometric_base() -> None:
    density = HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    assert np.allclose(
        density.distance.source_density_by_component,
        density.distance.base_source_density_by_component,
    )


def test_histogram_cmd_joint_density_evaluates_current_photometry() -> None:
    import jax.numpy as jnp
    from gapmoe.density.histogram_backend import CmdPriorEvaluator

    density = HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    cmd_prior = CmdPriorTable(
        coordinates=CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=np.asarray([0.0, 1.0]),
        color_edges=np.asarray([0.0, 1.0]),
        density_by_component=np.ones((11, 1, 1)),
    )
    values = (0.3, 0.26, 0.6, 5.0, 2.0)
    joint = density.cmd_joint_density(
        *values,
        cmd_prior=CmdPriorEvaluator.from_table(cmd_prior),
        reference_magnitude=0.5,
        color=0.5,
        magnitude_offsets=jnp.zeros(3),
    )

    assert joint == pytest.approx(density.density(*values))


def test_histogram_selection_uses_genulens_forward_source_geometric_factor() -> None:
    density = HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    probability = np.full_like(density.distance.base_source_density_by_component, 0.25)
    selection = SourceEvidenceGrid(
        distance_pc=density.distance.distance_pc,
        evidence_by_component=probability,
    )

    selected = density.with_source_evidence(selection)
    expected = density.distance.lens_density_by_component * 1.0e-6 * density.distance.distance_pc[:, None] ** 2
    assert np.allclose(selected.distance.source_density_by_component, 0.25 * expected)


def test_histogram_density_can_apply_selection_while_loading() -> None:
    base = HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    selection = SourceEvidenceGrid(
        distance_pc=base.distance.distance_pc,
        evidence_by_component=np.tile(
            (base.distance.distance_pc >= 500.0).astype(float)[:, None],
            (1, base.distance.source_density_by_component.shape[1]),
        ),
    )

    loaded = HistogramDensity.from_paths(
        FIXTURE / "mass.dat",
        FIXTURE / "rho.dat",
        FIXTURE / "murel.dat",
        source_evidence=selection,
    )

    assert loaded.distance.source_norm == pytest.approx(base.with_source_evidence(selection).distance.source_norm)


def test_histogram_density_builds_genulens_selection_on_its_native_grid() -> None:
    density = HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")

    class Builder:
        called_with = None

        def build(self, distance_pc, *, component_indices):
            self.called_with = (np.asarray(distance_pc), tuple(component_indices))
            return SourceEvidenceGrid(
                distance_pc=np.asarray(distance_pc),
                evidence_by_component=np.ones((len(distance_pc), len(component_indices))),
            )

    builder = Builder()
    selected = density.with_genulens_source_evidence(builder)

    assert builder.called_with is not None
    assert np.array_equal(builder.called_with[0], density.distance.distance_pc)
    assert builder.called_with[1] == tuple(range(11))
    assert np.allclose(
        selected.distance.source_density_by_component,
        density.distance.base_source_density_by_component,
    )


def test_histogram_loader_accepts_source_selection_table() -> None:
    base = HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    selection = SourceEvidenceGrid(
        distance_pc=base.distance.distance_pc,
        evidence_by_component=np.zeros_like(base.distance.source_density_by_component),
    )
    density = HistogramDensity.from_paths(
        FIXTURE / "mass.dat",
        FIXTURE / "rho.dat",
        FIXTURE / "murel.dat",
        source_evidence=selection,
    )

    assert float(density.distance.source_norm) == 0.0


def test_source_group_murel_histogram_reweights_kinematics(tmp_path: Path) -> None:
    row_thin = [1000.0, 500.0, 5.0, -1.0, 0.2, 0.2, 0.1, 0.9, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0]
    row_thick = [1000.0, 500.0, 15.0, 1.0, 0.2, 0.2, 0.1, 0.9, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0]
    path = tmp_path / "grouped_murel.dat"
    path.write_text(
        "# Grid: DL [0, 1000] step 500 pc (2 bins)\n"
        "# Grid: DS [500, 1500] step 500 pc (2 bins)\n"
        + "\n".join(" ".join(str(value) for value in row) for row in (row_thin, row_thick))
        + "\n"
    )
    murel = MurelHistogram.from_file(path)

    thin_mu, thin_phi = murel.densities(0.5, 1.0, 5.0, -1.0, np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    thick_mu, thick_phi = murel.densities(0.5, 1.0, 5.0, -1.0, np.array([0.0, 1.0, 0.0, 0.0, 0.0]))

    assert murel.has_source_groups
    assert thin_mu == pytest.approx(0.1)
    assert thin_phi == pytest.approx(0.1)
    assert thick_mu == pytest.approx(0.9)
    assert thick_phi == pytest.approx(0.9)

    from gapmoe.density.histogram_backend import MurelHistogram as BackendMurelHistogram

    jax_murel = BackendMurelHistogram.from_tables(SimpleNamespace(murel=murel))
    jax_mu, jax_phi = jax_murel.densities(0.5, 1.0, 5.0, -1.0, np.array([0.0, 1.0, 0.0, 0.0, 0.0]))
    assert float(jax_mu) == pytest.approx(0.9)
    assert float(jax_phi) == pytest.approx(0.9)
