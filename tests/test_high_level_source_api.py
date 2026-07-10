from __future__ import annotations

import json

import numpy as np
import pytest

from gapmoe import AgeMetallicityPoint, Model, SourcePopulation
from gapmoe.pre_runner import PreRunResult
from gapmoe.priors.high_level import IsochroneModel
from gapmoe.source_selection import CmdCoordinates, CmdPriorTable, GenulensSourceModel


def test_model_set_requires_known_settings_and_only_invalidates_for_sightline_changes(tmp_path):
    model = Model(genulens_root="../genulens")
    model._prepared = object()

    assert model.set(l=1.0, b=-3.9, ai_rc=1.2, evi_rc=0.7) is model
    assert model._prepared is None
    assert model._settings["l"] == 1.0
    model._prepared = object()
    model.set(dm_rc=14.4)
    assert model._prepared is not None
    with pytest.raises(TypeError, match="unknown"):
        model.set(typo=1)
def test_isochrone_exposes_single_cmd_chart_and_optional_selection_without_building():
    chart = IsochroneModel(
        reference_band="I",
        color_bands=("V", "I"),
        magnitude_range=(15.0, 21.0),
        color_range=(0.5, 3.0),
    )

    assert chart.coordinates.reference_band == "I"
    assert chart.coordinates.blue_band == "V"
    assert len(chart.selection.cuts) == 2


def test_isochrone_build_selects_internal_table_from_requested_bands(monkeypatch):
    captured = {}

    def fake_build(self, coordinates, **kwargs):
        captured["bands"] = self.bands
        return "table"

    monkeypatch.setattr(GenulensSourceModel, "build_cmd_prior", fake_build)
    chart = IsochroneModel(reference_band="Imag", color_bands=("Vmag", "Imag"))

    built = chart.build(reference_edges=[0.0, 1.0], color_edges=[0.0, 1.0])

    assert built.table == "table"
    assert captured["bands"] == ("Imag", "Vmag")


def test_isochrone_forwards_optional_source_population(monkeypatch):
    captured = {}

    def fake_build(self, coordinates, **kwargs):
        captured["population"] = self.population
        return "table"

    monkeypatch.setattr(GenulensSourceModel, "build_cmd_prior", fake_build)
    population = SourcePopulation(
        imf={"alpha2": -1.3},
        age_metallicity_by_component={8: (AgeMetallicityPoint(9.9, 0.1),)},
    )
    chart = IsochroneModel(reference_band="Imag", color_bands=("Vmag", "Imag"), population=population)

    chart.build(reference_edges=[0.0, 1.0], color_edges=[0.0, 1.0])

    assert captured["population"] == population


def test_isochrone_converts_named_magnitudes_to_its_internal_coordinates():
    chart = IsochroneModel(reference_band="Imag", color_bands=("Vmag", "Imag"))

    assert chart.values_from_magnitudes({"Imag": 19.0, "Vmag": 21.0}) == (19.0, 2.0)
    with pytest.raises(ValueError, match="Vmag"):
        chart.values_from_magnitudes({"Imag": 19.0})


def test_prepare_requires_a_sightline(tmp_path):
    model = Model(genulens_root="../genulens")

    with pytest.raises(ValueError, match="set l and b"):
        model.prepare(tmp_path / "event")


def test_prepare_uses_the_requested_directory_and_persists_settings(tmp_path, monkeypatch):
    directory = tmp_path / "event-001"
    model = Model(genulens_root="../genulens")
    model.set(l=1.0, b=-3.9, extinction={"Imag": 1.2})

    def fake_run(**kwargs):
        assert kwargs["run_name"] == "event-001"
        assert kwargs["l"] == 1.0
        assert kwargs["b"] == -3.9
        directory.mkdir()
        paths = {name: directory / f"{name}.dat" for name in ("mass", "rho", "murel")}
        for path in paths.values():
            path.touch()
        manifest = directory / "manifest.json"
        manifest.write_text("{}")
        return PreRunResult(
            ra_deg=None,
            dec_deg=None,
            l_deg=1.0,
            b_deg=-3.9,
            output_dir=directory,
            mass_path=paths["mass"],
            rho_path=paths["rho"],
            murel_path=paths["murel"],
            manifest_path=manifest,
        )

    monkeypatch.setattr(model._runner, "run", fake_run)
    model.prepare(directory, n_simu=100)

    metadata = (directory / "gapmoe.json").read_text()
    assert '"l_deg": 1.0' in metadata
    assert '"b_deg": -3.9' in metadata
    assert '"n_simu": 100' in metadata


def test_resume_reopens_an_existing_directory_from_metadata(tmp_path):
    directory = tmp_path / "event-001"
    directory.mkdir()
    paths = {name: directory / f"{name}.dat" for name in ("mass", "rho", "murel")}
    for path in paths.values():
        path.touch()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "l_deg": 1.0,
                "b_deg": -3.9,
                **{f"{name}_path": str(path) for name, path in paths.items()},
            }
        )
    )
    (directory / "gapmoe.json").write_text(
        json.dumps({"settings": {"l": 1.0, "b": -3.9, "dm_rc": 14.4}, "prepare_options": {}})
    )

    model = Model(genulens_root="../genulens").resume(directory)

    assert model._prepared is not None
    assert model._settings["l"] == 1.0
    assert model._settings["dm_rc"] == 14.4


def test_resume_requires_complete_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="no complete"):
        Model(genulens_root="../genulens").resume(tmp_path / "missing")


def test_prepare_rejects_a_cached_directory_at_a_different_sightline(tmp_path):
    directory = tmp_path / "event-001"
    directory.mkdir()
    paths = {name: directory / f"{name}.dat" for name in ("mass", "rho", "murel")}
    for path in paths.values():
        path.touch()
    (directory / "manifest.json").write_text(
        json.dumps(
            {"l_deg": 1.0, "b_deg": -3.9, **{f"{name}_path": str(path) for name, path in paths.items()}}
        )
    )
    (directory / "gapmoe.json").write_text(json.dumps({"settings": {"l": 1.0, "b": -3.9}}))

    model = Model(genulens_root="../genulens").set(l=2.0, b=-3.9)
    with pytest.raises(ValueError, match="sightline disagrees"):
        model.prepare(directory)


def test_set_flow_checks_the_release_sightline_coverage():
    model = Model(genulens_root="../genulens").set(l=1.0, b=-3.9)

    assert model.set_flow() is model
    with pytest.raises(ValueError, match="covers"):
        model.set(l=5.0)


def test_set_flow_requires_sightline_and_known_release():
    with pytest.raises(ValueError, match="set l and b"):
        Model(genulens_root="../genulens").set_flow()
    with pytest.raises(ValueError, match="unknown flow release"):
        Model(genulens_root="../genulens").set(l=1.0, b=-3.9).set_flow(release="missing")


def test_remnant_and_binary_are_forwarded_and_must_match_a_flow_release(tmp_path, monkeypatch):
    model = Model(genulens_root="../genulens").set(l=1.0, b=-3.9, remnant=1, binary=1)
    with pytest.raises(ValueError, match="requires REMNANT=0"):
        model.set_flow()

    model = Model(genulens_root="../genulens").set(l=1.0, b=-3.9, remnant=0, binary=0)
    captured = {}
    monkeypatch.setattr(model._runner, "run", lambda **kwargs: captured.update(kwargs))
    model.prepare(tmp_path / "event")
    assert captured["remnant"] == 0
    assert captured["binary"] == 0


def test_isochrone_reuses_the_cmd_table_saved_in_a_resumed_directory(tmp_path):
    directory = tmp_path / "event-001"
    directory.mkdir()
    paths = {name: directory / f"{name}.dat" for name in ("mass", "rho", "murel")}
    for path in paths.values():
        path.touch()
    (directory / "manifest.json").write_text(
        json.dumps(
            {"l_deg": 1.0, "b_deg": -3.9, **{f"{name}_path": str(path) for name, path in paths.items()}}
        )
    )
    (directory / "gapmoe.json").write_text(json.dumps({"settings": {"l": 1.0, "b": -3.9}}))
    reference_edges = np.linspace(-8.0, 20.0, 561)
    color_edges = np.linspace(-2.0, 8.0, 201)
    table = CmdPriorTable(
        coordinates=CmdCoordinates(reference_band="I", blue_band="V", red_band="I"),
        reference_edges=reference_edges,
        color_edges=color_edges,
        density_by_component=np.full((11, 560, 200), 1.0 / (28.0 * 10.0)),
    )
    table.save_npz(directory / "cmd_prior.npz")
    chart = IsochroneModel(reference_band="I", color_bands=("V", "I"))
    (directory / "isochrone.json").write_text(
        json.dumps(Model._isochrone_metadata(chart, 0.75), sort_keys=True)
    )

    restored = Model(genulens_root="../genulens").resume(directory).isochrone(
        reference_band="I", color_bands=("V", "I")
    )

    assert restored.table is not None
    assert np.array_equal(restored.table.reference_edges, reference_edges)
