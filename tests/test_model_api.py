from __future__ import annotations

import json

import gapmoe
import pytest


def _histogram_build_kwargs(**overrides):
    values = {
        "source": object(),
        "l": 1.0,
        "b": -3.9,
        "extinction": {},
        "dm_rc": None,
        "dust_scale_height_pc": 164.0,
        "include_event_rate": True,
        "remnant": 0,
        "binary": 0,
    }
    values.update(overrides)
    return values


def _write_histogram_manifest(tmp_path, *, commands=None):
    for name in ("mass", "rho", "murel"):
        (tmp_path / f"{name}.dat").touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "l_deg": 1.0,
                "b_deg": -3.9,
                "mass_path": "mass.dat",
                "rho_path": "rho.dat",
                "murel_path": "murel.dat",
                "commands": {} if commands is None else commands,
            }
        )
    )


def test_histogram_backend_opens_relative_manifest_paths(tmp_path):
    _write_histogram_manifest(tmp_path)

    backend = gapmoe.Histogram.open(tmp_path)

    assert backend.pre_run.mass_path == tmp_path / "mass.dat"
    assert backend.pre_run.rho_path == tmp_path / "rho.dat"
    assert backend.pre_run.murel_path == tmp_path / "murel.dat"


def test_histogram_backend_rejects_unverifiable_nondefault_model_options(tmp_path):
    _write_histogram_manifest(tmp_path)
    backend = gapmoe.Histogram.open(tmp_path)

    with pytest.raises(ValueError, match="does not record remnant"):
        backend.build(**_histogram_build_kwargs(remnant=1))


def test_histogram_backend_requires_options_to_match_preparation(tmp_path, monkeypatch):
    _write_histogram_manifest(
        tmp_path,
        commands={"mass": ["calc_mass_dist", "REMNANT", "1", "BINARY", "0"]},
    )
    backend = gapmoe.Histogram.open(tmp_path)

    with pytest.raises(ValueError, match="prepared with remnant=1"):
        backend.build(**_histogram_build_kwargs(remnant=0))

    monkeypatch.setattr(
        "gapmoe.model.GalaxyModel.from_pre_run",
        lambda *args, **kwargs: "physical-model",
    )
    assert backend.build(**_histogram_build_kwargs(remnant=1)) == "physical-model"
