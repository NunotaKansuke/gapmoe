from __future__ import annotations

import json

import pytest

from gapmoe.flow_package import FLOW_PACKAGE_SCHEMA, FlowPackage


def test_flow_package_validates_its_required_artifacts(tmp_path):
    root = tmp_path / "default"
    root.mkdir()
    (root / "event_kernel").mkdir()
    (root / "source_distance_grid.npz").touch()
    (root / "cmd_prior.npz").touch()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": FLOW_PACKAGE_SCHEMA,
                "release": "default",
                "coverage_deg": {"l": [-5.0, 5.0], "b": [-6.0, -2.0]},
                "model_options": {"remnant": 0, "binary": 0},
                "event_kernel": "event_kernel",
                "source_distance_grid": "source_distance_grid.npz",
                "cmd_prior": "cmd_prior.npz",
                "validation": {"max_ks": 0.03},
            }
        )
    )

    package = FlowPackage.open(root)

    assert package.manifest.release == "default"
    assert package.manifest.l_range_deg == (-5.0, 5.0)


def test_flow_package_rejects_missing_artifacts(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": FLOW_PACKAGE_SCHEMA,
                "release": "default",
                "coverage_deg": {"l": [-5.0, 5.0], "b": [-6.0, -2.0]},
                "model_options": {"remnant": 0, "binary": 0},
                "event_kernel": "event_kernel",
                "source_distance_grid": "source_distance_grid.npz",
                "cmd_prior": "cmd_prior.npz",
            }
        )
    )

    with pytest.raises(FileNotFoundError, match="incomplete"):
        FlowPackage.open(tmp_path)
