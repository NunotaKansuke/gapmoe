from __future__ import annotations

import hashlib
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
    assert package.manifest.event_rate_included is False


def test_flow_package_reads_rate_included_marker(tmp_path):
    root = tmp_path / "rate-included-v1"
    root.mkdir()
    (root / "event_kernel").mkdir()
    (root / "source_distance_grid.npz").touch()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": FLOW_PACKAGE_SCHEMA,
                "release": "rate-included-v1",
                "coverage_deg": {"l": [-5.0, 5.0], "b": [-6.0, -2.0]},
                "model_options": {"remnant": 0, "binary": 0},
                "event_rate_included": True,
                "event_kernel": "event_kernel",
                "source_distance_grid": "source_distance_grid.npz",
                "cmd_prior": None,
            }
        )
    )

    package = FlowPackage.open(root)

    assert package.manifest.event_rate_included is True


def test_bundled_rate_included_artifact_hashes_match_manifest():
    package = FlowPackage.bundled("rate-included-v1")
    payload = json.loads((package.root / "manifest.json").read_text())
    provenance = payload["provenance"]
    paths = {
        "main_event_kernel_config_sha256": package.event_kernel_path / "config.json",
        "main_event_kernel_flow_sha256": package.event_kernel_path / "flow.eqx",
        "rare_group_event_kernel_config_sha256": package.event_kernel_path / "rare_groups/config.json",
        "rare_group_event_kernel_flow_sha256": package.event_kernel_path / "rare_groups/flow.eqx",
        "group_overrides_sha256": package.event_kernel_path / "group_overrides.json",
        "source_distance_grid_sha256": package.source_distance_grid_path,
    }

    for name, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance[name]


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
