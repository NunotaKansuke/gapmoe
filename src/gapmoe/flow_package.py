"""On-disk representation of a bundled trained flow release."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


FLOW_PACKAGE_SCHEMA = "gapmoe-flow-v1"


@dataclass(frozen=True)
class FlowPackageManifest:
    release: str
    l_range_deg: tuple[float, float]
    b_range_deg: tuple[float, float]
    remnant: int
    binary: int
    event_kernel: str
    source_distance_grid: str
    cmd_prior: str | None
    validation: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "FlowPackageManifest":
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != FLOW_PACKAGE_SCHEMA:
            raise ValueError(f"unsupported flow package schema: {payload.get('schema')!r}")
        coverage = payload["coverage_deg"]
        options = payload["model_options"]
        return cls(
            release=str(payload["release"]),
            l_range_deg=tuple(float(value) for value in coverage["l"]),
            b_range_deg=tuple(float(value) for value in coverage["b"]),
            remnant=int(options["remnant"]),
            binary=int(options["binary"]),
            event_kernel=str(payload["event_kernel"]),
            source_distance_grid=str(payload["source_distance_grid"]),
            cmd_prior=str(payload["cmd_prior"]) if payload.get("cmd_prior") else None,
            validation=dict(payload.get("validation", {})),
        )


@dataclass(frozen=True)
class FlowPackage:
    root: Path
    manifest: FlowPackageManifest

    @classmethod
    def open(cls, root: str | Path) -> "FlowPackage":
        root = Path(root)
        manifest = FlowPackageManifest.from_json(root / "manifest.json")
        required = [root / manifest.event_kernel, root / manifest.source_distance_grid]
        if manifest.cmd_prior is not None:
            required.append(root / manifest.cmd_prior)
        missing = [str(path.name) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete flow package {root}: missing {', '.join(missing)}")
        return cls(root=root, manifest=manifest)

    @property
    def event_kernel_path(self) -> Path:
        return self.root / self.manifest.event_kernel

    @property
    def source_distance_grid_path(self) -> Path:
        return self.root / self.manifest.source_distance_grid

    @classmethod
    def bundled(cls, release: str = "default") -> "FlowPackage":
        root = files("gapmoe.data").joinpath("flows", release)
        try:
            return cls.open(Path(root))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"bundled flow release {release!r} is not installed; install a gapmoe release containing flow data"
            ) from exc
