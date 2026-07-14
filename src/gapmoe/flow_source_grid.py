"""Source-distance grid used by a trained event-kernel flow release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FlowSourceDistance:
    """One sightline's component-resolved source-distance measure."""

    distance_pc: jnp.ndarray
    source_by_component: jnp.ndarray
    source_norm: float

    def source_component_values(self, ds_kpc):
        values = jnp.stack(
            [
                jnp.interp(ds_kpc * 1000.0, self.distance_pc, self.source_by_component[:, component], left=0.0, right=0.0)
                for component in range(self.source_by_component.shape[1])
            ]
        )
        return values


@dataclass(frozen=True)
class FlowSourceDensity:
    """Minimal density facade consumed by ``SourceCmdPrior``."""

    distance: FlowSourceDistance


@dataclass(frozen=True)
class FlowSourceDistanceGrid:
    """Bilinearly interpolated source-distance measures over a sightline grid.

    ``source_by_component`` has shape ``(b, l, distance, component)`` and is
    evaluated before CMD selection. A published Flow release also includes
    the DS-dependent lens-column normalization needed by its base kernel.
    """

    l_deg: jnp.ndarray
    b_deg: jnp.ndarray
    distance_pc: jnp.ndarray
    source_by_component: jnp.ndarray

    @classmethod
    def from_rho_profiles(
        cls,
        *,
        l_deg: np.ndarray,
        b_deg: np.ndarray,
        distance_pc: np.ndarray,
        nms_by_sightline: np.ndarray,
    ) -> "FlowSourceDistanceGrid":
        """Build the raw source grid from ``calc_rho_profile SOURCE=0`` output.

        ``nms_by_sightline`` has shape ``(b, l, distance, component)`` and
        stores the eleven ``nMS`` columns from rho profiles.
        """

        distance_pc = np.asarray(distance_pc, dtype=float)
        nms = np.asarray(nms_by_sightline, dtype=float)
        expected = (len(b_deg), len(l_deg), len(distance_pc), 11)
        if nms.shape != expected:
            raise ValueError(f"nms_by_sightline must have shape {expected}, got {nms.shape}")
        source = nms * 1.0e-6 * distance_pc[None, None, :, None] ** 2
        return cls(
            l_deg=jnp.asarray(l_deg, dtype=float),
            b_deg=jnp.asarray(b_deg, dtype=float),
            distance_pc=jnp.asarray(distance_pc, dtype=float),
            source_by_component=jnp.asarray(source, dtype=float),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "FlowSourceDistanceGrid":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                l_deg=jnp.asarray(data["l_deg"], dtype=float),
                b_deg=jnp.asarray(data["b_deg"], dtype=float),
                distance_pc=jnp.asarray(data["distance_pc"], dtype=float),
                source_by_component=jnp.asarray(data["source_by_component"], dtype=float),
            )

    def save_npz(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            l_deg=np.asarray(self.l_deg),
            b_deg=np.asarray(self.b_deg),
            distance_pc=np.asarray(self.distance_pc),
            source_by_component=np.asarray(self.source_by_component),
        )

    def at(self, l_deg: float, b_deg: float) -> FlowSourceDensity:
        values = self._interpolate_sightline(l_deg, b_deg)
        source_norm = float(jnp.trapezoid(jnp.sum(values, axis=1), self.distance_pc / 1000.0))
        return FlowSourceDensity(
            distance=FlowSourceDistance(
                distance_pc=self.distance_pc,
                source_by_component=values,
                source_norm=source_norm,
            )
        )

    def _interpolate_sightline(self, l_deg: float, b_deg: float):
        if not (float(self.l_deg[0]) <= l_deg <= float(self.l_deg[-1])):
            raise ValueError("l is outside the source-distance grid")
        if not (float(self.b_deg[0]) <= b_deg <= float(self.b_deg[-1])):
            raise ValueError("b is outside the source-distance grid")
        il1 = int(np.searchsorted(np.asarray(self.l_deg), l_deg, side="right"))
        ib1 = int(np.searchsorted(np.asarray(self.b_deg), b_deg, side="right"))
        il0, il1 = max(0, il1 - 1), min(len(self.l_deg) - 1, il1)
        ib0, ib1 = max(0, ib1 - 1), min(len(self.b_deg) - 1, ib1)
        tl = 0.0 if il0 == il1 else (l_deg - float(self.l_deg[il0])) / float(self.l_deg[il1] - self.l_deg[il0])
        tb = 0.0 if ib0 == ib1 else (b_deg - float(self.b_deg[ib0])) / float(self.b_deg[ib1] - self.b_deg[ib0])
        return (
            (1.0 - tb) * ((1.0 - tl) * self.source_by_component[ib0, il0] + tl * self.source_by_component[ib0, il1])
            + tb * ((1.0 - tl) * self.source_by_component[ib1, il0] + tl * self.source_by_component[ib1, il1])
        )
