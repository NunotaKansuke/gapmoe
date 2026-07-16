"""Metadata for bundled, trained Galactic flow releases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowRelease:
    """A published flow release and its validated sky coverage."""

    name: str
    l_range_deg: tuple[float, float]
    b_range_deg: tuple[float, float]
    remnant: int = 0
    binary: int = 0
    event_rate_included: bool = False

    def contains(self, l_deg: float, b_deg: float) -> bool:
        return self.l_range_deg[0] <= l_deg <= self.l_range_deg[1] and self.b_range_deg[0] <= b_deg <= self.b_range_deg[1]

    def validate_sightline(self, l_deg: float, b_deg: float) -> None:
        if not self.contains(l_deg, b_deg):
            raise ValueError(
                f"flow release {self.name!r} covers "
                f"{self.l_range_deg[0]} <= l <= {self.l_range_deg[1]} and "
                f"{self.b_range_deg[0]} <= b <= {self.b_range_deg[1]} deg; "
                f"received (l, b)=({l_deg}, {b_deg})"
            )

    def validate_model_options(self, *, remnant: int, binary: int) -> None:
        if (remnant, binary) != (self.remnant, self.binary):
            raise ValueError(
                f"flow release {self.name!r} requires REMNANT={self.remnant}, BINARY={self.binary}; "
                f"received REMNANT={remnant}, BINARY={binary}"
            )


_RELEASES = {
    "default": FlowRelease(name="default", l_range_deg=(-5.0, 5.0), b_range_deg=(-6.0, -2.0)),
    "rate-included-v1": FlowRelease(
        name="rate-included-v1",
        l_range_deg=(-5.0, 5.0),
        b_range_deg=(-6.0, -2.0),
        event_rate_included=True,
    ),
}


def get_flow_release(name: str = "default") -> FlowRelease:
    try:
        return _RELEASES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_RELEASES))
        raise ValueError(f"unknown flow release {name!r}; available releases: {available}") from exc
