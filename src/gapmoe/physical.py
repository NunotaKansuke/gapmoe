from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt


@dataclass(frozen=True)
class PhysicalParams:
    """Canonical physical parameters for GAPMOE density evaluation.

    Distances are in pc. Proper motions are heliocentric and in mas/yr.
    """

    ML: float
    DL: float
    DS: float
    mu_N: float
    mu_E: float

    @property
    def mu(self) -> float:
        return sqrt(self.mu_N * self.mu_N + self.mu_E * self.mu_E)

    @property
    def phi(self) -> float:
        return atan2(self.mu_E, self.mu_N)
