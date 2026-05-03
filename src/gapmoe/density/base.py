from __future__ import annotations

from abc import ABC, abstractmethod

class DensityModel(ABC):
    """Interface for Galactic density backends."""

    @abstractmethod
    def log_density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> float:
        """Return log density for ML [Msun], DL [kpc], DS [kpc], mu_N [mas/yr], mu_E [mas/yr]."""
