from __future__ import annotations

from abc import ABC, abstractmethod

class DensityModel(ABC):
    """Interface for Galactic density backends."""

    @abstractmethod
    def log_density(self, ML: float, DL: float, DS: float, mu_N: float, mu_E: float) -> float:
        """Return log density for ML, DL, DS, mu_N, mu_E."""
