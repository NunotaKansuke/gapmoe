from __future__ import annotations

from abc import ABC, abstractmethod

from gapmoe.physical import PhysicalParams


class DensityModel(ABC):
    """Interface for Galactic density backends."""

    @abstractmethod
    def log_density(self, params: PhysicalParams) -> float:
        """Return log density for canonical physical parameters."""
