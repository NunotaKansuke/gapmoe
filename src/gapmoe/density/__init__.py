from .base import DensityModel
from .histogram_numpy import HistogramDensity

__all__ = ["DensityModel", "HistogramDensity", "JaxHistogramDensity"]


def __getattr__(name):
    if name == "JaxHistogramDensity":
        from .histogram_jax import JaxHistogramDensity

        return JaxHistogramDensity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
