"""gapmoe public API."""

__all__ = [
<<<<<<< HEAD
=======
    "Flow",
>>>>>>> codex/inference-mode-cleanup
    "Histogram",
    "Isochrone",
    "Model",
    "ParamType",
    "SourcePopulation",
    "AgeMetallicityPoint",
    "calc_vEarth",
]


def __getattr__(name):
<<<<<<< HEAD
    if name in {"Histogram", "Isochrone", "Model"}:
        from .model import Histogram, Isochrone, Model
        return {
=======
    if name in {"Flow", "Histogram", "Isochrone", "Model"}:
        from .model import Flow, Histogram, Isochrone, Model
        return {
            "Flow": Flow,
>>>>>>> codex/inference-mode-cleanup
            "Histogram": Histogram,
            "Isochrone": Isochrone,
            "Model": Model,
        }[name]
    if name == "ParamType":
        from .param_types import ParamType
        return ParamType
    if name == "calc_vEarth":
        from .param_types import calc_vEarth
        return calc_vEarth
    if name in {"SourcePopulation", "AgeMetallicityPoint"}:
        from .source_selection import AgeMetallicityPoint, SourcePopulation
        return {"SourcePopulation": SourcePopulation, "AgeMetallicityPoint": AgeMetallicityPoint}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
