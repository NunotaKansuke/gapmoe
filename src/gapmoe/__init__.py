__all__ = [
    "GalacticModel",
    "PreRunner",
    "PreRunResult",
    "SourceSelection",
    "PhysicalParams",
    "HistogramDensity",
    "GalacticPrior",
    "gapmoe",
]


def __getattr__(name):
    if name in {"GalacticModel", "gapmoe"}:
        from .model import GalacticModel, gapmoe

        exports = {"GalacticModel": GalacticModel, "gapmoe": gapmoe}
        return exports[name]
    if name in {"PreRunner", "PreRunResult", "SourceSelection"}:
        from .pre_runner import PreRunner, PreRunResult, SourceSelection

        exports = {
            "PreRunner": PreRunner,
            "PreRunResult": PreRunResult,
            "SourceSelection": SourceSelection,
        }
        return exports[name]
    if name == "PhysicalParams":
        from .physical import PhysicalParams

        return PhysicalParams
    if name == "HistogramDensity":
        from .density import HistogramDensity

        return HistogramDensity
    if name == "GalacticPrior":
        from .priors import GalacticPrior

        return GalacticPrior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
