__all__ = [
    "GalacticModel",
    "PreRunner",
    "PreRunResult",
    "SourceSelection",
    "HistogramDensity",
    "JaxHistogramDensity",
    "GalacticPrior",
    "JaxGalacticPrior",
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
    if name == "HistogramDensity":
        from .density import HistogramDensity

        return HistogramDensity
    if name == "JaxHistogramDensity":
        from .density import JaxHistogramDensity

        return JaxHistogramDensity
    if name == "GalacticPrior":
        from .priors import GalacticPrior

        return GalacticPrior
    if name == "JaxGalacticPrior":
        from .priors import JaxGalacticPrior

        return JaxGalacticPrior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
