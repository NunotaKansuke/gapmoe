__all__ = [
    "GalacticModel",
    "GenulensEnvironment",
    "PreRunner",
    "PreRunResult",
    "SourceSelection",
    "HistogramDensity",
    "JaxHistogramDensity",
    "GalacticPrior",
    "JaxGalacticPrior",
    "BinaryCircularParameterization",
    "BinaryCircularUseThEParameterization",
    "BinaryKeplerParameterization",
    "SingleLensParameterization",
    "SingleLensUseThEParameterization",
    "gapmoe",
]

_PARAMETERIZATIONS = {
    "BinaryCircularParameterization",
    "BinaryCircularUseThEParameterization",
    "BinaryKeplerParameterization",
    "SingleLensParameterization",
    "SingleLensUseThEParameterization",
}


def __getattr__(name):
    if name in {"GalacticModel", "gapmoe"}:
        from .model import GalacticModel, gapmoe

        exports = {"GalacticModel": GalacticModel, "gapmoe": gapmoe}
        return exports[name]
    if name in {"GenulensEnvironment", "PreRunner", "PreRunResult", "SourceSelection"}:
        from .pre_runner import GenulensEnvironment, PreRunner, PreRunResult, SourceSelection

        exports = {
            "GenulensEnvironment": GenulensEnvironment,
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
    if name in _PARAMETERIZATIONS:
        from . import parameterizations as _pm

        return getattr(_pm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
