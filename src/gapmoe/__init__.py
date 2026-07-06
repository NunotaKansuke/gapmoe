__all__ = [
    "GalacticModel",
    "GenulensEnvironment",
    "PreRunner",
    "PreRunResult",
    "SourceSelection",
    "HistogramDensity",
    "JaxHistogramDensity",
    "JaxGalacticModel",
    "BinaryCircularParamType",
    "BinaryCircularUseThEParamType",
    "BinaryKeplerParamType",
    "SingleLensParamType",
    "SingleLensUseThEParamType",
    "ParamType",
    "from_model_spec",
]

_PARAM_TYPES = {
    "BinaryCircularParamType",
    "BinaryCircularUseThEParamType",
    "BinaryKeplerParamType",
    "SingleLensParamType",
    "SingleLensUseThEParamType",
    "ParamType",
    "from_model_spec",
}


def __getattr__(name):
    if name == "GalacticModel":
        from .priors import GalacticModel

        return GalacticModel
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
    if name == "JaxGalacticModel":
        from .priors import JaxGalacticModel

        return JaxGalacticModel
    if name in _PARAM_TYPES:
        from . import param_types as _pm

        return getattr(_pm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
