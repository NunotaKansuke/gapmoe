__all__ = [
    "GalacticModel",
    "CmdGalacticModel",
    "EventPrior5D",
    "SourceCmdPrior",
    "GenulensEnvironment",
    "PreRunner",
    "PreRunResult",
    "SourceSelection",
    "GenulensSourceModel",
    "ExponentialDustOffsets",
    "ExponentialDustModel",
    "SourcePhotometry",
    "MagnitudeMeasurement",
    "ColorMeasurement",
    "SourceEvidenceGrid",
    "CmdCoordinates",
    "CmdSmoothing",
    "CmdPriorTable",
    "HistogramDensity",
    "MappedGalacticModel",
    "BinaryCircularParamType",
    "BinaryCircularUseThEParamType",
    "BinaryKeplerParamType",
    "SingleLensParamType",
    "SingleLensUseThEParamType",
    "ParamType",
    "from_model_spec",
    "source_selection",
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
    if name == "source_selection":
        import importlib

        return importlib.import_module(".source_selection", __name__)
    if name == "GalacticModel":
        from .priors import GalacticModel

        return GalacticModel
    if name == "CmdGalacticModel":
        from .priors import CmdGalacticModel

        return CmdGalacticModel
    if name in {"EventPrior5D", "SourceCmdPrior"}:
        from .priors import EventPrior5D, SourceCmdPrior

        return {"EventPrior5D": EventPrior5D, "SourceCmdPrior": SourceCmdPrior}[name]
    if name in {"GenulensEnvironment", "PreRunner", "PreRunResult"}:
        from .pre_runner import GenulensEnvironment, PreRunner, PreRunResult

        exports = {
            "GenulensEnvironment": GenulensEnvironment,
            "PreRunner": PreRunner,
            "PreRunResult": PreRunResult,
        }
        return exports[name]
    if name in {
        "SourceSelection",
        "GenulensSourceModel",
        "ExponentialDustOffsets",
        "ExponentialDustModel",
        "SourcePhotometry",
        "MagnitudeMeasurement",
        "ColorMeasurement",
        "SourceEvidenceGrid",
        "CmdCoordinates",
        "CmdSmoothing",
        "CmdPriorTable",
    }:
        from .source_selection import (
            ColorMeasurement,
            ExponentialDustOffsets,
            ExponentialDustModel,
            GenulensSourceModel,
            MagnitudeMeasurement,
            SourcePhotometry,
            SourceEvidenceGrid,
            SourceSelection,
            CmdCoordinates,
            CmdSmoothing,
            CmdPriorTable,
        )

        return {
            "SourceSelection": SourceSelection,
            "GenulensSourceModel": GenulensSourceModel,
            "ExponentialDustOffsets": ExponentialDustOffsets,
            "ExponentialDustModel": ExponentialDustModel,
            "SourcePhotometry": SourcePhotometry,
            "MagnitudeMeasurement": MagnitudeMeasurement,
            "ColorMeasurement": ColorMeasurement,
            "SourceEvidenceGrid": SourceEvidenceGrid,
            "CmdCoordinates": CmdCoordinates,
            "CmdSmoothing": CmdSmoothing,
            "CmdPriorTable": CmdPriorTable,
        }[name]
    if name == "HistogramDensity":
        from .density import HistogramDensity

        return HistogramDensity
    if name == "MappedGalacticModel":
        from .priors import MappedGalacticModel

        return MappedGalacticModel
    if name in _PARAM_TYPES:
        from . import param_types as _pm

        return getattr(_pm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
