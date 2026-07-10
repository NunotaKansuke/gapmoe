"""gapmoe public API."""

__all__ = [
    "Model",
    "GalacticModel",
    "JaxGalacticModel",
    "MappedGalacticModel",
    "ParamType",
    "from_model_spec",
    "PreRunner",
    "PreRunResult",
    "GenulensEnvironment",
    "HistogramDensity",
    "SourcePopulation",
    "AgeMetallicityPoint",
    "CmdCoordinates",
    "CmdPriorTable",
    "GenulensSourceModel",
    "SourceSelection",
    "SourceEvidenceGrid",
    "ExponentialDustModel",
    "ExponentialDustOffsets",
    "CmdGalacticModel",
    "EventPrior5D",
    "SourceCmdPrior",
]


def __getattr__(name):
    if name == "Model":
        from .priors.high_level import Model
        return Model
    if name == "GalacticModel":
        from .priors.galactic import GalacticModel

        return GalacticModel
    if name in {"JaxGalacticModel", "MappedGalacticModel"}:
        from .priors.mapped import MappedGalacticModel

        return MappedGalacticModel
    if name in {
        "BinaryCircularParamType",
        "BinaryCircularUseThEParamType",
        "BinaryKeplerParamType",
        "SingleLensParamType",
        "SingleLensUseThEParamType",
        "ParamType",
        "from_model_spec",
    }:
        from . import param_types

        return getattr(param_types, name)
    if name in {"PreRunner", "PreRunResult", "GenulensEnvironment"}:
        from .pre_runner import GenulensEnvironment, PreRunner, PreRunResult

        return {"PreRunner": PreRunner, "PreRunResult": PreRunResult, "GenulensEnvironment": GenulensEnvironment}[name]
    if name == "HistogramDensity":
        from .density import HistogramDensity

        return HistogramDensity
    if name in {
        "CmdCoordinates",
        "CmdPriorTable",
        "GenulensSourceModel",
        "SourceSelection",
        "SourceEvidenceGrid",
        "ExponentialDustModel",
        "ExponentialDustOffsets",
        "CmdGalacticModel",
        "EventPrior5D",
        "SourceCmdPrior",
    }:
        from .priors import CmdGalacticModel, EventPrior5D, SourceCmdPrior
        from .source_selection import (
            CmdCoordinates,
            CmdPriorTable,
            ExponentialDustModel,
            ExponentialDustOffsets,
            GenulensSourceModel,
            SourceEvidenceGrid,
            SourceSelection,
        )

        return {
            "CmdCoordinates": CmdCoordinates,
            "CmdPriorTable": CmdPriorTable,
            "GenulensSourceModel": GenulensSourceModel,
            "SourceSelection": SourceSelection,
            "SourceEvidenceGrid": SourceEvidenceGrid,
            "ExponentialDustModel": ExponentialDustModel,
            "ExponentialDustOffsets": ExponentialDustOffsets,
            "CmdGalacticModel": CmdGalacticModel,
            "EventPrior5D": EventPrior5D,
            "SourceCmdPrior": SourceCmdPrior,
        }[name]
    if name in {"SourcePopulation", "AgeMetallicityPoint"}:
        from .source_selection import AgeMetallicityPoint, SourcePopulation

        return {"SourcePopulation": SourcePopulation, "AgeMetallicityPoint": AgeMetallicityPoint}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
