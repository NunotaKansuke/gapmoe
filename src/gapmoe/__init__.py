__all__ = [
    "gapmoe",
    "PreRunner",
    "PreRunResult",
    "SourceSelection",
    "PhysicalParams",
    "HistogramDensity",
]


def __getattr__(name):
    if name == "gapmoe":
        from .gapmoe import gapmoe

        return gapmoe
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
