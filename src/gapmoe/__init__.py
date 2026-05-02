__all__ = ["gapmoe", "PreRunner", "PreRunResult", "SourceSelection"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
