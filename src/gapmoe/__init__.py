"""gapmoe public API."""

__all__ = ["Model", "SourcePopulation", "AgeMetallicityPoint"]


def __getattr__(name):
    if name == "Model":
        from .priors.high_level import Model
        return Model
    if name in {"SourcePopulation", "AgeMetallicityPoint"}:
        from .source_selection import AgeMetallicityPoint, SourcePopulation
        return {"SourcePopulation": SourcePopulation, "AgeMetallicityPoint": AgeMetallicityPoint}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
