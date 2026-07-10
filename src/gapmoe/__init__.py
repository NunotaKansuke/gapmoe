"""gapmoe public API."""

__all__ = ["Model"]


def __getattr__(name):
    if name == "Model":
        from .priors.high_level import Model

        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
