from .event_rate import log_event_rate
from .galactic import GalacticModel

__all__ = ["GalacticModel", "JaxGalacticModel", "jax_log_event_rate", "log_event_rate"]


def __getattr__(name):
    if name == "JaxGalacticModel":
        from .galactic_jax import JaxGalacticModel

        return JaxGalacticModel
    if name == "jax_log_event_rate":
        from .event_rate_jax import jax_log_event_rate

        return jax_log_event_rate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
