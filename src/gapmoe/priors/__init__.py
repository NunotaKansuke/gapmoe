from .event_rate import log_event_rate
from .galactic import GalacticPrior

__all__ = ["GalacticPrior", "JaxGalacticPrior", "jax_log_event_rate", "log_event_rate"]


def __getattr__(name):
    if name == "JaxGalacticPrior":
        from .galactic_jax import JaxGalacticPrior

        return JaxGalacticPrior
    if name == "jax_log_event_rate":
        from .event_rate_jax import jax_log_event_rate

        return jax_log_event_rate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
