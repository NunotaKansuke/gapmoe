from .event_rate import log_event_rate
from .cmd import CmdGalacticModel
from .galactic import GalacticModel
from .source import EventPrior5D, SourceCmdPrior

__all__ = ["GalacticModel", "CmdGalacticModel", "EventPrior5D", "SourceCmdPrior", "MappedGalacticModel", "log_event_rate_backend", "log_event_rate"]


def __getattr__(name):
    if name == "MappedGalacticModel":
        from .mapped import MappedGalacticModel

        return MappedGalacticModel
    if name == "log_event_rate_backend":
        from .event_rate_backend import log_event_rate_backend

        return log_event_rate_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
