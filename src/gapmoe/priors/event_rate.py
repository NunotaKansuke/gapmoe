from __future__ import annotations

from math import isfinite, log, sqrt

from gapmoe.physical import PhysicalParams


KAPPA = 8.1439


def log_event_rate(params: PhysicalParams, kappa: float = KAPPA) -> float:
    """Return log of the microlensing event-rate factor.

    Distances are expected in pc. For pc distances, pi_rel[mas] is
    1000 * (1 / DL - 1 / DS).
    """

    if params.ML <= 0.0 or params.DL <= 0.0 or params.DS <= params.DL or params.mu <= 0.0:
        return float("-inf")

    pi_rel = 1000.0 * ((1.0 / params.DL) - (1.0 / params.DS))
    if pi_rel <= 0.0:
        return float("-inf")

    theta_e = sqrt(params.ML * pi_rel * kappa)
    if not isfinite(theta_e) or theta_e <= 0.0:
        return float("-inf")

    return 2.0 * log(params.DL) + log(theta_e) + log(params.mu)
