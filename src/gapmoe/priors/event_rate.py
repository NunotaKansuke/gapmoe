from __future__ import annotations

from math import isfinite, log, sqrt


KAPPA = 8.1439


def log_event_rate(ML: float, DL: float, DS: float, mu: float, kappa: float = KAPPA) -> float:
    """Return log of the microlensing event-rate factor.

    Distances are expected in kpc. For kpc distances, pi_rel[mas] = 1/DL - 1/DS.
    """

    if ML <= 0.0 or DL <= 0.0 or DS <= DL or mu <= 0.0:
        return float("-inf")

    pi_rel = (1.0 / DL) - (1.0 / DS)
    if pi_rel <= 0.0:
        return float("-inf")

    theta_e = sqrt(ML * pi_rel * kappa)
    if not isfinite(theta_e) or theta_e <= 0.0:
        return float("-inf")

    return 2.0 * log(DL) + log(theta_e) + log(mu)
