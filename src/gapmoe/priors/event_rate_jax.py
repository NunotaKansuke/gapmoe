from __future__ import annotations

import jax.numpy as jnp

from gapmoe.priors.event_rate import KAPPA


def jax_log_event_rate(ML: float, DL: float, DS: float, mu: float, kappa: float = KAPPA) -> jnp.ndarray:
    """Return log of the microlensing event-rate factor using JAX arrays."""

    valid = (ML > 0.0) & (DL > 0.0) & (DS > DL) & (mu > 0.0)
    pi_rel = 1000.0 * ((1.0 / DL) - (1.0 / DS))
    theta_arg = jnp.where((ML > 0.0) & (pi_rel > 0.0), ML * pi_rel * kappa, 1.0)
    theta_e = jnp.sqrt(theta_arg)
    log_gamma = 2.0 * jnp.log(DL) + jnp.log(theta_e) + jnp.log(mu)
    valid = valid & (pi_rel > 0.0) & jnp.isfinite(theta_e) & (theta_e > 0.0)
    return jnp.where(valid, log_gamma, -jnp.inf)
