"""Single-lens light-curve param_types.

Each class maps a single-lens light-curve parameter vector to the five physical
parameters ``(ML, DL, DS, mu_N, mu_E)`` used by the Galactic density model, and
provides the corresponding log-Jacobian for ``gapmoe.Model``.

Two variants are provided:

* **rho-based** — uses the source-radius ratio ``rho`` together with an
  event-specific source angular radius ``thS`` from the context.
* **thE-based** — uses the Einstein radius ``thE`` directly; no ``thS`` is
  needed.

In both variants the source distance ``DS`` is a free parameter in *theta*
(given in kpc), allowing the prior to be applied to single-lens events where
the source distance is not separately constrained.

All classes require the following context key:

- ``"vEarth"`` : ``tuple[float, float]`` — heliocentric Earth velocity
  ``(v_N, v_E)`` in AU/yr at the reference time ``t0``.
  Compute with ``gapmoe.param_types.calc_vEarth``.

The rho-based class additionally requires:

- ``"thS"`` : float — source angular radius in mas.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
from jax import jacfwd, jit

from gapmoe.param_types.base import MappingContext

_G = 2.959122082855911e-4   # AU^3 / (Msun * day^2)
_KAPPA = 8.1429             # mas / Msun


# ---------------------------------------------------------------------------
# JAX kernel functions — exact math from the original parametrics module.
# These are private; use the ParamType classes instead.
# ---------------------------------------------------------------------------

@jit
def _lc_to_phys_single(theta, thS, vEarth, G=_G, KAPPA=_KAPPA):
    G = _G
    KAPPA = _KAPPA
    t0 = theta[0]
    tE = theta[1]
    u0 = theta[2]
    rho = theta[3]
    piEN = theta[4]
    piEE = theta[5]
    DS = theta[6]  # kpc

    piE = jnp.sqrt(piEN**2 + piEE**2)
    thE = thS / rho
    ML = thE / KAPPA / piE
    murel_geo = thE / tE * 365.25
    murel_N_geo = murel_geo * piEN / piE
    murel_E_geo = murel_geo * piEE / piE

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]

    return jnp.array([t0, u0, ML, DL, DS, murel_N_hel, murel_E_hel])


@jit
def _jacobian_single(theta, thS, vEarth):
    J = jacfwd(_lc_to_phys_single)(theta, thS, vEarth)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


@jit
def _lc_to_phys_single_use_thE(theta, vEarth, G=_G, KAPPA=_KAPPA):
    t0 = theta[0]
    tE = theta[1]
    u0 = theta[2]
    thE = theta[3]  # mas
    piEN = theta[4]
    piEE = theta[5]
    DS = theta[6]  # kpc

    piE = jnp.sqrt(piEN**2 + piEE**2)
    ML = thE / KAPPA / piE
    murel_geo = thE / tE * 365.25
    murel_N_geo = murel_geo * piEN / piE
    murel_E_geo = murel_geo * piEE / piE

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]

    return jnp.array([t0, u0, ML, DL, DS, murel_N_hel, murel_E_hel])


@jit
def _jacobian_single_use_thE(theta, vEarth):
    J = jacfwd(_lc_to_phys_single_use_thE)(theta, vEarth)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


# ---------------------------------------------------------------------------
# Helper — physical parameters from single-lens kernel output
# [t0, u0, ML, DL, DS, mu_N, mu_E]
# ---------------------------------------------------------------------------

def _phys_from_single_out(out):
    return (
        out[2],  # ML  [Msun]
        out[3],  # DL  [kpc]
        out[4],  # DS  [kpc]
        out[5],  # mu_N [mas/yr]
        out[6],  # mu_E [mas/yr]
    )


def _vEarth(context: Optional[MappingContext]):
    if context is None or "vEarth" not in context:
        raise ValueError(
            "context must include 'vEarth': (v_N, v_E) in AU/yr. "
            "Use gapmoe.param_types.calc_vEarth to compute it."
        )
    return context["vEarth"]


def _thS(context: Optional[MappingContext]):
    if context is None or "thS" not in context:
        raise ValueError(
            "context must include 'thS': source angular radius in mas."
        )
    return context["thS"]


# ---------------------------------------------------------------------------
# Public param_type classes
# ---------------------------------------------------------------------------

class SingleLensParamType:
    """Single-lens param_type with free source distance (rho-based).

    Parameter vector ``theta`` must have 7 elements in this order:

    ``(t0, tE, u0, rho, piEN, piEE, DS)``

    where ``DS`` is the source distance in **kpc**.

    Required context keys: ``"thS"``, ``"vEarth"``.
    """

    names: tuple[str, ...] = ("t0", "tE", "u0", "rho", "piEN", "piEE", "DS")

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        out = _lc_to_phys_single(theta, _thS(context), _vEarth(context))
        return _phys_from_single_out(out)

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return _jacobian_single(theta, _thS(context), _vEarth(context))


class SingleLensUseThEParamType:
    """Single-lens param_type with free source distance (thE-based).

    Like :class:`SingleLensParamType` but uses the Einstein radius
    ``thE`` directly instead of the source-radius ratio ``rho``.

    Parameter vector ``theta`` must have 7 elements:

    ``(t0, tE, u0, thE, piEN, piEE, DS)``

    where ``thE`` is in mas and ``DS`` is in **kpc**.

    Required context key: ``"vEarth"`` (``"thS"`` is not needed).
    """

    names: tuple[str, ...] = ("t0", "tE", "u0", "thE", "piEN", "piEE", "DS")

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        out = _lc_to_phys_single_use_thE(theta, _vEarth(context))
        return _phys_from_single_out(out)

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return _jacobian_single_use_thE(theta, _vEarth(context))
