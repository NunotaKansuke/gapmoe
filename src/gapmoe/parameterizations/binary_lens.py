"""Binary-lens light-curve parameterizations.

Each class maps a binary-lens light-curve parameter vector to the five physical
parameters ``(ML, DL, DS, mu_N, mu_E)`` used by the Galactic density model, and
provides the corresponding log-Jacobian for use with ``GalacticPrior``.

Two orbital models are provided:

* **Circular** — the lens binary is assumed to be in a circular orbit at the
  moment of the microlensing event, described by the instantaneous velocity
  components ``(gamma1, gamma2, gamma3)``.
* **Kepler** — the full Keplerian orbit is parameterized by ``(gamma1, gamma2,
  gamma3, r_s, a_s)``.

For each orbital model there is a *rho*-based variant (``rho`` = source radius
in units of the Einstein radius) and a *thE*-based variant where the Einstein
radius ``thE`` is a direct parameter instead.

All classes require the following context keys:

- ``"vEarth"`` : ``tuple[float, float]`` — heliocentric Earth velocity
  ``(v_N, v_E)`` in AU/yr at the reference time ``t0``.
  Compute with ``gapmoe.parameterizations.calc_vEarth``.
- ``"thS"`` : float — source angular radius in mas.
  Required for ``BinaryCircularParameterization`` and
  ``BinaryKeplerParameterization`` (rho-based); not needed for the thE
  variants.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
from jax import jacfwd, jit

from gapmoe.parameterizations.base import MappingContext

_G = 2.959122082855911e-4   # AU^3 / (Msun * day^2)
_KAPPA = 8.1429             # mas / Msun


# ---------------------------------------------------------------------------
# JAX kernel functions — exact math from the original parametrics module.
# These are private; use the Parameterization classes instead.
# ---------------------------------------------------------------------------

@jit
def _lc_to_phys_circular(theta, thS, vEarth, G=_G, KAPPA=_KAPPA):
    G = _G
    KAPPA = _KAPPA
    t0 = theta[0]
    tE = theta[1]
    u0 = theta[2]
    rho = theta[3]
    q = theta[4]
    s = theta[5]
    alpha = theta[6]
    piEN = theta[7]
    piEE = theta[8]
    gamma1 = theta[9]
    gamma2 = theta[10]
    gamma3 = theta[11]

    piE = jnp.sqrt(piEN**2 + piEE**2)
    thE = thS / rho
    ML = thE / KAPPA / piE
    murel_geo = thE / tE * 365.25
    murel_N_geo = murel_geo * piEN / piE
    murel_E_geo = murel_geo * piEE / piE

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_ratio = jnp.sqrt(1 + (gamma1 / gamma3)**2)
    orbital_scale = jnp.cbrt((s**3) * gamma_sq * gamma_ratio / (ML * G))
    DS = 1 / ((orbital_scale - piE) * thE)

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]

    RE = DL * thE
    orbital_radi = RE * s * gamma_ratio
    r = RE * s * jnp.array([1, 0, -gamma1 / gamma3])
    v = RE * s * jnp.array([gamma1, gamma2, gamma3])
    h = jnp.cross(r, v)
    z = h / jnp.sqrt(jnp.dot(h, h))
    cos_i = z[2]
    sin_i = jnp.sqrt(1 - cos_i**2)
    sin_Om0, cos_Om0 = z[0] / sin_i, -z[1] / sin_i
    Om0 = jnp.arctan2(sin_Om0, cos_Om0)
    Om_NE = Om0 + jnp.arctan2(piEE, piEN) - alpha
    Om_NE = jnp.arctan2(jnp.sin(Om_NE), jnp.cos(Om_NE))
    x = jnp.array([cos_Om0, sin_Om0, 0])
    y = jnp.cross(z, x)
    cos_phi0 = jnp.dot(r, x) / jnp.sqrt(jnp.dot(r, r))
    sin_phi0 = jnp.dot(r, y) / jnp.sqrt(jnp.dot(r, r))
    phi0 = jnp.arctan2(sin_phi0, cos_phi0)

    return jnp.array([t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel,
                      orbital_radi, cos_i, Om_NE, phi0])


@jit
def _jacobian_circular(theta, thS, vEarth):
    J = jacfwd(_lc_to_phys_circular)(theta, thS, vEarth)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


@jit
def _lc_to_phys_circular_use_thE(theta, vEarth, G=_G, KAPPA=_KAPPA):
    t0 = theta[0]
    tE = theta[1]
    u0 = theta[2]
    thE = theta[3]
    q = theta[4]
    s = theta[5]
    alpha = theta[6]
    piEN = theta[7]
    piEE = theta[8]
    gamma1 = theta[9]
    gamma2 = theta[10]
    gamma3 = theta[11]

    piE = jnp.sqrt(piEN**2 + piEE**2)
    ML = thE / KAPPA / piE
    murel_geo = thE / tE * 365.25
    murel_N_geo = murel_geo * piEN / piE
    murel_E_geo = murel_geo * piEE / piE

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    gamma_ratio = jnp.sqrt(1 + (gamma1 / gamma3)**2)
    orbital_scale = jnp.cbrt((s**3) * gamma_sq * gamma_ratio / (ML * G))
    DS = 1 / ((orbital_scale - piE) * thE)

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]

    RE = DL * thE
    orbital_radi = RE * s * gamma_ratio
    r = RE * s * jnp.array([1, 0, -gamma1 / gamma3])
    v = RE * s * jnp.array([gamma1, gamma2, gamma3])
    h = jnp.cross(r, v)
    z = h / jnp.sqrt(jnp.dot(h, h))
    cos_i = z[2]
    sin_i = jnp.sqrt(1 - cos_i**2)
    sin_Om0, cos_Om0 = z[0] / sin_i, -z[1] / sin_i
    Om0 = jnp.arctan2(sin_Om0, cos_Om0)
    Om_NE = Om0 + jnp.arctan2(piEE, piEN) - alpha
    Om_NE = jnp.arctan2(jnp.sin(Om_NE), jnp.cos(Om_NE))
    x = jnp.array([cos_Om0, sin_Om0, 0])
    y = jnp.cross(z, x)
    cos_phi0 = jnp.dot(r, x) / jnp.sqrt(jnp.dot(r, r))
    sin_phi0 = jnp.dot(r, y) / jnp.sqrt(jnp.dot(r, r))
    phi0 = jnp.arctan2(sin_phi0, cos_phi0)

    return jnp.array([t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel,
                      orbital_radi, cos_i, Om_NE, phi0])


@jit
def _jacobian_circular_use_thE(theta, vEarth):
    J = jacfwd(_lc_to_phys_circular_use_thE)(theta, vEarth)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


@jit
def _lc_to_phys_kepler(theta, thS, vEarth, G=_G, KAPPA=_KAPPA):
    G = _G
    KAPPA = _KAPPA
    t0 = theta[0]
    tE = theta[1]
    u0 = theta[2]
    rho = theta[3]
    q = theta[4]
    s = theta[5]
    alpha = theta[6]
    piEN = theta[7]
    piEE = theta[8]
    gamma1 = theta[9]
    gamma2 = theta[10]
    gamma3 = theta[11]
    r_s = theta[12]
    a_s = theta[13]

    piE = jnp.sqrt(piEN**2 + piEE**2)
    thE = thS / rho
    ML = thE / KAPPA / piE
    murel_geo = thE / tE * 365.25
    murel_N_geo = murel_geo * piEN / piE
    murel_E_geo = murel_geo * piEE / piE

    gamma_sq = gamma1**2 + gamma2**2 + gamma3**2
    orbital_scale = jnp.cbrt(
        (s**3) * a_s * jnp.sqrt(1 + r_s**2) * gamma_sq / (ML * G) / (2 * a_s - 1)
    )
    DS = 1 / ((orbital_scale - piE) * thE)

    pi_rel = thE * piE
    pi_S = 1 / DS
    pi_L = pi_rel + pi_S
    DL = 1 / pi_L

    murel_N_hel = murel_N_geo + thE * piE * vEarth[0]
    murel_E_hel = murel_E_geo + thE * piE * vEarth[1]

    RE = DL * thE
    a_norm = a_s * s * jnp.sqrt(1 + r_s**2)
    orbital_radi = RE * a_norm

    r = RE * s * jnp.array([1, 0, r_s])
    v = RE * s * jnp.array([gamma1, gamma2, gamma3])
    h = jnp.cross(r, v)
    A = jnp.cross(v, h) / (G * ML) - r / jnp.sqrt(jnp.dot(r, r))
    e = jnp.sqrt(jnp.dot(A, A))
    z = h / jnp.sqrt(jnp.dot(h, h))
    x = A / e
    y = jnp.cross(z, x)
    cos_i = z[2]
    sin_i = jnp.sqrt(1 - cos_i**2)
    sin_Om0, cos_Om0 = z[0] / sin_i, -z[1] / sin_i
    Om0 = jnp.arctan2(sin_Om0, cos_Om0)
    Om_NE = Om0 + jnp.arctan2(piEE, piEN) - alpha
    Om_NE = jnp.arctan2(jnp.sin(Om_NE), jnp.cos(Om_NE))
    sin_om, cos_om = x[2] / sin_i, y[2] / sin_i
    om = jnp.arctan2(sin_om, cos_om)
    cos_nu = jnp.dot(r, x) / jnp.sqrt(jnp.dot(r, r))
    sin_nu = jnp.dot(r, y) / jnp.sqrt(jnp.dot(r, r))
    nu = jnp.arctan2(sin_nu, cos_nu)

    return jnp.array([t0, u0, q, ML, DL, DS, murel_N_hel, murel_E_hel,
                      orbital_radi, e, cos_i, Om_NE, om, nu])


@jit
def _jacobian_kepler(theta, thS, vEarth):
    J = jacfwd(_lc_to_phys_kepler)(theta, thS, vEarth)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


# ---------------------------------------------------------------------------
# Physical-parameter indices in the kernel output arrays
# ---------------------------------------------------------------------------

# [t0, u0, q, ML, DL, DS, mu_N, mu_E, orbital_radi, ...]
_PHYS_IDX_BINARY = (3, 4, 5, 6, 7)


def _phys_from_binary_out(out):
    return (
        out[3],  # ML  [Msun]
        out[4],  # DL  [kpc]
        out[5],  # DS  [kpc]
        out[6],  # mu_N [mas/yr]
        out[7],  # mu_E [mas/yr]
    )


def _vEarth(context: Optional[MappingContext]):
    if context is None or "vEarth" not in context:
        raise ValueError(
            "context must include 'vEarth': (v_N, v_E) in AU/yr. "
            "Use gapmoe.parameterizations.calc_vEarth to compute it."
        )
    return context["vEarth"]


def _thS(context: Optional[MappingContext]):
    if context is None or "thS" not in context:
        raise ValueError(
            "context must include 'thS': source angular radius in mas."
        )
    return context["thS"]


# ---------------------------------------------------------------------------
# Public parameterization classes
# ---------------------------------------------------------------------------

class BinaryCircularParameterization:
    """Binary-lens circular-orbit parameterization (rho-based).

    Parameter vector ``theta`` must have 12 elements in this order:

    ``(t0, tE, u0, rho, q, s, alpha, piEN, piEE, gamma1, gamma2, gamma3)``

    Required context keys: ``"thS"``, ``"vEarth"``.
    """

    names: tuple[str, ...] = (
        "t0", "tE", "u0", "rho", "q", "s", "alpha",
        "piEN", "piEE", "gamma1", "gamma2", "gamma3",
    )

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        out = _lc_to_phys_circular(theta, _thS(context), _vEarth(context))
        return _phys_from_binary_out(out)

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return _jacobian_circular(theta, _thS(context), _vEarth(context))


class BinaryCircularUseThEParameterization:
    """Binary-lens circular-orbit parameterization (thE-based).

    Like :class:`BinaryCircularParameterization` but uses the Einstein radius
    ``thE`` directly instead of the source-radius ratio ``rho``.

    Parameter vector ``theta`` must have 12 elements:

    ``(t0, tE, u0, thE, q, s, alpha, piEN, piEE, gamma1, gamma2, gamma3)``

    Required context key: ``"vEarth"`` (``"thS"`` is not needed).
    """

    names: tuple[str, ...] = (
        "t0", "tE", "u0", "thE", "q", "s", "alpha",
        "piEN", "piEE", "gamma1", "gamma2", "gamma3",
    )

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        out = _lc_to_phys_circular_use_thE(theta, _vEarth(context))
        return _phys_from_binary_out(out)

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return _jacobian_circular_use_thE(theta, _vEarth(context))


class BinaryKeplerParameterization:
    """Binary-lens Keplerian-orbit parameterization (rho-based).

    Parameter vector ``theta`` must have 14 elements:

    ``(t0, tE, u0, rho, q, s, alpha, piEN, piEE, gamma1, gamma2, gamma3, r_s, a_s)``

    Required context keys: ``"thS"``, ``"vEarth"``.
    """

    names: tuple[str, ...] = (
        "t0", "tE", "u0", "rho", "q", "s", "alpha",
        "piEN", "piEE", "gamma1", "gamma2", "gamma3",
        "r_s", "a_s",
    )

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        out = _lc_to_phys_kepler(theta, _thS(context), _vEarth(context))
        return _phys_from_binary_out(out)

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return _jacobian_kepler(theta, _thS(context), _vEarth(context))
