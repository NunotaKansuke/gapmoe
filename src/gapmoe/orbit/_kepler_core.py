"""Shared Kepler solver and orientation helpers.

Single implementation shared by the H3 orbit-completion proposal sampler
(:mod:`gapmoe.param_types._orbit_proposal`) and the Projected-Separation
Kepler transformation (:mod:`gapmoe.orbit.projected_kepler`).  The PSK plan
(section 4.1) forbids independent duplicate implementations of the Kepler
solve and the perifocal-to-sky rotation, so both code paths import from here.

Every function takes an ``xp`` module argument (``numpy`` by default, or
``jax.numpy``) and is written against the subset of the array API the two
libraries share, so the exact same arithmetic serves the vectorized NumPy
path and the JAX (jit/AD-compatible) path.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "G_AU3_MSUN_DAY2",
    "apply_orientation",
    "rotate_z",
    "solve_kepler",
    "wrap_angle",
]

#: Gravitational constant in AU^3 / (Msun * day^2). Single canonical
#: definition for any orbit-package code needing GM in these units (plan
#: section 4.1's "no duplicate implementations" applies to physical
#: constants too, not just the Kepler solve). ``gapmoe.param_types
#: .binary_lens`` defines its own copy (``_G``) for historical reasons;
#: the two are asserted equal in ``tests/test_projected_kepler.py`` so any
#: future drift is caught rather than silently diverging.
G_AU3_MSUN_DAY2 = 2.959122082855911e-4


def solve_kepler(M, e, n_iter: int = 40, xp=np):
    """Fixed-iteration Newton solve of the elliptic Kepler equation.

    Solves ``E - e*sin(E) = M`` for the eccentric anomaly ``E`` with a fixed
    number of Newton steps (no data-dependent control flow, hence JAX
    jit/vmap/grad compatible).  Intended for ``0 <= e < 1``; for ``e >= 1``
    the iteration is not meaningful and the caller must flag the input.

    Parameters are broadcast against each other; angles in radians.
    """
    E = M + e * xp.sin(M)
    for _ in range(n_iter):
        f = E - e * xp.sin(E) - M
        fp = 1.0 - e * xp.cos(E)
        E = E - f / fp
    return E


def apply_orientation(u_pf, om, cos_i, xp=np):
    """``R_x(i) @ R_z(om)`` applied to perifocal vectors ``u_pf`` (..., 3).

    ``om`` and ``cos_i`` broadcast against the leading axes of ``u_pf``.
    ``sin(i)`` is taken as ``+sqrt(1 - cos_i**2)`` (i in [0, pi], convention
    C12 of the PSK convention document).
    """
    co, so = xp.cos(om), xp.sin(om)
    x = co * u_pf[..., 0] - so * u_pf[..., 1]
    y = so * u_pf[..., 0] + co * u_pf[..., 1]
    z = u_pf[..., 2]
    ci = cos_i
    si = xp.sqrt(xp.clip(1.0 - ci**2, 0.0, 1.0))
    return xp.stack([x, ci * y - si * z, si * y + ci * z], axis=-1)


def rotate_z(u, angle, xp=np):
    """``R_z(angle)`` applied to vectors ``u`` (..., 3).

    Positive ``angle`` rotates x toward y (right-handed about +z), matching
    the frame-T handedness of the PSK convention document (section 1).
    """
    c, s = xp.cos(angle), xp.sin(angle)
    return xp.stack(
        [
            c * u[..., 0] - s * u[..., 1],
            s * u[..., 0] + c * u[..., 1],
            u[..., 2],
        ],
        axis=-1,
    )


def wrap_angle(x, xp=np):
    """Wrap angles to (-pi, pi] via atan2 (PSK convention C14)."""
    return xp.arctan2(xp.sin(x), xp.cos(x))
