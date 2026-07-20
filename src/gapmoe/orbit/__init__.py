"""Orbit transformation utilities.

* :mod:`gapmoe.orbit._kepler_core` — the shared Kepler solver and
  perifocal-to-sky rotation helpers (single implementation; see PSK plan
  section 4.1).
* :mod:`gapmoe.orbit.projected_kepler` — the Projected-Separation Kepler
  (PSK) pure transformation module.  Has no hard JAX dependency itself (it
  takes an ``xp`` array-module argument, numpy by default), but is
  deliberately not re-exported here: callers should import it explicitly so
  the (still-experimental) PSK path stays opt-in.
"""

from gapmoe.orbit._kepler_core import (
    apply_orientation,
    rotate_z,
    solve_kepler,
    wrap_angle,
)

__all__ = [
    "apply_orientation",
    "rotate_z",
    "solve_kepler",
    "wrap_angle",
]
