from __future__ import annotations

from typing import Any, Optional, Protocol


MappingContext = dict[str, Any]


class Parameterization(Protocol):
    """Protocol for light-curve-to-physical parameter mappings.

    A Parameterization maps a user/light-curve parameter vector *theta* into
    the five physical parameters ``(ML, DL, DS, mu_N, mu_E)`` that the Galactic
    density model expects, and provides the log-Jacobian term for the change of
    variables.

    ``names`` documents the expected parameter ordering in *theta*.

    **Context keys**

    ``to_physical`` and ``log_abs_det_jacobian`` receive an event-specific
    context dict:

    - ``"thS"`` : float — source angular radius in mas.
      Required for rho-based parameterizations; not needed when theta already
      contains the Einstein radius directly.
    - ``"vEarth"`` : tuple[float, float] — heliocentric Earth velocity at the
      reference time, ``(v_N, v_E)`` in AU/yr.
      Obtain from ``gapmoe.parameterizations.calc_vEarth``.

    **Implementing a custom parameterization**

    Subclass or duck-type this protocol::

        class MyParam:
            names = ("t0", "tE", "u0", ...)

            def to_physical(self, theta, context=None):
                # ... your transformation ...
                return ML, DL, DS, mu_N, mu_E

            def log_abs_det_jacobian(self, theta, context=None):
                # ... JAX jacfwd or analytic formula ...
                return lndet
    """

    names: tuple[str, ...]

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        """Return ``(ML [Msun], DL [kpc], DS [kpc], mu_N [mas/yr], mu_E [mas/yr])``.

        Implementations may return Python scalars for eager NumPy use or JAX
        scalar arrays/tracers when called under ``jax.jit``.
        """
        ...

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        """Return ``log |det J|`` of the full parameter transformation."""
        ...
