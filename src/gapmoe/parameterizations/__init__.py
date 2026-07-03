"""Light-curve-to-physical parameter mappings for gapmoe.

Each parameterization converts a light-curve parameter vector *theta* into the
five physical parameters ``(ML, DL, DS, mu_N, mu_E)`` expected by the Galactic
density model, and provides the log-Jacobian of that transformation for use
with :class:`~gapmoe.priors.GalacticPrior`.

**Choosing a parameterization**

+-------------------------------------+------+---------+-----+
| Class                               | Lens | Orbit   | θ   |
+=====================================+======+=========+=====+
| BinaryCircularParameterization      | 2    | circular| ρ   |
+-------------------------------------+------+---------+-----+
| BinaryCircularUseThEParameterization| 2    | circular| θE  |
+-------------------------------------+------+---------+-----+
| BinaryKeplerParameterization        | 2    | Kepler  | ρ   |
+-------------------------------------+------+---------+-----+
| SingleLensParameterization          | 1    | —       | ρ   |
+-------------------------------------+------+---------+-----+
| SingleLensUseThEParameterization    | 1    | —       | θE  |
+-------------------------------------+------+---------+-----+

The *θ* column indicates whether the source-radius ratio *ρ* (requires
``"thS"`` in the context) or the Einstein radius *θE* directly is used.

**Context**

Pass an event-specific context dict to ``log_prob`` or directly to
``to_physical``/``log_abs_det_jacobian``::

    ctx = {
        "thS": 0.5,                             # source angular radius, mas
        "vEarth": calc_vEarth(t0_jd, ra, dec),  # (v_N, v_E), AU/yr
    }

**Custom parameterizations**

Implement the :class:`~gapmoe.parameterizations.base.Parameterization`
protocol to add your own::

    class MyParam:
        names = ("t0", "tE", ...)

        def to_physical(self, theta, context=None):
            return ML, DL, DS, mu_N, mu_E

        def log_abs_det_jacobian(self, theta, context=None):
            return lndet

    prior = GalacticPrior(density, parameterization=MyParam())
"""

__all__ = [
    "Parameterization",
    "MappingContext",
    "BinaryCircularParameterization",
    "BinaryCircularUseThEParameterization",
    "BinaryKeplerParameterization",
    "SingleLensParameterization",
    "SingleLensUseThEParameterization",
    "calc_vEarth",
]


def __getattr__(name):
    if name in {"Parameterization", "MappingContext"}:
        from gapmoe.parameterizations.base import MappingContext, Parameterization

        exports = {"Parameterization": Parameterization, "MappingContext": MappingContext}
        return exports[name]
    if name in {
        "BinaryCircularParameterization",
        "BinaryCircularUseThEParameterization",
        "BinaryKeplerParameterization",
    }:
        from gapmoe.parameterizations.binary_lens import (
            BinaryCircularParameterization,
            BinaryCircularUseThEParameterization,
            BinaryKeplerParameterization,
        )

        exports = {
            "BinaryCircularParameterization": BinaryCircularParameterization,
            "BinaryCircularUseThEParameterization": BinaryCircularUseThEParameterization,
            "BinaryKeplerParameterization": BinaryKeplerParameterization,
        }
        return exports[name]
    if name in {"SingleLensParameterization", "SingleLensUseThEParameterization"}:
        from gapmoe.parameterizations.single_lens import SingleLensParameterization, SingleLensUseThEParameterization

        exports = {
            "SingleLensParameterization": SingleLensParameterization,
            "SingleLensUseThEParameterization": SingleLensUseThEParameterization,
        }
        return exports[name]
    if name == "calc_vEarth":
        from gapmoe.EarthMotion import calc_vEarth

        return calc_vEarth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
