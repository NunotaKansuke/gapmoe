"""Light-curve-to-physical parameter mappings for gapmoe.

Each param_type converts a light-curve parameter vector *theta* into physical
parameters. The first five values are ``(ML, DL, DS, mu_N, mu_E)`` expected by
the Galactic density model; orbital-motion param_types append derived orbital
elements after those five values. Param_types also provide the log-Jacobian of
the density-coordinate transformation owned by :class:`gapmoe.Model`.

**Choosing a param_type**

The public selector is ``ParamType``. It keeps the light-curve model
choice in one small object and hides the concrete mapping class names::

    p = ParamType(lens="binary", parallax=True, orbital_motion="static")
    galaxy = gapmoe.Model(p, ...)

``parallax=True, orbital_motion="static"`` expects
``(t0, tE, u0, rho, piEN, piEE, DS)`` by default. Use
``distance="marginalize"`` to integrate over ``DS`` and expect
``(t0, tE, u0, rho, piEN, piEE)``. ``parallax=False`` expects
``(t0, tE, u0, rho)`` and marginalizes lens/source distances plus the
proper-motion direction. Use ``distance="sample"`` to sample ``DL`` and ``DS``
explicitly. Binary lens orbital motion can be selected with
``orbital_motion="circular"`` or ``"kepler"``; distance marginalization is
currently static-only.

No-parallax models use the histogram backend's native precomputed projections.
When source photometry or additional hidden-physical priors require an
uncollapsed integrand, sample the corresponding physical distances explicitly.

**Context**

Pass an event-specific context dict to ``log_prob`` or directly to
``to_physical``/``log_abs_det_jacobian``::

    ctx = {
        "thS": 0.5,                             # source angular radius, mas
        "vEarth": calc_vEarth(t0_jd, ra, dec),  # (v_N, v_E), AU/yr
    }

**Custom param_types**

Implement the :class:`~gapmoe.param_types.base.ParamTypeProtocol`
protocol to add your own::

    class MyParam:
        names = ("t0", "tE", ...)

        def to_physical(self, theta, context=None):
            return ML, DL, DS, mu_N, mu_E, ...

        def log_abs_det_jacobian(self, theta, context=None):
            return lndet

    galaxy = gapmoe.Model(MyParam(), ...)
"""

__all__ = [
    "ParamType",
    "ParamTypeProtocol",
    "MappingContext",
    "BinaryCircularParamType",
    "BinaryCircularUseThEParamType",
    "BinaryKeplerParamType",
    "SingleLensParamType",
    "SingleLensUseThEParamType",
    "from_model_spec",
    "calc_vEarth",
]


def __getattr__(name):
    if name in {"MappingContext", "ParamTypeProtocol"}:
        from gapmoe.param_types.base import MappingContext, ParamTypeProtocol

        exports = {
            "MappingContext": MappingContext,
            "ParamTypeProtocol": ParamTypeProtocol,
        }
        return exports[name]
    if name in {
        "BinaryCircularParamType",
        "BinaryCircularUseThEParamType",
        "BinaryKeplerParamType",
    }:
        from gapmoe.param_types.binary_lens import (
            BinaryCircularParamType,
            BinaryCircularUseThEParamType,
            BinaryKeplerParamType,
        )

        exports = {
            "BinaryCircularParamType": BinaryCircularParamType,
            "BinaryCircularUseThEParamType": BinaryCircularUseThEParamType,
            "BinaryKeplerParamType": BinaryKeplerParamType,
        }
        return exports[name]
    if name in {"SingleLensParamType", "SingleLensUseThEParamType"}:
        from gapmoe.param_types.single_lens import (
            SingleLensParamType,
            SingleLensUseThEParamType,
        )

        exports = {
            "SingleLensParamType": SingleLensParamType,
            "SingleLensUseThEParamType": SingleLensUseThEParamType,
        }
        return exports[name]
    if name == "calc_vEarth":
        from gapmoe.EarthMotion import calc_vEarth

        return calc_vEarth
    if name in {"ParamType", "from_model_spec"}:
        from gapmoe.param_types.param_type import (
            ParamType,
            from_model_spec,
        )

        exports = {"ParamType": ParamType, "from_model_spec": from_model_spec}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
