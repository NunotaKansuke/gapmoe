from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit

from gapmoe.param_types.base import MappingContext
from gapmoe.param_types.binary_lens import (
    BinaryCircularParamType,
    BinaryKeplerParamType,
)
from gapmoe.param_types.single_lens import SingleLensParamType

_KAPPA = 8.1429  # mas / Msun


@dataclass
class ParamType:
    """Standard gapmoe light-curve-to-physical param_type selector.

    ``ParamType`` is itself a parameterization object and can be passed
    directly to ``GalaxyModel.parameterize()``. The selected concrete mapping stays internal
    so users do not need to choose class names for parallax/static/orbital cases.

    Examples
    --------
    ``ParamType(lens="binary", parallax=True)`` expects
    ``(t0, tE, u0, rho, piEN, piEE, DS)``.

    ``ParamType(lens="binary", parallax=False)`` expects
    ``(t0, tE, u0, rho)`` and marginalizes lens/source distances and the
    proper-motion direction.

    ``ParamType(lens="binary", parallax=True, orbital_motion="circular")``
    uses the binary circular-orbit mapping.
    """
    lens: str = "binary"
    source: str = "single"
    orbital_motion: str = "static"
    xallarap: str = "none"
    parallax: bool = False
    distance: str = "auto"
    _impl: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_impl", _select_impl(self))

    @property
    def names(self) -> tuple[str, ...]:
        return self._impl.names

    @property
    def derived_names(self) -> tuple[str, ...]:
        return tuple(getattr(self._impl, "derived_names", ()))

    @property
    def uses_mu_physical(self) -> bool:
        return bool(getattr(self._impl, "uses_mu_physical", False))

    @property
    def uses_theta_mu_physical(self) -> bool:
        return bool(getattr(self._impl, "uses_theta_mu_physical", False))

    @property
    def marginalizes_distance(self) -> bool:
        return bool(getattr(self._impl, "marginalizes_distance", False))

    @property
    def supports_distance_grid(self) -> bool:
        return bool(getattr(self._impl, "supports_distance_grid", False))

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return self._impl.to_physical(theta, context)

    def to_derived(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        if not hasattr(self._impl, "to_derived"):
            return {}
        return self._impl.to_derived(theta, context)

    def to_mu_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        if not hasattr(self._impl, "to_mu_physical"):
            raise TypeError("This param_type returns vector proper motion; use to_physical().")
        return self._impl.to_mu_physical(theta, context)

    def to_theta_mu_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        if not hasattr(self._impl, "to_theta_mu_physical"):
            raise TypeError("This param_type does not marginalize distances.")
        return self._impl.to_theta_mu_physical(theta, context)

    def with_distance(self, theta: Any, distance: Any):
        if not hasattr(self._impl, "with_distance"):
            raise TypeError("This param_type does not accept an internal distance.")
        return self._impl.with_distance(theta, distance)

    def physical_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        if not hasattr(self._impl, "physical_with_distance_grid"):
            raise TypeError("This param_type does not provide a distance-grid transform.")
        return self._impl.physical_with_distance_grid(theta, distances, context)

    def log_abs_det_jacobian_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        if not hasattr(self._impl, "log_abs_det_jacobian_with_distance_grid"):
            raise TypeError("This param_type does not provide a distance-grid Jacobian.")
        return self._impl.log_abs_det_jacobian_with_distance_grid(
            theta,
            distances,
            context,
        )

    def jax_physical_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        if not hasattr(self._impl, "jax_physical_with_distance_grid"):
            raise TypeError("This param_type does not provide a JAX distance-grid transform.")
        return self._impl.jax_physical_with_distance_grid(theta, distances, context)

    def jax_log_abs_det_jacobian_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        if not hasattr(self._impl, "jax_log_abs_det_jacobian_with_distance_grid"):
            raise TypeError("This param_type does not provide a JAX distance-grid Jacobian.")
        return self._impl.jax_log_abs_det_jacobian_with_distance_grid(
            theta,
            distances,
            context,
        )

    @property
    def distance_impl(self):
        return getattr(self._impl, "distance_impl", self._impl)

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return self._impl.log_abs_det_jacobian(jnp.asarray(theta), context)


def from_model_spec(spec: Any) -> ParamType:
    """Build a gapmoe ``ParamType`` from an external spec-like object.

    This is primarily a bridge for external tools. Normal gapmoe code should
    instantiate ``ParamType`` directly.
    """
    if isinstance(spec, ParamType):
        return spec
    if hasattr(spec, "spec"):
        spec = spec.spec
    return ParamType(
        lens=_get(spec, "lens", "binary"),
        source=_get(spec, "source", "single"),
        orbital_motion=_get(spec, "orbital_motion", "static"),
        xallarap=_get(spec, "xallarap", "none"),
        parallax=bool(_get(spec, "parallax", False)),
        distance=_get(spec, "distance", "auto"),
    )


def _select_impl(spec: ParamType):
    lens = _get(spec, "lens", "binary")
    source = _get(spec, "source", "single")
    orbital_motion = _get(spec, "orbital_motion", "static")
    xallarap = _get(spec, "xallarap", "none")
    parallax = bool(_get(spec, "parallax", False))
    distance = _get(spec, "distance", "auto")

    if source != "single":
        raise NotImplementedError("ParamType currently supports source='single' only")
    if xallarap != "none":
        raise NotImplementedError("ParamType does not support xallarap yet")
    if lens not in {"binary", "triple"}:
        raise NotImplementedError("ParamType supports lens='binary' or 'triple'")
    if distance not in {"auto", "sample", "marginalize"}:
        raise ValueError("distance must be 'auto', 'sample', or 'marginalize'")

    if orbital_motion == "static":
        if parallax:
            if distance == "marginalize":
                return _StaticParallaxMarginalDistanceParamType()
            return SingleLensParamType()
        if distance in {"auto", "marginalize"}:
            return _StaticNoParallaxMarginalDistanceParamType()
        return _StaticNoParallaxParamType()

    if distance == "marginalize":
        raise NotImplementedError("distance='marginalize' is currently supported only for static models")
    if lens != "binary":
        raise NotImplementedError(
            "lens orbital motion is currently supported only for binary lenses"
        )
    if not parallax:
        raise NotImplementedError(
            "lens orbital motion param_types currently require parallax=True"
        )
    if orbital_motion == "circular":
        return BinaryCircularParamType()
    if orbital_motion == "kepler":
        return BinaryKeplerParamType()
    raise NotImplementedError(
        "ParamType supports orbital_motion='static', 'circular', or 'kepler'"
    )


def _get(spec: Any, name: str, default: Any) -> Any:
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


@jit
def _lc_to_theta_mu_static_no_parallax(theta, thS):
    t0 = theta[0]
    tE = theta[1]
    u0 = theta[2]
    rho = theta[3]

    thE = thS / rho
    mu = thE / tE * 365.25

    return jnp.array([t0, u0, thE, mu])


@jit
def _jacobian_theta_mu_static_no_parallax(theta, thS):
    J = jacfwd(_lc_to_theta_mu_static_no_parallax)(theta, thS)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


@jit
def _lc_to_phys_static_no_parallax(theta, thS, KAPPA=_KAPPA):
    t0 = theta[0]
    tE = theta[1]
    u0 = theta[2]
    rho = theta[3]
    DL = theta[4]  # kpc
    DS = theta[5]  # kpc

    thE = thS / rho
    pi_rel = 1.0 / DL - 1.0 / DS
    ML = thE * thE / (KAPPA * pi_rel)
    mu = thE / tE * 365.25

    return jnp.array([t0, u0, ML, DL, DS, mu])


@jit
def _jacobian_static_no_parallax(theta, thS):
    J = jacfwd(_lc_to_phys_static_no_parallax)(theta, thS)
    _, lndet = jnp.linalg.slogdet(J)
    return lndet


def _thS(context: Optional[MappingContext]):
    if context is None or "thS" not in context:
        raise ValueError("context must include 'thS': source angular radius in mas.")
    return context["thS"]


def _vEarth(context: Optional[MappingContext]):
    if context is None or "vEarth" not in context:
        raise ValueError("context must include 'vEarth': (v_N, v_E) in AU/yr.")
    return context["vEarth"]


def _phys_from_static_out(out):
    return (
        out[2],  # ML  [Msun]
        out[3],  # DL  [kpc]
        out[4],  # DS  [kpc]
        out[5],  # mu [mas/yr]
    )


def _theta_mu_from_static_out(out):
    return (
        out[2],  # thetaE [mas]
        out[3],  # mu [mas/yr]
    )


class _StaticNoParallaxMarginalDistanceParamType:
    """No-parallax static mapping with marginalized lens/source distances."""

    names: tuple[str, ...] = ("t0", "tE", "u0", "rho")
    uses_theta_mu_physical = True

    def to_theta_mu_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        out = _lc_to_theta_mu_static_no_parallax(theta, _thS(context))
        return _theta_mu_from_static_out(out)

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        raise TypeError(
            "No-parallax static param_type marginalizes lens/source "
            "distances; use to_theta_mu_physical() instead."
        )

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return _jacobian_theta_mu_static_no_parallax(theta, _thS(context))


class _StaticNoParallaxParamType:
    """No-parallax static mapping selected by ``ParamType``.

    This is intentionally private; user-facing selection should go through
    ``ParamType``. The proper-motion direction is marginalized by the
    prior, so it is not part of the sampled parameter vector.
    """

    names: tuple[str, ...] = ("t0", "tE", "u0", "rho", "DL", "DS")
    uses_mu_physical = True

    def to_mu_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        out = _lc_to_phys_static_no_parallax(theta, _thS(context))
        return _phys_from_static_out(out)

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        raise TypeError(
            "No-parallax static param_type marginalizes the proper-motion "
            "direction; use to_mu_physical() instead."
        )

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        return _jacobian_static_no_parallax(theta, _thS(context))


class _StaticParallaxMarginalDistanceParamType:
    """Static parallax mapping with source distance marginalized by the prior."""

    names: tuple[str, ...] = ("t0", "tE", "u0", "rho", "piEN", "piEE")
    marginalizes_distance = True
    supports_distance_grid = True
    distance_impl = SingleLensParamType()

    def with_distance(self, theta: Any, distance: Any):
        return jnp.concatenate([jnp.asarray(theta), jnp.asarray([distance])])

    def physical_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        return _static_parallax_physical_with_distance_grid_np(theta, distances, context)

    def log_abs_det_jacobian_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        return _static_parallax_log_abs_det_with_distance_grid_np(theta, distances, context)

    def jax_physical_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        return _static_parallax_physical_with_distance_grid_jax(theta, distances, context)

    def jax_log_abs_det_jacobian_with_distance_grid(
        self,
        theta: Any,
        distances: Any,
        context: Optional[MappingContext] = None,
    ):
        return _static_parallax_log_abs_det_with_distance_grid_jax(theta, distances, context)

    def to_physical(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        raise TypeError(
            "This param_type marginalizes source distance; evaluate it "
            "through a parameterized GalaxyModel.log_density()."
        )

    def log_abs_det_jacobian(
        self,
        theta: Any,
        context: Optional[MappingContext] = None,
    ):
        raise TypeError(
            "This param_type marginalizes source distance; the full "
            "Jacobian is integrated inside a parameterized GalaxyModel.log_density()."
        )


def _static_parallax_physical_with_distance_grid_np(
    theta: Any,
    distances: Any,
    context: Optional[MappingContext],
):
    theta = np.asarray(theta, dtype=float)
    DS = np.asarray(distances, dtype=float)
    tE = theta[1]
    rho = theta[3]
    piEN = theta[4]
    piEE = theta[5]
    thE = float(_thS(context)) / rho
    piE = np.hypot(piEN, piEE)
    mu_geo = thE / tE * 365.25
    vN, vE = _vEarth(context)

    ML = np.full_like(DS, thE / (_KAPPA * piE), dtype=float)
    DL = 1.0 / (thE * piE + 1.0 / DS)
    mu_N = np.full_like(DS, mu_geo * piEN / piE + thE * piE * vN, dtype=float)
    mu_E = np.full_like(DS, mu_geo * piEE / piE + thE * piE * vE, dtype=float)
    return ML, DL, DS, mu_N, mu_E


def _static_parallax_log_abs_det_with_distance_grid_np(
    theta: Any,
    distances: Any,
    context: Optional[MappingContext],
):
    theta = np.asarray(theta, dtype=float)
    DS = np.asarray(distances, dtype=float)
    thS = float(_thS(context))
    tE = theta[1]
    rho = theta[3]
    piE = np.hypot(theta[4], theta[5])
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.log(2.0)
            + 4.0 * np.log(abs(thS))
            + 2.0 * np.log(np.abs(DS))
            + 2.0 * np.log(365.25)
            - np.log(abs(_KAPPA))
            - 3.0 * np.log(abs(tE))
            - 3.0 * np.log(abs(rho))
            - 2.0 * np.log(piE)
            - 2.0 * np.log(np.abs(thS * DS * piE + rho))
        )


def _static_parallax_physical_with_distance_grid_jax(
    theta: Any,
    distances: Any,
    context: Optional[MappingContext],
):
    theta = jnp.asarray(theta)
    DS = jnp.asarray(distances)
    tE = theta[1]
    rho = theta[3]
    piEN = theta[4]
    piEE = theta[5]
    thE = _thS(context) / rho
    piE = jnp.hypot(piEN, piEE)
    mu_geo = thE / tE * 365.25
    vN, vE = _vEarth(context)

    ML = jnp.full_like(DS, thE / (_KAPPA * piE))
    DL = 1.0 / (thE * piE + 1.0 / DS)
    mu_N = jnp.full_like(DS, mu_geo * piEN / piE + thE * piE * vN)
    mu_E = jnp.full_like(DS, mu_geo * piEE / piE + thE * piE * vE)
    return ML, DL, DS, mu_N, mu_E


def _static_parallax_log_abs_det_with_distance_grid_jax(
    theta: Any,
    distances: Any,
    context: Optional[MappingContext],
):
    theta = jnp.asarray(theta)
    DS = jnp.asarray(distances)
    thS = _thS(context)
    tE = theta[1]
    rho = theta[3]
    piE = jnp.hypot(theta[4], theta[5])
    return (
        jnp.log(2.0)
        + 4.0 * jnp.log(jnp.abs(thS))
        + 2.0 * jnp.log(jnp.abs(DS))
        + 2.0 * jnp.log(365.25)
        - jnp.log(jnp.abs(_KAPPA))
        - 3.0 * jnp.log(jnp.abs(tE))
        - 3.0 * jnp.log(jnp.abs(rho))
        - 2.0 * jnp.log(piE)
        - 2.0 * jnp.log(jnp.abs(thS * DS * piE + rho))
    )
