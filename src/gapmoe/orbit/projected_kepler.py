"""Projected-Separation Kepler (PSK) pure transformation module.

Implements the forward map
``(log_s, projected_pa, h, k, cos_i, mean_longitude) -> (a, e, i, Omega,
omega, mean_anomaly, r, v, rs, gamma1, gamma2, gamma3, a_s)``
and its inverse (old Kepler elements -> PSK coordinates), per
``.note/PSK_conventions.md`` (the binding coordinate-convention document,
sections 1-2 and 4bis-4quater) and ``.note/PSK-LAPS_implementation_plan.md``
sections 3, 4, 7.2, and 15 ("first implementation ticket").

Scope and prohibitions (plan section 7.3; user decisions 2026-07-20 recorded
in ``PSK_conventions.md`` sections 4/4bis/4ter/4quater)
------------------------------------------------------------------------
This module is a **pure coordinate transformation**. It deliberately does
NOT contain:

* **a-bounds.** No ``a_min``/``a_max``/``e_max`` config lives here. The only
  validity this module checks is the transform's own mathematical domain:
  a bound elliptical orbit (``e`` below the Kepler-solver domain limit) and a
  non-singular projection (``g_proj`` bounded away from zero). Scientific
  a-bounds belong to the prior layer (plan P4, ``KeplerPriorBlock``), which
  must set them explicitly -- there is no library default here.
* **Prior density or branch weighting.** ``OrbitalTransformResult.log_prior``
  is always ``0.0`` in this module; it exists only for shape-compatibility
  with the plan section 7.2 interface and is populated downstream (P4).
  Likewise the branch-canonicalization helpers below (``is_canonical``,
  ``to_canonical``, ``reflect_state``) are pure coordinate operators -- they
  do not decide which branch to sample or how to recombine mirrored draws;
  that is the prior layer's responsibility.
* **Year-unit I/O.** Everything here is in day / day^-1. Julian year
  (365.25 day) conversions, where needed at all, live entirely outside this
  module; ``period_years``-style quantities must never feed back into the
  dynamics (``PSK_conventions.md`` section 4quater; see also discrepancy D4
  in that document re: the fixture's inconsistent ``period_years`` metadata).

z-reflection canonical branch (``PSK_conventions.md`` section 4bis)
--------------------------------------------------------------------
The map ``(h, k, lambda) -> (-h, -k, wrap(lambda + pi))`` is an exact,
fixed-point-free involution under which the light curve is invariant (frame
T's ``(rs, gamma3)`` flip sign, ``Omega``/``omega`` shift by pi, everything
else -- including ``cos_i`` -- is unchanged). Because the reflection always
shifts ``lambda`` by exactly pi *regardless of* ``(h, k)``, the sign of
``cos(lambda)`` alone splits the full coordinate space into the two branches
without ever touching the ``(h, k)`` disk (so a canonical-branch prior can
halve the ``lambda`` range and leave the ``(e, omega)`` disk prior
untouched). ``is_canonical``/``to_canonical``/``reflect_state`` implement
this; ``reflect_transform_result`` implements the corresponding physical-side
mirror for reconstructing the dropped branch from a canonical-branch sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gapmoe.orbit._kepler_core import (
    G_AU3_MSUN_DAY2,
    apply_orientation,
    rotate_z,
    solve_kepler,
    wrap_angle,
)

__all__ = [
    "ProjectedKeplerState",
    "OrbitalContext",
    "OrbitalTransformResult",
    "ProjectedKeplerTransform",
    "forward",
    "from_orbital_elements",
    "derived_absolute_position_angle",
    "reflect_state",
    "is_canonical",
    "to_canonical",
    "reflect_transform_result",
]

#: Numerical domain limit of the elliptical Kepler solver. This is a
#: numerical-stability guard on the transform itself, NOT a scientific
#: eccentricity prior cutoff (that is the prior layer's "e_max", which this
#: module never sees).
_E_DOMAIN_MAX = 1.0 - 1.0e-9

#: Below this, the projection is numerically singular (g_proj ~ sin(i)*sin(u)
#: -> 0, so a = s*R_E/Xi blows up). Flags PROJECTION_SINGULARITY-class inputs
#: (plan section 9); it is a transform-domain guard, not a prior.
_G_PROJ_SINGULARITY_EPS = 1.0e-8


@dataclass(frozen=True)
class ProjectedKeplerState:
    """PSK coordinates, at the reference epoch, in frame T.

    ``projected_pa`` (phi_perp) is carried for interface completeness and for
    the optional derived-Omega_NE reporting helper below; the forward
    transform's physical outputs (rs, gamma1..3, a_s) do not depend on it
    (``PSK_conventions.md`` section 3, the "phi_perp is never sampled" gauge
    finding).
    """

    log_s: float
    projected_pa: float
    h: float
    k: float
    cos_i: float
    mean_longitude: float


@dataclass(frozen=True)
class OrbitalContext:
    """Context needed to evaluate the forward transform.

    ``total_mass`` is the lens total mass in Msun (GM = G_AU3_MSUN_DAY2 *
    total_mass is formed internally); ``einstein_radius_physical`` is R_E in
    AU. ``proper_motion_pa`` (phi_mu) and ``reference_epoch`` (t_ref) are
    metadata only: ``forward`` does not use them numerically (frame T's
    (rs, gamma1..3, a_s) depend only on (log_s, h, k, cos_i, mean_longitude,
    total_mass, einstein_radius_physical) by construction); they are carried
    through for the optional ``derived_absolute_position_angle`` helper and
    for run-metadata bookkeeping upstream.
    """

    total_mass: float
    einstein_radius_physical: float
    proper_motion_pa: Optional[float]
    reference_epoch: float


@dataclass(frozen=True)
class OrbitalTransformResult:
    """Forward-transform output: old Kepler elements + frame-T state.

    ``log_prior`` is always ``0.0`` here (see module docstring); ``valid`` is
    ``True`` iff the input is within this module's own transform domain
    (bound orbit, non-singular projection) -- it carries no a-bounds or
    prior-support information.
    """

    a: np.ndarray
    e: np.ndarray
    i: np.ndarray
    Omega: np.ndarray
    omega: np.ndarray
    mean_anomaly: np.ndarray
    rs: np.ndarray
    gamma1: np.ndarray
    gamma2: np.ndarray
    gamma3: np.ndarray
    a_s: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    # Extra diagnostics beyond the plan section 7.2 minimal interface (kept
    # as plain attributes on the same frozen dataclass rather than a second
    # return channel, so callers get them "for free"):
    r: np.ndarray
    v: np.ndarray
    consistency_error: np.ndarray


def _default_consistency_tol(x, xp) -> float:
    """Pick a self-consistency tolerance from the array's own precision.

    The a_s = 1/(1 - e*cos E) = a/|r| identity (plan section 4.3) should hold
    to float64 precision on the numpy path; the JAX path used for P2's
    autodiff Jacobian checks runs in this repo's default float32 (no global
    ``jax_enable_x64``, confirmed absent from this codebase), so it needs a
    looser tolerance for the exact same arithmetic -- matching the tolerance
    policy already established for float32 JAX elsewhere in this branch
    (e.g. ``tests/test_laps_theta_star_explicit.py``).
    """
    dtype = getattr(x, "dtype", None)
    if dtype is not None:
        try:
            eps = float(xp.finfo(dtype).eps)
        except (TypeError, ValueError):
            eps = np.finfo(np.float64).eps
        if eps > 1.0e-10:
            return 1.0e-4
    return 1.0e-8


def forward(
    state: ProjectedKeplerState,
    context: OrbitalContext,
    *,
    xp=np,
    n_iter: int = 40,
    consistency_tol: Optional[float] = None,
    strict: bool = True,
) -> OrbitalTransformResult:
    """PSK -> old Kepler elements -> frame-T state vector.

    Vectorized: fields of ``state``/``context`` may be scalars or
    broadcast-compatible arrays. Pass ``xp=jax.numpy`` for a jit/vmap/grad
    -compatible path (used by P2's analytic-vs-AD Jacobian checks); the
    numpy path is the default and is what the ``PSK_conventions.md`` worked
    example (Roman fixture, 12-digit round trip) was verified against.

    ``strict`` (default ``True``) selects how a genuine self-consistency
    violation (plan section 4.3: ``a_s = 1/(1-e*cos E)`` vs ``a/|r|``) is
    reported. In debug/test usage (the default) this must be a hard failure,
    not a silently-downgraded flag -- so on the eager numpy path a violation
    at an otherwise-``valid`` point raises ``RuntimeError`` as before. That
    behavior is impossible to keep under ``xp=jax.numpy`` (Python-level
    branching on traced values is illegal under jit/vmap/grad) and is
    unfriendly to production numpy batch processing (one bad point would
    abort an entire batch). Passing ``strict=False`` selects the
    production-path design instead: no path ever raises, and points whose
    consistency error exceeds ``tol`` are folded into ``valid=False`` (for
    both the numpy and JAX array modules alike), so callers doing
    vectorized/JAX inference get a uniform invalid-mask contract rather than
    an unhandled violation.
    """
    log_s = xp.asarray(state.log_s)
    h = xp.asarray(state.h)
    k = xp.asarray(state.k)
    cos_i = xp.asarray(state.cos_i)
    mean_longitude = xp.asarray(state.mean_longitude)
    total_mass = xp.asarray(context.total_mass)
    R_E = xp.asarray(context.einstein_radius_physical)

    e = h**2 + k**2
    omega = xp.arctan2(k, h)
    M = wrap_angle(mean_longitude - omega, xp=xp)

    # Elliptical-domain guard applied *before* the Newton solve so that
    # out-of-domain e (>= 1, unbound) never produces a silently wrong E via
    # sqrt of a negative number; clip only for arithmetic safety, `valid`
    # below is what actually flags these points -- the clip must not be
    # mistaken for a prior e_max cutoff (see module docstring).
    e_safe = xp.clip(e, 0.0, _E_DOMAIN_MAX)

    E = solve_kepler(M, e_safe, n_iter=n_iter, xp=xp)
    nu = 2.0 * xp.arctan2(
        xp.sqrt(xp.clip(1.0 + e_safe, 0.0, None)) * xp.sin(E / 2.0),
        xp.sqrt(xp.clip(1.0 - e_safe, 0.0, None)) * xp.cos(E / 2.0),
    )
    u = omega + nu

    sin_i = xp.sqrt(xp.clip(1.0 - cos_i**2, 0.0, 1.0))
    g_proj = xp.sqrt(xp.clip(1.0 - sin_i**2 * xp.sin(u) ** 2, 0.0, 1.0))
    Xi = (1.0 - e_safe * xp.cos(E)) * g_proj

    s = xp.exp(log_s)
    Xi_safe = xp.where(xp.abs(Xi) > _G_PROJ_SINGULARITY_EPS, Xi, 1.0)
    a = s * R_E / Xi_safe

    GM = G_AU3_MSUN_DAY2 * total_mass
    n_mean_motion = xp.sqrt(GM / xp.clip(a, 1.0e-300, None) ** 3)

    # Alignment gauge: Omega0 = -chi identically in frame T (PSK_conventions
    # section 1, C13; verified to 3e-14 deg in the P0 numerical experiments).
    Omega0 = -xp.arctan2(cos_i * xp.sin(u), xp.cos(u))

    sqrt_1me2 = xp.sqrt(xp.clip(1.0 - e_safe**2, 0.0, None))
    r_pf = a[..., None] * xp.stack(
        [xp.cos(E) - e_safe, sqrt_1me2 * xp.sin(E), xp.zeros_like(E)], axis=-1
    )
    speed_factor = a * n_mean_motion / xp.clip(1.0 - e_safe * xp.cos(E), 1.0e-300, None)
    v_pf = speed_factor[..., None] * xp.stack(
        [-xp.sin(E), sqrt_1me2 * xp.cos(E), xp.zeros_like(E)], axis=-1
    )

    r_body = apply_orientation(r_pf, omega, cos_i, xp=xp)
    v_body = apply_orientation(v_pf, omega, cos_i, xp=xp)
    r_sky = rotate_z(r_body, Omega0, xp=xp)
    v_sky = rotate_z(v_body, Omega0, xp=xp)

    scale = s * R_E
    scale_safe = xp.where(xp.abs(scale) > 0.0, scale, 1.0)
    rs = r_sky[..., 2] / scale_safe
    gamma = v_sky / scale_safe[..., None]

    a_s = 1.0 / xp.clip(1.0 - e_safe * xp.cos(E), 1.0e-300, None)

    r_norm = xp.sqrt(xp.sum(r_sky**2, axis=-1))
    r_norm_safe = xp.where(r_norm > 0.0, r_norm, 1.0)
    consistency_error = xp.abs(a_s - a / r_norm_safe)

    valid = (e < _E_DOMAIN_MAX) & (xp.abs(Xi) > _G_PROJ_SINGULARITY_EPS)

    tol = consistency_tol if consistency_tol is not None else _default_consistency_tol(e, xp)

    if strict:
        if xp is np:
            bad = valid & (consistency_error > tol)
            if np.any(bad):
                idx = np.flatnonzero(np.atleast_1d(bad))
                raise RuntimeError(
                    "PSK forward transform: a_s = 1/(1-e*cos E) disagrees with "
                    f"a/|r| beyond tolerance {tol:g} at {idx.size} point(s) "
                    f"(first index {int(idx[0])}, error "
                    f"{float(np.atleast_1d(consistency_error)[idx[0]]):g}). This is "
                    "a self-consistency failure in the transform, not an invalid "
                    "input -- per plan section 4.3 it must raise, not silently "
                    "flag invalid."
                )
        # xp is not np: raising on traced values is impossible; strict mode
        # simply has no enforcement on the JAX path (documented above).
    else:
        valid = valid & (consistency_error <= tol)

    i_angle = xp.arccos(xp.clip(cos_i, -1.0, 1.0))
    zeros = xp.zeros_like(e)

    return OrbitalTransformResult(
        a=a,
        e=e,
        i=i_angle,
        Omega=Omega0,
        omega=omega,
        mean_anomaly=M,
        rs=rs,
        gamma1=gamma[..., 0],
        gamma2=gamma[..., 1],
        gamma3=gamma[..., 2],
        a_s=a_s,
        log_prior=zeros,
        valid=valid,
        r=r_sky,
        v=v_sky,
        consistency_error=consistency_error,
    )


class ProjectedKeplerTransform:
    """Stateless callable matching plan section 7.2's suggested interface.

    Holds no configuration -- in particular no a-bounds/e_max (see module
    docstring) -- and simply delegates to the module-level :func:`forward`.
    """

    def forward(
        self, state: ProjectedKeplerState, context: OrbitalContext, **kwargs
    ) -> OrbitalTransformResult:
        return forward(state, context, **kwargs)


def from_orbital_elements(
    *,
    s,
    e,
    cos_i,
    omega,
    nu,
    projected_pa=0.0,
    xp=np,
) -> ProjectedKeplerState:
    """Inverse direction: old Kepler elements -> PSK coordinates.

    Implements ``PSK_conventions.md`` section 2.1. Only ``(s, e, cos_i,
    omega, nu)`` determine ``(log_s, h, k, cos_i, mean_longitude)`` --
    ``projected_pa`` (phi_perp) is independent of the element inversion (it
    is a separate, optionally-derived quantity; see
    ``derived_absolute_position_angle`` below) and defaults to ``0.0`` (the
    no-parallax gauge fix, ``PSK_conventions.md`` section 3).
    """
    s = xp.asarray(s)
    e = xp.asarray(e)
    omega = xp.asarray(omega)
    nu = xp.asarray(nu)

    E = 2.0 * xp.arctan2(
        xp.sqrt(xp.clip(1.0 - e, 0.0, None)) * xp.sin(nu / 2.0),
        xp.sqrt(xp.clip(1.0 + e, 0.0, None)) * xp.cos(nu / 2.0),
    )
    M = E - e * xp.sin(E)
    h = xp.sqrt(xp.clip(e, 0.0, None)) * xp.cos(omega)
    k = xp.sqrt(xp.clip(e, 0.0, None)) * xp.sin(omega)
    lam = wrap_angle(M + omega, xp=xp)
    log_s = xp.log(s)

    return ProjectedKeplerState(
        log_s=log_s,
        projected_pa=projected_pa,
        h=h,
        k=k,
        cos_i=xp.asarray(cos_i),
        mean_longitude=lam,
    )


def derived_absolute_position_angle(
    state: ProjectedKeplerState, result: OrbitalTransformResult, *, xp=np
):
    """Omega_NE = phi_perp + Omega0 (PSK_conventions.md section 2.2/6).

    Purely a reporting identity (no prior, no context beyond what the two
    dataclasses already carry): equivalent to ``phi_mu - alpha - chi`` since
    ``Omega0 = -chi``. Only meaningful when ``state.projected_pa`` is an
    actual derived phi_perp (parallax case); in the no-parallax gauge
    (phi_perp := 0) this returns ``Omega0`` itself and the NE label has no
    physical meaning (Omega_NE is undefined without parallax, per
    ``PSK_conventions.md`` section 3).
    """
    return wrap_angle(
        xp.asarray(state.projected_pa) + result.Omega, xp=xp
    )


# ---------------------------------------------------------------------------
# z-reflection canonical branch (PSK_conventions.md section 4bis).
# ---------------------------------------------------------------------------


def reflect_state(state: ProjectedKeplerState, *, xp=np) -> ProjectedKeplerState:
    """The (h, k, lambda) -> (-h, -k, wrap(lambda + pi)) involution.

    ``log_s``, ``projected_pa``, and ``cos_i`` are unchanged. Applying this
    twice returns the original state (up to angle wrapping); it has no fixed
    points. See module docstring for the physical-side counterpart
    (:func:`reflect_transform_result`) and the branch-selection functions
    below.
    """
    return ProjectedKeplerState(
        log_s=state.log_s,
        projected_pa=state.projected_pa,
        h=-xp.asarray(state.h),
        k=-xp.asarray(state.k),
        cos_i=state.cos_i,
        mean_longitude=wrap_angle(xp.asarray(state.mean_longitude) + np.pi, xp=xp),
    )


def is_canonical(mean_longitude, *, xp=np):
    """canonical iff cos(lambda) >= 0, i.e. lambda in [-pi/2, pi/2].

    This split is independent of ``(h, k)`` because the reflection always
    shifts ``lambda`` by exactly pi regardless of ``(h, k)`` -- so a
    canonical-branch prior restriction only ever halves the ``lambda``
    range and never touches the ``(e, omega)`` disk (``PSK_conventions.md``
    section 4bis). The boundary ``cos(lambda) == 0`` is measure zero and is
    assigned to the canonical branch by the ``>=``.
    """
    return xp.cos(xp.asarray(mean_longitude)) >= 0.0


def to_canonical(state: ProjectedKeplerState, *, xp=np):
    """Map ``state`` to its canonical-branch representative.

    Returns ``(canonical_state, flipped)`` where ``flipped`` is a boolean
    (array) recording whether :func:`reflect_state` was applied. This is a
    pure coordinate map: it does not decide *whether* the caller should
    restrict sampling to the canonical branch, nor how to weight/recombine
    the two branches when reporting a posterior -- that is the prior
    layer's responsibility (P4).
    """
    canonical = is_canonical(state.mean_longitude, xp=xp)
    flipped = ~canonical
    reflected = reflect_state(state, xp=xp)

    def _select(a, b):
        a = xp.asarray(a)
        b = xp.asarray(b)
        return xp.where(canonical, a, b)

    out = ProjectedKeplerState(
        log_s=state.log_s,
        projected_pa=state.projected_pa,
        h=_select(state.h, reflected.h),
        k=_select(state.k, reflected.k),
        cos_i=state.cos_i,
        mean_longitude=_select(state.mean_longitude, reflected.mean_longitude),
    )
    return out, flipped


def reflect_transform_result(
    result: OrbitalTransformResult, *, xp=np
) -> OrbitalTransformResult:
    """Physical-side mirror of :func:`reflect_state`.

    ``rs`` and ``gamma3`` flip sign; ``Omega``/``omega`` shift by pi (wrap);
    everything else (a, e, i, mean_anomaly, gamma1, gamma2, a_s, r, v,
    log_prior, valid, consistency_error) is unchanged -- this is exactly the
    identity confirmed numerically in the P0 experiments (E6d/E7: the two
    branches produce identical light curves). Use this to reconstruct the
    dropped mirror branch when a caller has sampled the canonical branch
    only (see module docstring).
    """
    pi = np.pi
    return OrbitalTransformResult(
        a=result.a,
        e=result.e,
        i=result.i,
        Omega=wrap_angle(result.Omega + pi, xp=xp),
        omega=wrap_angle(result.omega + pi, xp=xp),
        mean_anomaly=result.mean_anomaly,
        rs=-result.rs,
        gamma1=result.gamma1,
        gamma2=result.gamma2,
        gamma3=-result.gamma3,
        a_s=result.a_s,
        log_prior=result.log_prior,
        valid=result.valid,
        r=result.r * xp.asarray([1.0, 1.0, -1.0]),
        v=result.v * xp.asarray([1.0, 1.0, -1.0]),
        consistency_error=result.consistency_error,
    )
