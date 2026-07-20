"""Private conditional orbit-completion proposal sampler (LAPS-SMC H3).

.. warning::
   **Private, proposal-construction-only API.** This module is intentionally
   *not* exported from ``gapmoe``'s public ``__init__``. It exists so that an
   external sampler (moasarc's LAPS-SMC) can build a *reference/proposal*
   distribution for the orbital degrees of freedom of
   :class:`~gapmoe.param_types.binary_lens.BinaryKeplerParamType` (and the
   circular variant). Nothing here participates in target-density evaluation:
   no prior term, no Jacobian, no ``Model`` code path is touched.

Method (guardrails section 4.4, "Method A")
-------------------------------------------
This sampler returns **samples and diagnostics only** — it deliberately does
*not* compute or return a proposal log-density. The caller is expected to fit
a density model (e.g. a Student-t mixture) to ``OrbitProposal.draws`` in the
*sampling coordinates* and use that fit's own ``log_prob`` as the reference
density. Because no element-space density or transformation Jacobian is ever
reported from here, Jacobian double-counting between element coordinates and
sampling coordinates is impossible by construction.

How draws are generated
-----------------------
The Kepler light-curve coordinates ``(g1, g2, g3, lom_szs, lom_ar)`` consumed
by ``BinaryKeplerParamType`` are, physically,

* ``(g1, g2, g3)``: the companion's velocity vector divided by the projected
  separation ``RE*s`` (units: 1/day),
* ``lom_szs`` (= ``r_s``): line-of-sight separation / projected separation,
* ``lom_ar`` (= ``a_s``): semi-major axis / instantaneous 3-D separation,

in a frame whose x-axis lies along the instantaneous projected binary axis
and whose z-axis is the line of sight. gapmoe's forward map
``_lc_to_phys_kepler`` *defines* the physical scale ``RE = DL*thE`` such that
the vis-viva relation holds identically, so a candidate is a valid bound
orbit iff ``a_s > 1/2``, ``gamma^2 > 0`` and the implied ``RE < 1/piE``
(equivalently ``DS > DL > 0``).

We therefore simulate *physically*, never drawing a naive box in ``g``
(guardrails section 4.1): draw Keplerian elements from a documented element
prior conditioned on the fixed observables, construct the position/velocity
vectors, align the projected separation with the +x axis (this is exact
conditional sampling because the prior over the node angle is uniform), scale
to the drawn lens distance, and push into the ``(g, lom)`` coordinates. Every
candidate is then spliced into a full ``theta`` with the caller's fixed
values and accepted only if gapmoe's own ``_valid_kepler_out`` predicate —
the *same* validity definition used by ``log_abs_det_jacobian`` on the target
side — is satisfied. Fixed (pinned) orbit components supplied by the caller
are spliced in *before* the validity check, so they are respected bitwise and
participate in rejection.

Default element prior (all overridable via ``element_prior``)
-------------------------------------------------------------
* eccentricity ``e``            ~ Uniform(0, 0.95)          (Kepler only)
* orbit-normal ``cos i``        ~ Uniform(-1, 1)            (isotropic)
* argument of pericenter ``om`` ~ Uniform(-pi, pi)          (Kepler only)
* mean anomaly / phase          ~ Uniform(-pi, pi)          (time-uniform)
* lens distance ``DL``          ~ log-uniform on
  ``(0.02, 0.98) * DL_max`` with ``DL_max = 1/(thE*piE)``, the entire
  physically allowed range given the fixed observables (``DS > DL > 0``).
  The implied semi-major axis is ``a = RE*s*sqrt(1+r_s^2)*a_s`` — i.e. the
  scale is drawn *conditioned on s*, as required by the H3 plan.

The node angle is not drawn: it is fixed by the frame alignment, which for a
uniform node prior leaves the joint law of the remaining elements unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from gapmoe.orbit._kepler_core import (
    apply_orientation as _apply_orientation,
    solve_kepler as _solve_kepler,
)
from gapmoe.param_types.binary_lens import (
    _G,
    _KAPPA,
    _lc_to_phys_circular,
    _lc_to_phys_kepler,
    _valid_circular_out,
    _valid_kepler_out,
    BinaryCircularParamType,
    BinaryKeplerParamType,
)

__all__ = [
    "OrbitProposal",
    "sample_orbit_completion",
    "sample_orbit_completion_circular",
]

_KEPLER_THETA_NAMES: Tuple[str, ...] = BinaryKeplerParamType.names
_CIRCULAR_THETA_NAMES: Tuple[str, ...] = BinaryCircularParamType.names

_KEPLER_ORBIT_NAMES: Tuple[str, ...] = ("g1", "g2", "g3", "lom_szs", "lom_ar")
_CIRCULAR_ORBIT_NAMES: Tuple[str, ...] = ("g1", "g2", "g3")

_BASE_FIXED_NAMES: Tuple[str, ...] = (
    "t0", "tE", "u0", "rho", "q", "s", "alpha", "piEN", "piEE",
)

_KEPLER_DERIVED_COLS = {
    "orbital_radi": 8, "e": 9, "cos_i": 10, "Om_NE": 11, "om": 12, "nu": 13,
}
_CIRCULAR_DERIVED_COLS = {
    "orbital_radi": 8, "cos_i": 9, "Om_NE": 10, "phi0": 11,
}


@dataclass(frozen=True)
class OrbitProposal:
    """Unweighted proposal draws for the non-fixed orbit DOF.

    Proposal-construction-only (guardrails 4.4 Method A): carries **samples
    and diagnostics, never a proposal log-density**. Fit your density model
    to ``draws`` in these sampling coordinates and use that fit's log_prob.
    """

    draws: np.ndarray                    # (n_accepted_returned, k) float64
    columns: Tuple[str, ...]             # names of the k non-fixed orbit DOF
    conditioned_on: Dict[str, float]     # exactly the caller's `fixed` values
    acceptance: float                    # n_accepted / n_proposed
    n_accepted: int                      # total accepted within the budget
    n_proposed: int                      # total candidates simulated
    n_requested: int                     # the `n` the caller asked for
    variant: str                         # "kepler" or "circular"
    theta_names: Tuple[str, ...]         # full theta layout for splicing
    derived: Optional[Dict[str, np.ndarray]] = None
    prior_spec: Dict[str, Any] = field(default_factory=dict)

    def spliced_thetas(self) -> np.ndarray:
        """Return (n, len(theta_names)) thetas: fixed values + draws.

        Fixed components are inserted bitwise from ``conditioned_on``.
        """
        m = self.draws.shape[0]
        thetas = np.empty((m, len(self.theta_names)), dtype=np.float64)
        for j, name in enumerate(self.theta_names):
            if name in self.conditioned_on:
                thetas[:, j] = self.conditioned_on[name]
            else:
                thetas[:, j] = self.draws[:, self.columns.index(name)]
        return thetas


def _as_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _check_fixed(fixed: Mapping[str, float], orbit_names: Tuple[str, ...],
                 variant: str) -> Tuple[str, ...]:
    missing = [k for k in _BASE_FIXED_NAMES if k not in fixed]
    if missing:
        raise ValueError(
            f"fixed must provide all non-orbit theta components; missing {missing}"
        )
    allowed = set(_BASE_FIXED_NAMES) | set(orbit_names)
    unknown = [k for k in fixed if k not in allowed]
    if unknown:
        raise ValueError(
            f"unknown fixed keys {unknown} for the {variant} variant"
        )
    free = tuple(nm for nm in orbit_names if nm not in fixed)
    if not free:
        raise ValueError(
            "all orbit components are fixed; nothing to sample"
        )
    return free


def _context_scalars(fixed: Mapping[str, float], context: Mapping[str, Any]):
    if context is None or "thS" not in context or "vEarth" not in context:
        raise ValueError(
            "context must include 'thS' (mas) and 'vEarth' ((v_N, v_E) AU/yr), "
            "exactly as required by the Kepler param_types."
        )
    thS = float(context["thS"])
    vEarth = tuple(float(v) for v in context["vEarth"])
    thE = thS / float(fixed["rho"])
    piE = float(np.hypot(fixed["piEN"], fixed["piEE"]))
    if not (piE > 0.0):
        raise ValueError("piE = hypot(piEN, piEE) must be positive")
    ML = thE / _KAPPA / piE
    GM = _G * ML
    DL_max = 1.0 / (thE * piE)      # kpc; DS > DL > 0 requires DL < DL_max
    return thS, vEarth, thE, piE, ML, GM, DL_max


def _rot_and_align(r1: np.ndarray, v1: np.ndarray):
    """Rotate about the line of sight so the projected separation is +x.

    r1, v1: (m, 3). Returns (r2, v2) with r2[:,1] == 0 and r2[:,0] >= 0.
    """
    phi = np.arctan2(r1[:, 1], r1[:, 0])
    c, s = np.cos(phi), np.sin(phi)
    r2 = np.empty_like(r1)
    v2 = np.empty_like(v1)
    r2[:, 0] = c * r1[:, 0] + s * r1[:, 1]
    r2[:, 1] = 0.0
    r2[:, 2] = r1[:, 2]
    v2[:, 0] = c * v1[:, 0] + s * v1[:, 1]
    v2[:, 1] = -s * v1[:, 0] + c * v1[:, 1]
    v2[:, 2] = v1[:, 2]
    return r2, v2


_DEFAULT_DL_FRAC_RANGE = (0.02, 0.98)


def _default_priors_kepler() -> Dict[str, Callable]:
    lo, hi = np.log(_DEFAULT_DL_FRAC_RANGE[0]), np.log(_DEFAULT_DL_FRAC_RANGE[1])
    return {
        "e": lambda rng, m: rng.uniform(0.0, 0.95, m),
        "cos_i": lambda rng, m: rng.uniform(-1.0, 1.0, m),
        "omega": lambda rng, m: rng.uniform(-np.pi, np.pi, m),
        "mean_anomaly": lambda rng, m: rng.uniform(-np.pi, np.pi, m),
        "dl_frac": lambda rng, m: np.exp(rng.uniform(lo, hi, m)),
    }


def _default_priors_circular() -> Dict[str, Callable]:
    lo, hi = np.log(_DEFAULT_DL_FRAC_RANGE[0]), np.log(_DEFAULT_DL_FRAC_RANGE[1])
    return {
        "cos_i": lambda rng, m: rng.uniform(-1.0, 1.0, m),
        "phase": lambda rng, m: rng.uniform(-np.pi, np.pi, m),
        "dl_frac": lambda rng, m: np.exp(rng.uniform(lo, hi, m)),
    }


def _merge_priors(defaults: Dict[str, Callable],
                  element_prior: Optional[Mapping[str, Callable]]):
    priors = dict(defaults)
    if element_prior is not None:
        unknown = [k for k in element_prior if k not in defaults]
        if unknown:
            raise ValueError(
                f"element_prior keys {unknown} not recognized; allowed: "
                f"{sorted(defaults)}"
            )
        priors.update(element_prior)
    return priors


def _candidates_kepler(rng, m, priors, fixed, GM, thE, DL_max):
    """Forward-simulate m candidate (g1, g2, g3, lom_szs, lom_ar) rows."""
    s = float(fixed["s"])
    e = np.asarray(priors["e"](rng, m), dtype=np.float64)
    cos_i = np.asarray(priors["cos_i"](rng, m), dtype=np.float64)
    om = np.asarray(priors["omega"](rng, m), dtype=np.float64)
    M_anom = np.asarray(priors["mean_anomaly"](rng, m), dtype=np.float64)
    dl_frac = np.asarray(priors["dl_frac"](rng, m), dtype=np.float64)

    E = _solve_kepler(M_anom, e)
    cE, sE = np.cos(E), np.sin(E)
    b_over_a = np.sqrt(np.clip(1.0 - e**2, 0.0, None))
    r_unit_mag = 1.0 - e * cE                      # |r| in units of a

    # Perifocal position and velocity for a = 1, GM = 1.
    r_pf = np.stack([cE - e, b_over_a * sE, np.zeros(m)], axis=1)
    v_pf = np.stack([-sE, b_over_a * cE, np.zeros(m)], axis=1) / r_unit_mag[:, None]

    r1 = _apply_orientation(r_pf, om, cos_i)
    v1 = _apply_orientation(v_pf, om, cos_i)
    r2, v2 = _rot_and_align(r1, v1)

    r_perp_unit = r2[:, 0]                         # projected sep, units of a
    with np.errstate(divide="ignore", invalid="ignore"):
        r_s = r2[:, 2] / r_perp_unit
        a_s = 1.0 / r_unit_mag                     # a / |r|
        DL = dl_frac * DL_max                      # kpc
        RE = DL * thE                              # AU
        a_phys = RE * s / r_perp_unit              # AU
        v_scale = np.sqrt(GM / a_phys)             # AU / day
        g = v_scale[:, None] * v2 / (RE * s)[:, None]   # 1 / day

    cand = np.column_stack([g[:, 0], g[:, 1], g[:, 2], r_s, a_s])
    return cand


def _candidates_circular(rng, m, priors, fixed, GM, thE, DL_max):
    """Forward-simulate m candidate (g1, g2, g3) rows for circular orbits."""
    s = float(fixed["s"])
    cos_i = np.asarray(priors["cos_i"](rng, m), dtype=np.float64)
    phase = np.asarray(priors["phase"](rng, m), dtype=np.float64)
    dl_frac = np.asarray(priors["dl_frac"](rng, m), dtype=np.float64)

    cp, sp = np.cos(phase), np.sin(phase)
    r_pf = np.stack([cp, sp, np.zeros(m)], axis=1)          # |r| = a = 1
    v_pf = np.stack([-sp, cp, np.zeros(m)], axis=1)         # |v| = 1
    zeros = np.zeros(m)
    r1 = _apply_orientation(r_pf, zeros, cos_i)
    v1 = _apply_orientation(v_pf, zeros, cos_i)
    r2, v2 = _rot_and_align(r1, v1)

    r_perp_unit = r2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        DL = dl_frac * DL_max
        RE = DL * thE
        a_phys = RE * s / r_perp_unit
        v_scale = np.sqrt(GM / a_phys)
        g = v_scale[:, None] * v2 / (RE * s)[:, None]
    return g


def _run_rejection(*, fixed, context, n, element_prior, rng, max_tries,
                   variant, with_derived=True):
    if variant == "kepler":
        theta_names = _KEPLER_THETA_NAMES
        orbit_names = _KEPLER_ORBIT_NAMES
        defaults = _default_priors_kepler()
        candidate_fn = _candidates_kepler
        kernel = _lc_to_phys_kepler
        valid_fn = _valid_kepler_out
        derived_cols = _KEPLER_DERIVED_COLS
    elif variant == "circular":
        theta_names = _CIRCULAR_THETA_NAMES
        orbit_names = _CIRCULAR_ORBIT_NAMES
        defaults = _default_priors_circular()
        candidate_fn = _candidates_circular
        kernel = _lc_to_phys_circular
        valid_fn = _valid_circular_out
        derived_cols = _CIRCULAR_DERIVED_COLS
    else:  # pragma: no cover
        raise ValueError(f"unknown variant {variant!r}")

    if n <= 0:
        raise ValueError("n must be a positive integer")
    free_cols = _check_fixed(fixed, orbit_names, variant)
    thS, vEarth, thE, piE, ML, GM, DL_max = _context_scalars(fixed, context)
    rng = _as_rng(rng)
    priors = _merge_priors(defaults, element_prior)
    if max_tries is None:
        max_tries = max(200 * n, 10_000)

    name_to_idx = {nm: j for j, nm in enumerate(theta_names)}
    free_idx = np.array([name_to_idx[nm] for nm in free_cols], dtype=int)
    orbit_idx = {nm: j for j, nm in enumerate(orbit_names)}

    theta_template = np.zeros(len(theta_names), dtype=np.float64)
    for nm, val in fixed.items():
        theta_template[name_to_idx[nm]] = float(val)

    batch_kernel = jax.jit(jax.vmap(kernel, in_axes=(0, None, None)))
    batch_valid = jax.jit(jax.vmap(valid_fn))

    accepted_rows = []
    accepted_outs = []
    n_accepted = 0
    n_proposed = 0
    while n_accepted < n and n_proposed < max_tries:
        m = int(min(max(2 * n, 1024), max_tries - n_proposed))
        cand = candidate_fn(rng, m, priors, fixed, GM, thE, DL_max)
        n_proposed += m

        thetas = np.tile(theta_template, (m, 1))
        for nm in free_cols:
            thetas[:, name_to_idx[nm]] = cand[:, orbit_idx[nm]]
        # Pinned orbit components stay at their fixed values from the
        # template and participate in the validity check below.

        finite = np.all(np.isfinite(thetas[:, free_idx]), axis=1)
        outs = np.asarray(batch_kernel(jnp.asarray(thetas), thS, vEarth))
        valid = np.asarray(batch_valid(jnp.asarray(outs))) & finite

        if np.any(valid):
            accepted_rows.append(thetas[valid][:, free_idx])
            accepted_outs.append(outs[valid])
            n_accepted += int(valid.sum())

    if n_accepted == 0:
        raise RuntimeError(
            f"orbit-completion rejection sampler accepted 0 of {n_proposed} "
            f"candidates (max_tries={max_tries}); the fixed values "
            "(e.g. pinned g components) are likely inconsistent with any "
            "bound orbit. Loosen the pins or the element prior."
        )

    draws = np.concatenate(accepted_rows, axis=0)
    outs = np.concatenate(accepted_outs, axis=0)
    keep = min(n, draws.shape[0])
    draws = draws[:keep]
    outs = outs[:keep]

    derived = None
    if with_derived:
        derived = {nm: outs[:, col].copy() for nm, col in derived_cols.items()}

    prior_spec = {
        "thE_mas": thE,
        "piE": piE,
        "ML_Msun": ML,
        "DL_max_kpc": DL_max,
        "dl_frac_range": _DEFAULT_DL_FRAC_RANGE,
        "overridden": sorted(element_prior) if element_prior else [],
        "defaults": {
            "kepler": "e~U(0,0.95), cos_i~U(-1,1), omega~U(-pi,pi), "
                      "mean_anomaly~U(-pi,pi), DL~logU(0.02,0.98)*DL_max",
            "circular": "cos_i~U(-1,1), phase~U(-pi,pi), "
                        "DL~logU(0.02,0.98)*DL_max",
        }[variant],
    }

    return OrbitProposal(
        draws=draws,
        columns=free_cols,
        conditioned_on={k: float(v) for k, v in fixed.items()},
        acceptance=n_accepted / n_proposed,
        n_accepted=n_accepted,
        n_proposed=n_proposed,
        n_requested=n,
        variant=variant,
        theta_names=theta_names,
        derived=derived,
        prior_spec=prior_spec,
    )


def sample_orbit_completion(
    *,
    fixed: Mapping[str, float],
    context: Mapping[str, Any],
    n: int,
    element_prior: Optional[Mapping[str, Callable]] = None,
    rng=None,
    max_tries: Optional[int] = None,
) -> OrbitProposal:
    """Draw proposal completions of the Kepler orbit DOF conditioned on `fixed`.

    Proposal-construction-only (guardrails 4.4 Method A): returns **samples
    and diagnostics only, never a proposal log-density** — fit a density model
    to ``draws`` in sampling coordinates and use that fit's log_prob as the
    reference density. This prevents Jacobian double-counting by construction.

    Parameters
    ----------
    fixed
        Conditioning values. Must contain all nine non-orbit components of
        ``BinaryKeplerParamType.names`` (``t0, tE, u0, rho, q, s, alpha,
        piEN, piEE``) and may additionally pin any subset of the orbit DOF
        ``(g1, g2, g3, lom_szs, lom_ar)`` (e.g. detected components). Pinned
        values are respected bitwise and enter the validity rejection.
    context
        Same context as the Kepler param_type: ``{"thS": mas, "vEarth": (vN, vE)}``.
    n
        Number of accepted draws wanted.
    element_prior
        Optional dict overriding default element-prior draws. Recognized keys:
        ``"e", "cos_i", "omega", "mean_anomaly", "dl_frac"``; each maps to a
        callable ``(rng, size) -> ndarray``. ``dl_frac`` is DL as a fraction
        of ``DL_max = 1/(thE*piE)``.
    rng
        ``numpy.random.Generator`` or seed.
    max_tries
        Total candidate budget (default ``max(200*n, 10000)``).

    Returns
    -------
    OrbitProposal
        ``draws`` has one column per non-fixed orbit DOF, ordered as in
        ``BinaryKeplerParamType.names``. If fewer than ``n`` candidates are
        accepted within ``max_tries``, a *partial* result is returned
        (``draws.shape[0] < n``; inspect ``acceptance``/``n_accepted``).
        If zero candidates are accepted, ``RuntimeError`` is raised.
    """
    return _run_rejection(
        fixed=fixed, context=context, n=n, element_prior=element_prior,
        rng=rng, max_tries=max_tries, variant="kepler",
    )


def sample_orbit_completion_circular(
    *,
    fixed: Mapping[str, float],
    context: Mapping[str, Any],
    n: int,
    element_prior: Optional[Mapping[str, Callable]] = None,
    rng=None,
    max_tries: Optional[int] = None,
) -> OrbitProposal:
    """Circular-orbit variant of :func:`sample_orbit_completion`.

    Same contract, for ``BinaryCircularParamType``: the orbit DOF are
    ``(g1, g2, g3)`` only and the element prior has no ``e``/``omega``
    (``e = 0``; recognized keys: ``"cos_i", "phase", "dl_frac"``).
    Samples-and-diagnostics only; no proposal log-density (Method A).
    """
    return _run_rejection(
        fixed=fixed, context=context, n=n, element_prior=element_prior,
        rng=rng, max_tries=max_tries, variant="circular",
    )
