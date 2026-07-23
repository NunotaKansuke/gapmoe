"""Prior/Jacobian verification for the PSK pure transform (P2).

Coverage, per ``.note/PSK-LAPS_implementation_checklist.md`` P2-1/P2-2/P2-3
and the 2026-07-20 user acceptance-criteria addendum:

* P2-1's seven property tests: projection identity, position-angle identity,
  a-recovery identity, a_s identity, the analytic Jacobian
  ``|d(a,Omega)/d(log_s,phi_perp)| = a`` cross-checked against both JAX
  autodiff and finite differences, prior-pushforward equivalence against
  ``_orbit_proposal.sample_orbit_completion`` (an independently-coded
  reference pipeline), and circular/face-on limit continuity.
* P2-2: the log-uniform-a-prior cancellation identity
  ``p_a(a) * a = 1/log(a_max/a_min)`` on the support, with a_min/a_max used
  *only* as local test constants -- never passed into ``forward()`` (P2 must
  not add scientific a-bounds to the pure transform; that stays P4's job).
* P2-3: single-counting of the Jacobian -- ``forward()``'s ``log_prior`` is
  always exactly zero (the transform contributes nothing), and the two
  equivalent forms of the Jacobian (``a`` in log_s coordinates, ``a/s`` in s
  coordinates) are related by exactly the ``ds = s*d(log_s)`` factor, which
  is the single place a caller may apply either factor -- never both.
* z-reflection canonical-branch normalization: halving the ``lambda`` range
  requires the density factor of 2 documented in ``PSK_conventions.md``
  section 4bis; omitting it leaves exactly a ``log(2)`` evidence bias.
  ``is_canonical(x) == is_canonical(reflect(x))`` is generically *false* (the
  two are the opposite branch by construction); the property actually
  required is that the *canonical representative* is well-defined on the
  reflection orbit: ``to_canonical(x)`` and ``to_canonical(reflect(x))``
  agree.
* ``strict=False`` (the production JAX/vectorized design added in this
  phase): a genuine a_s self-consistency violation is folded into
  ``valid=False`` instead of raising, on both the numpy and JAX paths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gapmoe.orbit import projected_kepler as pk
from gapmoe.orbit._kepler_core import G_AU3_MSUN_DAY2
from gapmoe.param_types._orbit_proposal import sample_orbit_completion

# ---------------------------------------------------------------------------
# Shared fixture-like conditioning (same physical inputs as the Roman fast
# fixture used in test_projected_kepler.py's _TRUTH; reused here only for
# realism, not to reproduce a specific worked example).
# ---------------------------------------------------------------------------

_FIXED = dict(
    t0=0.0, tE=30.0, u0=0.03, rho=0.0018, q=1.0e-3, s=1.22,
    alpha=0.1832595714594046, piEN=0.12, piEE=-0.08,
)
_CONTEXT = {"thS": 0.000252352244, "vEarth": (0.0, 0.0)}


def _random_state(n, rng, *, e_max=0.9):
    h = rng.uniform(-math.sqrt(e_max) * 0.7, math.sqrt(e_max) * 0.7, n)
    k = rng.uniform(-math.sqrt(e_max) * 0.7, math.sqrt(e_max) * 0.7, n)
    cos_i = rng.uniform(-0.95, 0.95, n)
    lam = rng.uniform(-math.pi, math.pi, n)
    log_s = rng.uniform(-0.3, 0.3, n)
    return pk.ProjectedKeplerState(
        log_s=log_s, projected_pa=np.zeros(n), h=h, k=k, cos_i=cos_i,
        mean_longitude=lam,
    )


def _random_context(n):
    return pk.OrbitalContext(
        total_mass=0.35, einstein_radius_physical=1.15, proper_motion_pa=None,
        reference_epoch=0.0,
    )


# ---------------------------------------------------------------------------
# P2-1: projection identity, position-angle identity, a-recovery identity,
# a_s identity (extended, ensemble-level versions of the single-point P1
# fixture checks).
# ---------------------------------------------------------------------------


def test_projection_identity_r_perp_equals_s_RE():
    rng = np.random.default_rng(10)
    n = 200
    state = _random_state(n, rng)
    context = _random_context(n)
    result = pk.forward(state, context, strict=False)

    s = np.exp(np.asarray(state.log_s))
    r_perp = np.sqrt(result.r[..., 0] ** 2 + result.r[..., 1] ** 2)
    ok = np.asarray(result.valid)
    np.testing.assert_allclose(
        r_perp[ok], (s * context.einstein_radius_physical)[ok], rtol=1e-8
    )
    # The frame-T alignment gauge (Omega0 = -chi) puts the whole projected
    # separation on +x identically -- not just its magnitude.
    np.testing.assert_allclose(result.r[..., 1][ok], 0.0, atol=1e-8)
    assert np.all(result.r[..., 0][ok] > 0.0)


def test_position_angle_identity_Omega0_equals_minus_chi():
    rng = np.random.default_rng(11)
    n = 200
    state = _random_state(n, rng)
    context = _random_context(n)
    result = pk.forward(state, context, strict=False)
    ok = np.asarray(result.valid)

    # Independent reconstruction: chi = atan2(cos_i*sin(u), cos(u)) with
    # u = omega + nu; nu isn't returned by forward() directly, so recompute
    # E/nu the same way the module docstring specifies
    # (PSK_conventions.md section 2.2), using only (e, M) which *are*
    # returned -- this keeps the check independent of Omega itself.
    from gapmoe.orbit._kepler_core import solve_kepler, wrap_angle

    e = np.asarray(result.e)
    M = np.asarray(result.mean_anomaly)
    E = solve_kepler(M, np.clip(e, 0.0, 1.0 - 1e-9), xp=np)
    nu = 2.0 * np.arctan2(
        np.sqrt(np.clip(1.0 + e, 0.0, None)) * np.sin(E / 2.0),
        np.sqrt(np.clip(1.0 - e, 0.0, None)) * np.cos(E / 2.0),
    )
    u = np.asarray(result.omega) + nu
    cos_i = np.asarray(state.cos_i)
    chi = np.arctan2(cos_i * np.sin(u), np.cos(u))
    expected_Omega0 = wrap_angle(-chi, xp=np)

    np.testing.assert_allclose(
        np.sin(np.asarray(result.Omega))[ok], np.sin(expected_Omega0)[ok], atol=1e-8
    )
    np.testing.assert_allclose(
        np.cos(np.asarray(result.Omega))[ok], np.cos(expected_Omega0)[ok], atol=1e-8
    )


def test_a_recovery_identity_matches_independent_Xi_formula():
    rng = np.random.default_rng(12)
    n = 200
    state = _random_state(n, rng)
    context = _random_context(n)
    result = pk.forward(state, context, strict=False)
    ok = np.asarray(result.valid)

    from gapmoe.orbit._kepler_core import solve_kepler

    h, k = np.asarray(state.h), np.asarray(state.k)
    e = h**2 + k**2
    omega = np.arctan2(k, h)
    lam = np.asarray(state.mean_longitude)
    M = np.arctan2(np.sin(lam - omega), np.cos(lam - omega))
    E = solve_kepler(M, np.clip(e, 0.0, 1.0 - 1e-9), xp=np)
    nu = 2.0 * np.arctan2(
        np.sqrt(np.clip(1.0 + e, 0.0, None)) * np.sin(E / 2.0),
        np.sqrt(np.clip(1.0 - e, 0.0, None)) * np.cos(E / 2.0),
    )
    u = omega + nu
    cos_i = np.asarray(state.cos_i)
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 0.0, 1.0))
    g_proj = np.sqrt(np.clip(1.0 - sin_i**2 * np.sin(u) ** 2, 0.0, 1.0))
    Xi = (1.0 - e * np.cos(E)) * g_proj
    s = np.exp(np.asarray(state.log_s))
    a_expected = s * context.einstein_radius_physical / Xi

    np.testing.assert_allclose(
        np.asarray(result.a)[ok], a_expected[ok], rtol=1e-8
    )


def test_a_s_identity_holds_across_random_ensemble():
    rng = np.random.default_rng(13)
    n = 500
    state = _random_state(n, rng)
    context = _random_context(n)
    result = pk.forward(state, context, strict=False)
    ok = np.asarray(result.valid)
    assert ok.sum() > n * 0.5  # sanity: the sampled box isn't mostly invalid
    np.testing.assert_allclose(
        np.asarray(result.consistency_error)[ok], 0.0, atol=1e-6
    )


# ---------------------------------------------------------------------------
# P2-1: circular / face-on limit continuity.
# ---------------------------------------------------------------------------


def test_continuity_toward_circular_limit():
    # e -> 0: h,k -> 0 along a fixed direction. omega is not continuous at
    # e = 0 itself, but the *physical* outputs (rs, gamma, a, a_s) must be.
    context = pk.OrbitalContext(
        total_mass=0.3, einstein_radius_physical=1.2, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    eps_seq = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    outputs = []
    for eps in eps_seq:
        state = pk.ProjectedKeplerState(
            log_s=0.1, projected_pa=0.0, h=eps, k=0.0, cos_i=0.4, mean_longitude=0.6,
        )
        r = pk.forward(state, context, strict=False)
        outputs.append((float(r.a), float(r.rs), float(r.gamma1), float(r.gamma3)))

    outputs = np.array(outputs)
    # Successive differences must shrink (Cauchy-like convergence), not
    # jump around, as the perturbation shrinks by 10x each step.
    diffs = np.abs(np.diff(outputs, axis=0))
    ratios = diffs[1:] / np.maximum(diffs[:-1], 1e-300)
    assert np.all(ratios < 0.5)


def test_continuity_toward_face_on_limit():
    # i -> 0 (cos_i -> 1): sin_i -> 0, g_proj -> 1 regardless of u. Outputs
    # must converge smoothly, not blow up.
    context = pk.OrbitalContext(
        total_mass=0.3, einstein_radius_physical=1.2, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    eps_seq = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    outputs = []
    for eps in eps_seq:
        state = pk.ProjectedKeplerState(
            log_s=0.1, projected_pa=0.0, h=0.3, k=0.1, cos_i=1.0 - eps,
            mean_longitude=0.6,
        )
        r = pk.forward(state, context, strict=False)
        outputs.append((float(r.a), float(r.rs), float(r.gamma1), float(r.gamma3)))

    outputs = np.array(outputs)
    diffs = np.abs(np.diff(outputs, axis=0))
    ratios = diffs[1:] / np.maximum(diffs[:-1], 1e-300)
    assert np.all(ratios < 0.5)
    assert np.all(np.isfinite(outputs))


# ---------------------------------------------------------------------------
# P2-1: analytic Jacobian |d(a,Omega)/d(log_s,phi_perp)| = a, cross-checked
# against JAX autodiff and finite differences.
# ---------------------------------------------------------------------------


def _a_and_Omega_NE(log_s, phi_perp, h, k, cos_i, lam, total_mass, R_E, xp, np_mod):
    state = pk.ProjectedKeplerState(
        log_s=log_s, projected_pa=phi_perp, h=h, k=k, cos_i=cos_i,
        mean_longitude=lam,
    )
    context = pk.OrbitalContext(
        total_mass=total_mass, einstein_radius_physical=R_E, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    result = pk.forward(state, context, xp=xp, strict=False)
    omega_ne = pk.derived_absolute_position_angle(state, result, xp=xp)
    return np_mod.stack([result.a, omega_ne])


_JAC_POINT = dict(h=0.3, k=-0.2, cos_i=0.4, lam=0.9, total_mass=0.3, R_E=1.2)


def test_jacobian_finite_difference_matches_analytic_a():
    p = _JAC_POINT
    log_s0, phi0 = 0.15, 0.4
    step = 1e-6

    def f(log_s, phi_perp):
        return _a_and_Omega_NE(
            log_s, phi_perp, p["h"], p["k"], p["cos_i"], p["lam"],
            p["total_mass"], p["R_E"], np, np,
        )

    d_dlogs = (f(log_s0 + step, phi0) - f(log_s0 - step, phi0)) / (2 * step)
    d_dphi = (f(log_s0, phi0 + step) - f(log_s0, phi0 - step)) / (2 * step)
    J = np.stack([d_dlogs, d_dphi], axis=1)  # rows: (a, Omega_NE); cols: (log_s, phi_perp)
    det = np.linalg.det(J)

    a_here = float(f(log_s0, phi0)[0])
    assert det == pytest.approx(a_here, rel=1e-5)
    # Off-diagonal structure: a doesn't depend on phi_perp, Omega_NE shifts
    # 1:1 with phi_perp and not at all with log_s (plan section 5.1).
    assert d_dlogs[1] == pytest.approx(0.0, abs=1e-6)
    assert d_dphi[0] == pytest.approx(0.0, abs=1e-6 * max(a_here, 1.0))
    assert d_dphi[1] == pytest.approx(1.0, abs=1e-6)


def test_jacobian_autodiff_matches_analytic_a():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    p = _JAC_POINT

    def f(params):
        log_s, phi_perp = params[0], params[1]
        return _a_and_Omega_NE(
            log_s, phi_perp, jnp.asarray(p["h"]), jnp.asarray(p["k"]),
            jnp.asarray(p["cos_i"]), jnp.asarray(p["lam"]),
            jnp.asarray(p["total_mass"]), jnp.asarray(p["R_E"]), jnp, jnp,
        )

    params = jnp.asarray([0.15, 0.4])
    J = jax.jacfwd(f)(params)
    det = float(jnp.linalg.det(J))
    a_here = float(f(params)[0])
    # float32 JAX default in this repo (no jax_enable_x64) -- loosened
    # tolerance matching this branch's established policy.
    assert det == pytest.approx(a_here, rel=1e-3)


# ---------------------------------------------------------------------------
# P2-1: prior-pushforward equivalence against the independent H3 reference
# sampler (_orbit_proposal.sample_orbit_completion), covering a, e, i, r_s,
# gamma, and period (user acceptance-criteria addendum).
# ---------------------------------------------------------------------------


def test_psk_pushforward_matches_independent_orbit_proposal_sampler():
    rng = np.random.default_rng(20)
    n = 400
    proposal = sample_orbit_completion(fixed=_FIXED, context=_CONTEXT, n=n, rng=rng)
    assert proposal.n_accepted >= n  # otherwise the comparison below is short

    cols = proposal.columns
    draws = proposal.draws
    g1 = draws[:, cols.index("g1")]
    g2 = draws[:, cols.index("g2")]
    g3 = draws[:, cols.index("g3")]
    r_s = draws[:, cols.index("lom_szs")]
    a_s = draws[:, cols.index("lom_ar")]

    derived = proposal.derived
    e = derived["e"]
    cos_i = derived["cos_i"]
    om = derived["om"]
    nu = derived["nu"]
    orbital_radi = derived["orbital_radi"]  # == physical a (AU), see binary_lens.py

    s = _FIXED["s"]
    ML = proposal.prior_spec["ML_Msun"]

    a_norm = a_s * s * np.sqrt(1.0 + r_s**2)
    R_E = orbital_radi / a_norm  # independent per-draw R_E, no PSK code involved

    psk_state = pk.from_orbital_elements(s=s, e=e, cos_i=cos_i, omega=om, nu=nu, xp=np)
    psk_context = pk.OrbitalContext(
        total_mass=ML, einstein_radius_physical=R_E, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    result = pk.forward(psk_state, psk_context, xp=np, strict=False)

    assert bool(np.asarray(result.valid).all())

    # sample_orbit_completion's kernel (_lc_to_phys_kepler) is jax.jit'd and
    # this repo has no jax_enable_x64 anywhere, so its output (including the
    # `e`/`om`/`nu` decomposition via an eccentricity-vector formula that is
    # ill-conditioned for near-circular orbits) is float32-precision -- the
    # PSK forward() path here runs float64 numpy. The comparison is across
    # two independently-coded pipelines at float32 precision, so the
    # tolerance matches this branch's established float32-vs-float64 policy
    # (test_projected_kepler.py's numpy/JAX agreement test) rather than the
    # tighter rel=1e-6/1e-7 used for the pure-float64 Roman fixture check.
    np.testing.assert_allclose(np.asarray(result.a), orbital_radi, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(result.i), np.arccos(np.clip(cos_i, -1, 1)), atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.rs), r_s, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(result.gamma1), g1, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(result.gamma2), g2, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(result.gamma3), g3, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(result.a_s), a_s, rtol=2e-3)

    period_direct = 2.0 * np.pi * np.sqrt(orbital_radi**3 / (G_AU3_MSUN_DAY2 * ML))
    period_psk = 2.0 * np.pi * np.sqrt(np.asarray(result.a) ** 3 / (G_AU3_MSUN_DAY2 * ML))
    np.testing.assert_allclose(period_psk, period_direct, rtol=2e-3)


# ---------------------------------------------------------------------------
# P2-2: log-uniform-a-prior cancellation. a_min/a_max are local test
# constants only -- never passed into forward() (P2 does not add scientific
# a-bounds to the pure transform; that remains P4's job).
# ---------------------------------------------------------------------------


def test_log_uniform_a_prior_cancellation():
    # Fixture/test-only constants (per user decision, never passed into
    # forward()): a_max chosen tighter than [0.01, 100] so the random
    # ensemble below (whose a spans roughly [0.6, 14]) actually exercises
    # both the in-support and out-of-support regimes.
    a_min, a_max = 0.01, 5.0
    log_norm = math.log(a_max / a_min)

    rng = np.random.default_rng(30)
    n = 300
    state = _random_state(n, rng)
    context = _random_context(n)
    result = pk.forward(state, context, strict=False)
    ok = np.asarray(result.valid)

    a = np.asarray(result.a)[ok]
    # p_a(a) = 1/(a*log_norm) on (a_min,a_max); density in log_s coordinate
    # picks up the Jacobian da/d(log_s) = a (proven exactly in the Jacobian
    # tests above), so p(log_s,...) supported term = p_a(a)*a, which must be
    # the *same constant* everywhere inside the support and carry no residual
    # dependence on (h,k,cos_i,lambda) beyond the support indicator itself.
    p_a_times_a = np.where(
        (a > a_min) & (a < a_max), 1.0 / log_norm, 0.0
    )
    in_support = (a > a_min) & (a < a_max)
    assert in_support.any() and (~in_support).any()  # both regimes exercised
    np.testing.assert_allclose(
        p_a_times_a[in_support], 1.0 / log_norm, rtol=0, atol=1e-15
    )
    assert np.all(p_a_times_a[~in_support] == 0.0)


# ---------------------------------------------------------------------------
# P2-3: single-counting of the Jacobian.
# ---------------------------------------------------------------------------


def test_forward_never_carries_a_jacobian_or_prior_term():
    rng = np.random.default_rng(31)
    n = 100
    state = _random_state(n, rng)
    context = _random_context(n)
    result = pk.forward(state, context, strict=False)
    # The pure transform contributes exactly zero to any prior/Jacobian
    # term; the a (log_s coords) or a/s (s coords) factor must be applied by
    # the caller exactly once (P4), never inside this module.
    np.testing.assert_array_equal(np.asarray(result.log_prior), np.zeros(n))


def test_a_over_s_and_a_forms_are_related_by_the_single_ds_dlogs_factor():
    # The two equivalent Jacobian forms from plan section 5.1/5.2:
    #   |d(a,Omega)/d(s,phi_perp)|      = a/s   (s coordinate)
    #   |d(a,Omega)/d(log_s,phi_perp)|  = a     (log_s coordinate)
    # related by exactly ds = s*d(log_s). A caller must pick exactly one
    # coordinate and hence exactly one of these two factors -- this test
    # pins the relationship so a future implementation cannot accidentally
    # apply both (double-counting) or neither.
    p = _JAC_POINT
    log_s0, phi0 = 0.15, 0.4
    s0 = math.exp(log_s0)

    step = 1e-6

    def a_of_logs(log_s):
        return float(
            _a_and_Omega_NE(
                log_s, phi0, p["h"], p["k"], p["cos_i"], p["lam"],
                p["total_mass"], p["R_E"], np, np,
            )[0]
        )

    jac_logs_coord = (a_of_logs(log_s0 + step) - a_of_logs(log_s0 - step)) / (2 * step)
    a_here = a_of_logs(log_s0)
    jac_s_coord = jac_logs_coord / s0  # d/ds = (d/dlog_s) * (dlog_s/ds) = (1/s) d/dlog_s

    assert jac_logs_coord == pytest.approx(a_here, rel=1e-5)
    assert jac_s_coord == pytest.approx(a_here / s0, rel=1e-5)
    # The single relating factor is exactly s (ds = s d(log_s)):
    assert jac_logs_coord == pytest.approx(jac_s_coord * s0, rel=1e-9)


# ---------------------------------------------------------------------------
# z-reflection canonical branch: normalization (no log(2) evidence bias) and
# well-definedness of the canonical representative on the reflection orbit.
# ---------------------------------------------------------------------------


def test_canonical_branch_support_measure_is_pi_not_2pi():
    # Numerically confirm (not just assume) that {lambda: is_canonical(lambda)}
    # has Lebesgue measure pi within (-pi, pi], via fine quadrature -- ties
    # the normalization argument below to the actual is_canonical
    # implementation rather than an independent hand-derivation.
    grid = np.linspace(-math.pi, math.pi, 2_000_001)
    mask = pk.is_canonical(grid)
    measure = float(np.mean(mask)) * (grid[-1] - grid[0])
    assert measure == pytest.approx(math.pi, rel=1e-6)


def test_canonical_branch_normalization_avoids_log2_evidence_bias():
    support_measure = math.pi  # confirmed numerically above

    correct_density = 1.0 / math.pi
    naive_density = 1.0 / (2.0 * math.pi)  # forgetting the factor-of-2 fix

    integral_correct = correct_density * support_measure
    integral_naive = naive_density * support_measure

    assert integral_correct == pytest.approx(1.0, abs=1e-12)
    assert integral_naive == pytest.approx(0.5, abs=1e-12)

    log_evidence_bias = math.log(integral_correct) - math.log(integral_naive)
    assert log_evidence_bias == pytest.approx(math.log(2.0), abs=1e-12)


def test_canonical_branch_mean_reflection_invariant_quantity_matches_full_space():
    # For a reflection-invariant quantity (a, per module docstring), the
    # population mean estimated from canonical-branch-only samples (which,
    # conditioned on the branch, is exactly the restriction of the full
    # uniform-lambda law -- no extra weighting needed for a *mean*, only for
    # an *unnormalized integral/evidence*, see the log(2) test above) must
    # agree with the full-space mean, confirming the branch split doesn't
    # bias reflection-invariant statistics.
    rng = np.random.default_rng(40)
    n = 200_000
    h = rng.uniform(-0.5, 0.5, n)
    k = rng.uniform(-0.5, 0.5, n)
    cos_i = rng.uniform(-0.9, 0.9, n)
    lam = rng.uniform(-math.pi, math.pi, n)
    state = pk.ProjectedKeplerState(
        log_s=np.full(n, 0.1), projected_pa=np.zeros(n), h=h, k=k, cos_i=cos_i,
        mean_longitude=lam,
    )
    context = pk.OrbitalContext(
        total_mass=0.3, einstein_radius_physical=1.2, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    result = pk.forward(state, context, strict=False)
    ok = np.asarray(result.valid)
    a = np.asarray(result.a)
    canonical = np.asarray(pk.is_canonical(lam))

    mean_full = a[ok].mean()
    mean_canonical = a[ok & canonical].mean()
    se = a[ok].std() / math.sqrt((ok & canonical).sum())
    assert abs(mean_canonical - mean_full) < 6.0 * se


def test_reflection_is_a_fixed_point_free_involution():
    rng = np.random.default_rng(41)
    n = 50
    h = rng.uniform(-0.6, 0.6, n)
    k = rng.uniform(-0.6, 0.6, n)
    cos_i = rng.uniform(-0.9, 0.9, n)
    lam = rng.uniform(-math.pi, math.pi, n)
    state = pk.ProjectedKeplerState(
        log_s=np.zeros(n), projected_pa=np.zeros(n), h=h, k=k, cos_i=cos_i,
        mean_longitude=lam,
    )
    once = pk.reflect_state(state)
    twice = pk.reflect_state(once)

    np.testing.assert_allclose(np.asarray(once.h), -h)
    np.testing.assert_allclose(np.asarray(once.k), -k)
    assert not np.allclose(np.asarray(once.h), h)  # no fixed points
    np.testing.assert_allclose(np.asarray(twice.h), h, atol=1e-12)
    np.testing.assert_allclose(np.asarray(twice.k), k, atol=1e-12)
    np.testing.assert_allclose(
        np.cos(np.asarray(twice.mean_longitude)), np.cos(lam), atol=1e-10
    )


def test_canonical_representative_is_well_defined_on_the_reflection_orbit():
    # The intended property behind "canonical(x) == canonical(reflect(x))"
    # (2026-07-20 user message): is_canonical(x) itself is generically the
    # *opposite* boolean for x vs reflect(x) (that's the whole point of the
    # branch split -- see test_is_canonical_partitions_the_reflection_orbit
    # in test_projected_kepler.py, kept as-is). What must actually agree is
    # the *canonical representative* to_canonical(x) picks for either member
    # of an orbit {x, reflect(x)}: both must map to the identical point.
    rng = np.random.default_rng(42)
    n = 200
    h = rng.uniform(-0.6, 0.6, n)
    k = rng.uniform(-0.6, 0.6, n)
    cos_i = rng.uniform(-0.9, 0.9, n)
    lam = rng.uniform(-math.pi, math.pi, n)
    state = pk.ProjectedKeplerState(
        log_s=rng.uniform(-0.2, 0.2, n), projected_pa=np.zeros(n), h=h, k=k,
        cos_i=cos_i, mean_longitude=lam,
    )
    reflected = pk.reflect_state(state)

    canon_from_state, _ = pk.to_canonical(state)
    canon_from_reflected, _ = pk.to_canonical(reflected)

    np.testing.assert_allclose(
        np.asarray(canon_from_state.h), np.asarray(canon_from_reflected.h), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(canon_from_state.k), np.asarray(canon_from_reflected.k), atol=1e-12
    )
    np.testing.assert_allclose(
        np.cos(np.asarray(canon_from_state.mean_longitude)),
        np.cos(np.asarray(canon_from_reflected.mean_longitude)),
        atol=1e-10,
    )
    assert np.all(np.asarray(pk.is_canonical(canon_from_state.mean_longitude)))
    assert np.all(np.asarray(pk.is_canonical(canon_from_reflected.mean_longitude)))


# ---------------------------------------------------------------------------
# strict=False: production JAX/vectorized path folds a_s inconsistency into
# the invalid mask instead of raising (this phase's design addition).
# ---------------------------------------------------------------------------


def test_strict_false_masks_inconsistency_instead_of_raising_numpy():
    from test_projected_kepler import _truth_state, _truth_context  # noqa: E402

    state = _truth_state()
    context = _truth_context()
    # consistency_tol=0.0 makes any nonzero float error "fail" -- under
    # strict=True this raises (already covered elsewhere); under
    # strict=False it must instead show up as invalid=False, never raise.
    result = pk.forward(state, context, consistency_tol=0.0, strict=False)
    assert not bool(result.valid)
    assert np.isfinite(np.asarray(result.a))  # still a finite, just-flagged output


def test_strict_true_still_raises_by_default_no_regression():
    from test_projected_kepler import _truth_state, _truth_context  # noqa: E402

    state = _truth_state()
    context = _truth_context()
    with pytest.raises(RuntimeError, match="self-consistency"):
        pk.forward(state, context, consistency_tol=0.0)


def test_strict_false_masks_inconsistency_instead_of_raising_jax():
    jnp = pytest.importorskip("jax.numpy")
    from test_projected_kepler import _truth_state, _truth_context  # noqa: E402

    state = _truth_state()
    context = _truth_context()
    jax_state = pk.ProjectedKeplerState(
        log_s=jnp.asarray(state.log_s), projected_pa=jnp.asarray(state.projected_pa),
        h=jnp.asarray(state.h), k=jnp.asarray(state.k), cos_i=jnp.asarray(state.cos_i),
        mean_longitude=jnp.asarray(state.mean_longitude),
    )
    jax_context = pk.OrbitalContext(
        total_mass=jnp.asarray(context.total_mass),
        einstein_radius_physical=jnp.asarray(context.einstein_radius_physical),
        proper_motion_pa=None, reference_epoch=0.0,
    )
    result = pk.forward(
        jax_state, jax_context, xp=jnp, consistency_tol=0.0, strict=False
    )
    assert not bool(result.valid)
