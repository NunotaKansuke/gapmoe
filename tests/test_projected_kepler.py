"""Tests for the Projected-Separation Kepler (PSK) pure transform (P1).

Coverage, per ``.note/PSK-LAPS_implementation_checklist.md`` P1-1..P1-4 and
``.note/PSK_conventions.md`` (the binding convention document this module is
built from):

* fixture regression against the Roman fast binary GULLS worked example
  (``PSK_conventions.md`` section 6) in both directions (inverse: old
  elements -> PSK; forward: PSK -> state);
* numerical stability at e->0, i->0/pi, g_proj->0, large a, e near the
  elliptical-solver domain limit;
* the a_s = a/|r| self-consistency check actually raises on a genuine
  mismatch (not just flags invalid);
* the z-reflection canonical-branch operators (P1-4): involution property,
  branch coverage, and that the physical-side mirror
  (``reflect_transform_result``) agrees with transforming the
  coordinate-level mirror (``reflect_state``) through ``forward``;
* the shared physical constant hasn't drifted from
  ``gapmoe.param_types.binary_lens._G``;
* numpy vs JAX agreement (skipped if JAX float32 tolerance would make the
  comparison meaningless -- loosened per this branch's established float32
  policy).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gapmoe.orbit import projected_kepler as pk


# ---------------------------------------------------------------------------
# Roman fast binary GULLS worked example (PSK_conventions.md section 6).
# Truth values are transcribed directly from that document (not re-derived
# here) so this test is a genuine regression check against the P0 deliverable
# rather than a tautology against this module's own arithmetic.
# ---------------------------------------------------------------------------

_TRUTH = dict(
    s=1.22,
    alpha=0.1832595714594046,
    piEN=0.12,
    piEE=-0.08,
    ML=0.119377889587,
    RE=1.066429607,
    a=1.49050606165,
    e=0.274777750949,
    cos_i=-0.00121290856386,
    omega=math.radians(250.191219490) - 2 * math.pi,  # wrap to (-pi, pi]
    nu=math.radians(274.719380570),
    M=-0.956637945,
    Omega0_deg=179.981262726,
    h=-0.177639449,
    k=-0.493175402,
    lam=-2.873162712,
    log_s=0.198850858745,
    gamma1=7.030743672e-06,
    gamma2=-4.999275539e-06,
    gamma3=-4.119826095e-03,
    rs=0.26962203,
    a_s=1.1061229,
)


def _truth_state(projected_pa=0.0) -> pk.ProjectedKeplerState:
    t = _TRUTH
    return pk.ProjectedKeplerState(
        log_s=math.log(t["s"]),
        projected_pa=projected_pa,
        h=t["h"],
        k=t["k"],
        cos_i=t["cos_i"],
        mean_longitude=t["lam"],
    )


def _truth_context() -> pk.OrbitalContext:
    t = _TRUTH
    phi_mu = math.atan2(t["piEE"], t["piEN"])
    return pk.OrbitalContext(
        total_mass=t["ML"],
        einstein_radius_physical=t["RE"],
        proper_motion_pa=phi_mu,
        reference_epoch=11662.0,
    )


def test_inverse_matches_worked_example_psk_values():
    t = _TRUTH
    state = pk.from_orbital_elements(
        s=t["s"], e=t["e"], cos_i=t["cos_i"], omega=t["omega"], nu=t["nu"]
    )
    assert state.log_s == pytest.approx(t["log_s"], abs=1e-9)
    assert state.h == pytest.approx(t["h"], abs=1e-8)
    assert state.k == pytest.approx(t["k"], abs=1e-8)
    assert state.mean_longitude == pytest.approx(t["lam"], abs=1e-8)


def test_forward_matches_worked_example_state():
    state = _truth_state()
    context = _truth_context()
    result = pk.forward(state, context)

    t = _TRUTH
    assert result.a == pytest.approx(t["a"], rel=1e-7)
    assert result.e == pytest.approx(t["e"], abs=1e-9)
    assert result.a_s == pytest.approx(t["a_s"], rel=1e-6)
    assert result.rs == pytest.approx(t["rs"], rel=1e-6)
    assert result.gamma1 == pytest.approx(t["gamma1"], rel=1e-6)
    assert result.gamma2 == pytest.approx(t["gamma2"], rel=1e-6)
    assert result.gamma3 == pytest.approx(t["gamma3"], rel=1e-6)
    assert math.degrees(float(result.Omega)) == pytest.approx(
        t["Omega0_deg"], abs=1e-4
    )
    assert bool(result.valid)
    assert float(result.consistency_error) < 1e-8

    # Frame-T anchor (PSK_conventions.md C7/section 6): projected separation
    # lies along +x with the fixture's own R_perp = s*R_E.
    assert float(result.r[0]) == pytest.approx(t["s"] * t["RE"], rel=1e-8)
    assert float(result.r[1]) == pytest.approx(0.0, abs=1e-9)


def test_derived_absolute_position_angle_matches_Omega_NE():
    state = _truth_state(projected_pa=_TRUTH["piEN"])  # placeholder, overwritten below
    phi_mu = math.atan2(_TRUTH["piEE"], _TRUTH["piEN"])
    phi_perp = phi_mu - _TRUTH["alpha"]
    state = pk.ProjectedKeplerState(
        log_s=state.log_s,
        projected_pa=phi_perp,
        h=state.h,
        k=state.k,
        cos_i=state.cos_i,
        mean_longitude=state.mean_longitude,
    )
    context = _truth_context()
    result = pk.forward(state, context)
    omega_ne = pk.derived_absolute_position_angle(state, result)
    assert math.degrees(float(omega_ne)) == pytest.approx(135.791195201, abs=1e-3)


# ---------------------------------------------------------------------------
# Numerical stability (P1-2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h,k,cos_i,lam",
    [
        (0.0, 0.0, 0.3, 0.7),  # e -> 0 (circular): omega undefined but continuous downstream
        (0.5, 0.1, 1.0 - 1e-12, 0.2),  # i -> 0
        (0.5, 0.1, -(1.0 - 1e-12), 0.2),  # i -> pi
        (0.9, 0.1, 0.0, math.pi / 2),  # cos_i = 0, sin(u)=1 at some phase: near projection edge
        (0.1, 0.05, 0.0, 0.0),  # g_proj can approach 0 depending on omega+nu
        (3.0, 3.0, 0.5, 1.0),  # large e (still < 1): e = 18 here, exceeds domain -> must be invalid, not nan
        (0.05, 0.05, 0.2, -3.0),
    ],
)
def test_numerical_stability_no_nan_and_domain_flagged(h, k, cos_i, lam):
    state = pk.ProjectedKeplerState(
        log_s=0.5, projected_pa=0.0, h=h, k=k, cos_i=cos_i, mean_longitude=lam
    )
    context = pk.OrbitalContext(
        total_mass=0.3, einstein_radius_physical=1.2, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    result = pk.forward(state, context)
    for field in (result.a, result.e, result.i, result.Omega, result.omega,
                  result.mean_anomaly, result.rs, result.gamma1, result.gamma2,
                  result.gamma3, result.a_s):
        assert np.isfinite(np.asarray(field)).all(), (h, k, cos_i, lam, field)
    e = h**2 + k**2
    if e >= 1.0:
        assert not bool(result.valid)


def test_large_semimajor_axis_stays_finite():
    # a is *derived*, not bounded here (no a-bounds in P1); a very large R_E
    # with near-singular projection pushes a large without triggering a-bound
    # logic that doesn't exist in this module.
    state = pk.ProjectedKeplerState(
        log_s=0.0, projected_pa=0.0, h=0.01, k=0.01, cos_i=0.999, mean_longitude=0.3
    )
    context = pk.OrbitalContext(
        total_mass=1.0, einstein_radius_physical=1e4, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    result = pk.forward(state, context)
    assert np.isfinite(result.a)


def test_e_near_domain_limit_is_flagged_invalid_not_nan():
    # h^2 + k^2 = e = 1.0 exactly: at/beyond the elliptical solver's domain
    # limit (_E_DOMAIN_MAX = 1 - 1e-9), must be flagged invalid, not nan.
    state = pk.ProjectedKeplerState(
        log_s=0.0, projected_pa=0.0, h=1.0, k=0.0, cos_i=0.1, mean_longitude=0.4
    )
    context = pk.OrbitalContext(
        total_mass=1.0, einstein_radius_physical=1.0, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    result = pk.forward(state, context)
    assert np.isfinite(result.a)
    assert not bool(result.valid)


# ---------------------------------------------------------------------------
# Self-consistency check must raise, not silently flag invalid (P1-3, plan
# section 4.3).
# ---------------------------------------------------------------------------


def test_consistency_check_raises_on_genuine_mismatch():
    state = _truth_state()
    context = _truth_context()
    with pytest.raises(RuntimeError, match="self-consistency"):
        pk.forward(state, context, consistency_tol=0.0)


def test_consistency_check_passes_with_default_tolerance():
    state = _truth_state()
    context = _truth_context()
    result = pk.forward(state, context)
    assert float(result.consistency_error) < 1e-6


# ---------------------------------------------------------------------------
# z-reflection canonical branch (P1-4).
# ---------------------------------------------------------------------------


def test_reflect_state_is_a_fixed_point_free_involution():
    state = pk.ProjectedKeplerState(
        log_s=0.3, projected_pa=0.1, h=0.4, k=-0.2, cos_i=0.5, mean_longitude=1.0
    )
    once = pk.reflect_state(state)
    twice = pk.reflect_state(once)

    assert once.h == pytest.approx(-state.h)
    assert once.k == pytest.approx(-state.k)
    assert once.cos_i == state.cos_i
    assert once.log_s == state.log_s
    assert math.cos(once.mean_longitude) == pytest.approx(
        -math.cos(state.mean_longitude), abs=1e-12
    )
    # no fixed points: reflected state must differ from the original.
    assert once.h != pytest.approx(state.h) or once.mean_longitude != pytest.approx(
        state.mean_longitude
    )
    # involution: applying twice returns to the original.
    assert twice.h == pytest.approx(state.h, abs=1e-12)
    assert twice.k == pytest.approx(state.k, abs=1e-12)
    assert math.cos(twice.mean_longitude) == pytest.approx(
        math.cos(state.mean_longitude), abs=1e-12
    )


@pytest.mark.parametrize("lam", [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
def test_is_canonical_partitions_the_reflection_orbit(lam):
    state = pk.ProjectedKeplerState(
        log_s=0.0, projected_pa=0.0, h=0.3, k=0.2, cos_i=0.1, mean_longitude=lam
    )
    reflected = pk.reflect_state(state)
    # Exactly one of {state, reflected} is canonical (measure-zero boundary
    # cos(lambda) == 0 aside, none of the parametrized values hit it).
    assert bool(pk.is_canonical(state.mean_longitude)) != bool(
        pk.is_canonical(reflected.mean_longitude)
    )


def test_to_canonical_never_touches_hk_disk_radius():
    # The canonical-branch split via lambda alone must never change e = h^2+k^2,
    # since it is only ever a sign flip of both h and k together.
    rng = np.random.default_rng(0)
    h = rng.uniform(-0.8, 0.8, size=64)
    k = rng.uniform(-0.8, 0.8, size=64)
    lam = rng.uniform(-math.pi, math.pi, size=64)
    state = pk.ProjectedKeplerState(
        log_s=np.zeros(64), projected_pa=np.zeros(64), h=h, k=k, cos_i=np.full(64, 0.2),
        mean_longitude=lam,
    )
    canonical, flipped = pk.to_canonical(state)
    np.testing.assert_allclose(
        canonical.h**2 + canonical.k**2, h**2 + k**2, atol=1e-12
    )
    assert np.all(np.cos(np.asarray(canonical.mean_longitude)) >= -1e-12)
    assert flipped.any() and (~flipped).any()  # both branches present in this sample


def test_reflect_transform_result_matches_forward_of_reflected_state():
    state = _truth_state()
    context = _truth_context()
    result = pk.forward(state, context)
    mirrored_via_physics = pk.reflect_transform_result(result)

    reflected_state = pk.reflect_state(state)
    mirrored_via_coords = pk.forward(reflected_state, context)

    assert mirrored_via_physics.rs == pytest.approx(
        float(mirrored_via_coords.rs), rel=1e-6
    )
    assert mirrored_via_physics.gamma3 == pytest.approx(
        float(mirrored_via_coords.gamma3), rel=1e-6
    )
    assert mirrored_via_physics.gamma1 == pytest.approx(
        float(mirrored_via_coords.gamma1), rel=1e-6
    )
    assert mirrored_via_physics.gamma2 == pytest.approx(
        float(mirrored_via_coords.gamma2), rel=1e-6
    )
    assert mirrored_via_physics.a == pytest.approx(float(mirrored_via_coords.a), rel=1e-6)
    assert mirrored_via_physics.e == pytest.approx(float(mirrored_via_coords.e), abs=1e-9)
    # The reflection is an exact light-curve degeneracy: rs, gamma3 negate.
    assert mirrored_via_physics.rs == pytest.approx(-float(result.rs), rel=1e-6)
    assert mirrored_via_physics.gamma3 == pytest.approx(-float(result.gamma3), rel=1e-6)
    assert mirrored_via_physics.gamma1 == pytest.approx(float(result.gamma1), rel=1e-6)
    assert mirrored_via_physics.gamma2 == pytest.approx(float(result.gamma2), rel=1e-6)


# ---------------------------------------------------------------------------
# Constant-drift guard and NumPy/JAX agreement.
# ---------------------------------------------------------------------------


def test_shared_gravitational_constant_matches_binary_lens():
    from gapmoe.param_types.binary_lens import _G

    assert pk.G_AU3_MSUN_DAY2 == _G


def test_vectorized_batch_matches_scalar_loop():
    rng = np.random.default_rng(1)
    n = 32
    h = rng.uniform(-0.6, 0.6, n)
    k = rng.uniform(-0.6, 0.6, n)
    cos_i = rng.uniform(-0.9, 0.9, n)
    lam = rng.uniform(-math.pi, math.pi, n)
    log_s = rng.uniform(-0.5, 0.5, n)

    state = pk.ProjectedKeplerState(
        log_s=log_s, projected_pa=np.zeros(n), h=h, k=k, cos_i=cos_i,
        mean_longitude=lam,
    )
    context = pk.OrbitalContext(
        total_mass=0.4, einstein_radius_physical=1.1, proper_motion_pa=None,
        reference_epoch=0.0,
    )
    batch = pk.forward(state, context)

    for j in range(n):
        scalar_state = pk.ProjectedKeplerState(
            log_s=log_s[j], projected_pa=0.0, h=h[j], k=k[j], cos_i=cos_i[j],
            mean_longitude=lam[j],
        )
        single = pk.forward(scalar_state, context)
        if not bool(batch.valid[j]):
            continue
        assert single.a == pytest.approx(float(batch.a[j]), rel=1e-9)
        assert single.rs == pytest.approx(float(batch.rs[j]), rel=1e-9)
        assert single.gamma3 == pytest.approx(float(batch.gamma3[j]), rel=1e-9)


def test_numpy_and_jax_paths_agree():
    jnp = pytest.importorskip("jax.numpy")
    state = _truth_state()
    context = _truth_context()

    numpy_result = pk.forward(state, context)

    jax_state = pk.ProjectedKeplerState(
        log_s=jnp.asarray(state.log_s),
        projected_pa=jnp.asarray(state.projected_pa),
        h=jnp.asarray(state.h),
        k=jnp.asarray(state.k),
        cos_i=jnp.asarray(state.cos_i),
        mean_longitude=jnp.asarray(state.mean_longitude),
    )
    jax_context = pk.OrbitalContext(
        total_mass=jnp.asarray(context.total_mass),
        einstein_radius_physical=jnp.asarray(context.einstein_radius_physical),
        proper_motion_pa=None,
        reference_epoch=0.0,
    )
    # JAX defaults to float32 in this repo (no jax_enable_x64 anywhere) -- the
    # tolerance here matches this branch's established float32 policy rather
    # than the float64 precision used in the fixture-regression tests above.
    jax_result = pk.forward(jax_state, jax_context, xp=jnp, consistency_tol=1e-2)

    assert float(jax_result.a) == pytest.approx(float(numpy_result.a), rel=1e-4)
    assert float(jax_result.rs) == pytest.approx(float(numpy_result.rs), rel=1e-4)
    assert float(jax_result.gamma3) == pytest.approx(
        float(numpy_result.gamma3), rel=1e-4
    )
