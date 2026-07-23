"""Tests for the private orbit-completion proposal sampler (LAPS-SMC H3).

The sampler is proposal-construction-only: these tests check (a) fixed
components are respected bitwise, (b) every draw is a valid bound orbit under
the *same* validity definition the target side uses (the public
``ParamType(...).log_abs_det_jacobian`` / ``to_derived`` path), (c) acceptance
diagnostics are present and sane, and (d) the induced element distribution
follows the documented default prior rather than an arbitrary box in ``g``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gapmoe.param_types import ParamType
from gapmoe.param_types._orbit_proposal import (
    OrbitProposal,
    sample_orbit_completion,
    sample_orbit_completion_circular,
)


_CTX = {"thS": 0.5, "vEarth": (28.0, -3.0)}

_FIXED = {
    "t0": 2460000.0,
    "tE": 50.0,
    "u0": 0.1,
    "rho": 0.005,
    "q": 0.1,
    "s": 1.0,
    "alpha": 0.5,
    "piEN": 0.1,
    "piEE": 0.05,
}


def _kepler_pt():
    return ParamType(parallax=True, orbital_motion="kepler")


def _circular_pt():
    return ParamType(parallax=True, orbital_motion="circular")


# ---------------------------------------------------------------------------
# 1. Fixed components respected bitwise; columns match the non-fixed subset
# ---------------------------------------------------------------------------

def test_columns_are_the_non_fixed_orbit_subset():
    prop = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=50, rng=0)
    assert prop.columns == ("g1", "g2", "g3", "lom_szs", "lom_ar")
    assert prop.draws.shape == (50, 5)
    assert prop.theta_names == _kepler_pt().names


def test_pinned_orbit_component_is_bitwise_respected():
    g1_pin = -0.000123456789
    fixed = dict(_FIXED, g1=g1_pin)
    prop = sample_orbit_completion(fixed=fixed, context=_CTX, n=40, rng=0)

    assert prop.columns == ("g2", "g3", "lom_szs", "lom_ar")
    assert prop.draws.shape == (40, 4)
    assert "g1" in prop.conditioned_on
    assert prop.conditioned_on["g1"] == g1_pin  # bitwise

    thetas = prop.spliced_thetas()
    j = prop.theta_names.index("g1")
    assert np.all(thetas[:, j] == g1_pin)  # bitwise in the spliced theta
    for name, val in _FIXED.items():
        k = prop.theta_names.index(name)
        assert np.all(thetas[:, k] == val)


def test_conditioned_on_records_exactly_the_fixed_values():
    prop = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=10, rng=0)
    assert prop.conditioned_on == _FIXED
    # metadata for the conditioning-derived scalars is also recorded
    assert prop.prior_spec["DL_max_kpc"] > 0
    assert prop.prior_spec["ML_Msun"] > 0


# ---------------------------------------------------------------------------
# 2. Every draw valid through the public target-side path
# ---------------------------------------------------------------------------

def test_all_draws_valid_via_public_jacobian_and_derived_path():
    prop = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=300, rng=1)
    pt = _kepler_pt()
    thetas = prop.spliced_thetas()
    for i in range(thetas.shape[0]):
        lad = float(pt.log_abs_det_jacobian(thetas[i], _CTX))
        assert np.isfinite(lad), f"non-finite Jacobian for draw {i}"
        derived = pt.to_derived(thetas[i], _CTX)
        for name, val in derived.items():
            assert np.isfinite(float(val)), f"non-finite derived {name} at {i}"


def test_all_draws_valid_circular_via_public_path():
    prop = sample_orbit_completion_circular(
        fixed=_FIXED, context=_CTX, n=200, rng=3
    )
    assert prop.columns == ("g1", "g2", "g3")
    pt = _circular_pt()
    thetas = prop.spliced_thetas()
    assert thetas.shape == (200, len(pt.names))
    for i in range(thetas.shape[0]):
        assert np.isfinite(float(pt.log_abs_det_jacobian(thetas[i], _CTX)))


# ---------------------------------------------------------------------------
# 3. Acceptance diagnostics; hostile fixed values terminate cleanly
# ---------------------------------------------------------------------------

def test_acceptance_diagnostics_present_and_sane():
    prop = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=100, rng=2)
    assert 0.0 < prop.acceptance <= 1.0
    assert prop.n_accepted >= 100
    assert prop.n_proposed >= prop.n_accepted
    assert prop.acceptance == prop.n_accepted / prop.n_proposed
    assert prop.n_requested == 100


def test_hostile_pinned_g_errors_without_hanging():
    # Pinning all g components to ~0 forces the implied Einstein radius
    # RE = 1/orbital_scale above 1/piE, i.e. DS <= 0: no valid orbit exists.
    hostile = dict(_FIXED, g1=1e-12, g2=1e-12, g3=1e-12)
    with pytest.raises(RuntimeError, match="accepted 0"):
        sample_orbit_completion(
            fixed=hostile, context=_CTX, n=50, rng=0, max_tries=4000
        )


def test_all_orbit_components_fixed_is_an_error():
    fixed = dict(
        _FIXED, g1=-1e-4, g2=-1e-4, g3=-0.01, lom_szs=0.1, lom_ar=1.1
    )
    with pytest.raises(ValueError, match="nothing to sample"):
        sample_orbit_completion(fixed=fixed, context=_CTX, n=10, rng=0)


def test_missing_base_component_is_an_error():
    fixed = {k: v for k, v in _FIXED.items() if k != "s"}
    with pytest.raises(ValueError, match="missing"):
        sample_orbit_completion(fixed=fixed, context=_CTX, n=10, rng=0)


# ---------------------------------------------------------------------------
# 4. Distributional sanity of the induced element prior; reproducibility
# ---------------------------------------------------------------------------

def test_derived_elements_follow_documented_prior():
    prop = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=4000, rng=7)
    d = prop.derived

    cos_i = d["cos_i"]
    assert np.all(np.abs(cos_i) <= 1.0)
    # roughly uniform over (-1, 1): each quartile bin near 0.25
    hist, _ = np.histogram(cos_i, bins=4, range=(-1.0, 1.0))
    frac = hist / cos_i.size
    assert np.all(frac > 0.17) and np.all(frac < 0.33)

    e = d["e"]
    assert np.all(e >= 0.0)
    assert np.all(e < 0.96)  # documented default range (float32 slack)
    assert e.max() > 0.7 and e.min() < 0.2  # spans the range

    assert np.all(d["orbital_radi"] > 0.0)
    assert np.all(np.isfinite(d["Om_NE"]))
    assert np.all(np.isfinite(d["om"]))
    assert np.all(np.isfinite(d["nu"]))


def test_element_prior_override_pins_conditioned_elements():
    # Constant element draws must reappear in gapmoe's derived elements:
    # proves the forward construction inverts gapmoe's LC->element map.
    override = {
        "e": lambda rng, m: np.full(m, 0.3),
        "cos_i": lambda rng, m: np.full(m, 0.4),
    }
    prop = sample_orbit_completion(
        fixed=_FIXED, context=_CTX, n=200, element_prior=override, rng=5
    )
    assert np.allclose(prop.derived["e"], 0.3, atol=2e-3)       # float32 path
    assert np.allclose(prop.derived["cos_i"], 0.4, atol=2e-3)


def test_seed_reproducibility():
    a = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=64, rng=42)
    b = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=64, rng=42)
    c = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=64, rng=43)
    assert np.array_equal(a.draws, b.draws)
    assert not np.array_equal(a.draws, c.draws)


def test_result_type_and_no_density_attribute():
    prop = sample_orbit_completion(fixed=_FIXED, context=_CTX, n=8, rng=0)
    assert isinstance(prop, OrbitProposal)
    # Method A contract: samples only, never a proposal density.
    assert not hasattr(prop, "log_prob")
    assert not hasattr(prop, "log_density")
