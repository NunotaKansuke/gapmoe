from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gapmoe.parameterizations import (
    BinaryCircularParameterization,
    BinaryCircularUseThEParameterization,
    BinaryKeplerParameterization,
    SingleLensParameterization,
    SingleLensUseThEParameterization,
    calc_vEarth,
)
from gapmoe import BinaryCircularParameterization as _top_level_import
from gapmoe import JaxGalacticPrior


# Representative context: values at event peak
_CTX = {"thS": 0.5, "vEarth": (28.0, -3.0)}

# Physically rough but numerically stable parameter vectors
_THETA_CIRC = jnp.array([
    2460000.0,   # t0 [day]
    50.0,        # tE [day]
    0.1,         # u0
    0.005,       # rho
    0.1,         # q
    1.0,         # s
    0.5,         # alpha [rad]
    0.1,         # piEN
    0.05,        # piEE
    0.001,       # gamma1
    0.002,       # gamma2
    -0.003,      # gamma3
])

_THETA_CIRC_THE = jnp.array([
    2460000.0,   # t0
    50.0,        # tE
    0.1,         # u0
    1.0,         # thE [mas]
    0.1,         # q
    1.0,         # s
    0.5,         # alpha
    0.1,         # piEN
    0.05,        # piEE
    0.001,       # gamma1
    0.002,       # gamma2
    -0.003,      # gamma3
])

_THETA_KEPLER = jnp.array([
    2460000.0,   # t0
    50.0,        # tE
    0.1,         # u0
    0.005,       # rho
    0.1,         # q
    1.0,         # s
    0.5,         # alpha
    0.1,         # piEN
    0.05,        # piEE
    0.001,       # gamma1
    0.002,       # gamma2
    -0.003,      # gamma3
    0.1,         # r_s
    1.1,         # a_s  (must be > 0.5 for denominator 2*a_s - 1 > 0)
])

_THETA_SINGLE = jnp.array([
    2460000.0,   # t0
    50.0,        # tE
    0.1,         # u0
    0.005,       # rho
    0.1,         # piEN
    0.05,        # piEE
    8.0,         # DS [kpc]
])

_THETA_SINGLE_THE = jnp.array([
    2460000.0,   # t0
    50.0,        # tE
    0.1,         # u0
    1.0,         # thE [mas]
    0.1,         # piEN
    0.05,        # piEE
    8.0,         # DS [kpc]
])


def test_top_level_import():
    assert _top_level_import is BinaryCircularParameterization


def test_binary_circular_names():
    p = BinaryCircularParameterization()
    assert len(p.names) == 12
    assert p.names[3] == "rho"


def test_binary_circular_use_thE_names():
    p = BinaryCircularUseThEParameterization()
    assert len(p.names) == 12
    assert p.names[3] == "thE"


def test_binary_kepler_names():
    p = BinaryKeplerParameterization()
    assert len(p.names) == 14
    assert "r_s" in p.names
    assert "a_s" in p.names


def test_single_lens_names():
    p = SingleLensParameterization()
    assert len(p.names) == 7
    assert p.names[-1] == "DS"


def test_single_lens_use_thE_names():
    p = SingleLensUseThEParameterization()
    assert len(p.names) == 7
    assert p.names[3] == "thE"


@pytest.mark.parametrize("cls,theta,ctx", [
    (BinaryCircularParameterization, _THETA_CIRC, _CTX),
    (BinaryCircularUseThEParameterization, _THETA_CIRC_THE, _CTX),
    (BinaryKeplerParameterization, _THETA_KEPLER, _CTX),
    (SingleLensParameterization, _THETA_SINGLE, _CTX),
    (SingleLensUseThEParameterization, _THETA_SINGLE_THE, _CTX),
])
def test_to_physical_returns_five_finite_scalars(cls, theta, ctx):
    p = cls()
    result = p.to_physical(theta, ctx)
    assert len(result) == 5
    assert all(np.isfinite(float(v)) for v in result)


@pytest.mark.parametrize("cls,theta,ctx", [
    (BinaryCircularParameterization, _THETA_CIRC, _CTX),
    (BinaryCircularUseThEParameterization, _THETA_CIRC_THE, _CTX),
    (BinaryKeplerParameterization, _THETA_KEPLER, _CTX),
    (SingleLensParameterization, _THETA_SINGLE, _CTX),
    (SingleLensUseThEParameterization, _THETA_SINGLE_THE, _CTX),
])
def test_jacobian_is_finite(cls, theta, ctx):
    p = cls()
    lndet = p.log_abs_det_jacobian(theta, ctx)
    assert np.isfinite(float(lndet))


def test_missing_vEarth_raises():
    p = BinaryCircularParameterization()
    with pytest.raises(ValueError, match="vEarth"):
        p.to_physical(_THETA_CIRC, {"thS": 0.5})


def test_missing_thS_raises():
    p = BinaryCircularParameterization()
    with pytest.raises(ValueError, match="thS"):
        p.to_physical(_THETA_CIRC, {"vEarth": (28.0, -3.0)})


def test_thE_variant_does_not_need_thS():
    p = BinaryCircularUseThEParameterization()
    ctx_no_thS = {"vEarth": (28.0, -3.0)}
    result = p.to_physical(_THETA_CIRC_THE, ctx_no_thS)
    assert len(result) == 5


def test_single_lens_ds_is_in_theta():
    p = SingleLensParameterization()
    _, _, DS, _, _ = p.to_physical(_THETA_SINGLE, _CTX)
    # DS is derived from theta[6] = 8.0 kpc; DL should be < DS
    _, DL, DS_out, _, _ = p.to_physical(_THETA_SINGLE, _CTX)
    assert DS_out == pytest.approx(8.0, rel=1e-4)
    assert DL < DS_out


def test_calc_vEarth_returns_tuple():
    v_N, v_E = calc_vEarth(2460000.0, 270.0, -30.0)
    assert np.isfinite(v_N)
    assert np.isfinite(v_E)
    assert v_N == pytest.approx(-0.6556513128, rel=1e-9)
    assert v_E == pytest.approx(2.7273324385, rel=1e-9)


def test_calc_vEarth_accepts_reduced_jd():
    v_N, v_E = calc_vEarth(9000.0, 270.0, -30.0)
    assert v_N == pytest.approx(-0.2569462867, rel=1e-9)
    assert v_E == pytest.approx(-5.7716627122, rel=1e-9)


class _DummyJaxDensity:
    def log_density(self, ML, DL, DS, mu_N, mu_E):
        return -(ML + DL + DS + 0.01 * jnp.hypot(mu_N, mu_E))


def test_jax_prior_with_binary_parameterization_is_jittable():
    prior = JaxGalacticPrior(
        _DummyJaxDensity(),
        parameterization=BinaryCircularParameterization(),
        include_event_rate=False,
    )

    @jax.jit
    def lp(theta):
        return prior.log_prob(theta, context=_CTX)

    value = lp(_THETA_CIRC)
    assert np.isfinite(float(value))
