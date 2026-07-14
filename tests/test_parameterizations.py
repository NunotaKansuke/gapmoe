from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gapmoe.param_types import (
    BinaryCircularParamType,
    BinaryCircularUseThEParamType,
    BinaryKeplerParamType,
    ParamType,
    SingleLensParamType,
    SingleLensUseThEParamType,
    calc_vEarth,
    from_model_spec,
)
from gapmoe.priors.galactic import GalacticModel
from gapmoe.priors.mapped import MappedGalacticModel


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
    -0.0001,     # gamma1
    -0.0001,     # gamma2
    -0.01,       # gamma3
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
    -0.0001,     # gamma1
    -0.0001,     # gamma2
    -0.001,      # gamma3
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
    -0.0001,     # gamma1
    -0.0001,     # gamma2
    -0.01,       # gamma3
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

_THETA_STATIC_NO_PARALLAX = jnp.array([
    2460000.0,   # t0
    50.0,        # tE
    0.1,         # u0
    0.005,       # rho
])

_THETA_STATIC_NO_PARALLAX_SAMPLE_DISTANCE = jnp.array([
    2460000.0,   # t0
    50.0,        # tE
    0.1,         # u0
    0.005,       # rho
    4.0,         # DL [kpc]
    8.0,         # DS [kpc]
])


def test_parameter_type_imports():
    assert BinaryCircularParamType is not None
    assert ParamType is not None


def test_binary_circular_names():
    p = BinaryCircularParamType()
    assert len(p.names) == 12
    assert p.names[3] == "rho"


def test_binary_circular_use_thE_names():
    p = BinaryCircularUseThEParamType()
    assert len(p.names) == 12
    assert p.names[3] == "thE"


def test_binary_kepler_names():
    p = BinaryKeplerParamType()
    assert len(p.names) == 14
    assert "r_s" in p.names
    assert "a_s" in p.names


def test_single_lens_names():
    p = SingleLensParamType()
    assert len(p.names) == 7
    assert p.names[-1] == "DS"


def test_single_lens_use_thE_names():
    p = SingleLensUseThEParamType()
    assert len(p.names) == 7
    assert p.names[3] == "thE"


def test_impl_with_parallax_uses_generic_parallax_mapping():
    p = ParamType(lens="triple", parallax=True)
    assert p.names == ("t0", "tE", "u0", "rho", "piEN", "piEE", "DS")
    result = p.to_physical(_THETA_SINGLE, _CTX)
    assert len(result) == 5
    assert all(np.isfinite(float(v)) for v in result)


def test_impl_without_parallax_marginalizes_distances_by_default():
    p = ParamType(parallax=False)
    assert p.names == ("t0", "tE", "u0", "rho")
    with pytest.raises(TypeError, match="to_theta_mu_physical"):
        p.to_physical(_THETA_STATIC_NO_PARALLAX, {"thS": 0.5})
    result = p.to_theta_mu_physical(_THETA_STATIC_NO_PARALLAX, {"thS": 0.5})
    assert len(result) == 2
    thE, mu = result
    assert float(thE) == pytest.approx(0.5 / 0.005)
    assert float(mu) == pytest.approx(0.5 / 0.005 / 50.0 * 365.25)
    lndet = p.log_abs_det_jacobian(_THETA_STATIC_NO_PARALLAX, {"thS": 0.5})
    assert np.isfinite(float(lndet))


def test_impl_without_parallax_can_sample_distances():
    p = ParamType(parallax=False, distance="sample")
    assert p.names == ("t0", "tE", "u0", "rho", "DL", "DS")
    with pytest.raises(TypeError, match="to_mu_physical"):
        p.to_physical(_THETA_STATIC_NO_PARALLAX_SAMPLE_DISTANCE, {"thS": 0.5})
    result = p.to_mu_physical(_THETA_STATIC_NO_PARALLAX_SAMPLE_DISTANCE, {"thS": 0.5})
    assert len(result) == 4
    assert all(np.isfinite(float(v)) for v in result)
    ML, DL, DS, mu = result
    assert ML > 0.0
    assert DL == pytest.approx(4.0)
    assert DS == pytest.approx(8.0)
    expected_mu = 0.5 / 0.005 / 50.0 * 365.25
    assert float(mu) == pytest.approx(expected_mu)
    lndet = p.log_abs_det_jacobian(_THETA_STATIC_NO_PARALLAX_SAMPLE_DISTANCE, {"thS": 0.5})
    assert np.isfinite(float(lndet))


def test_from_model_spec_accepts_external_spec_like_objects():
    raw = SimpleNamespace(
        lens="binary",
        source="single",
        orbital_motion="static",
        xallarap="none",
        parallax=True,
    )
    p = from_model_spec(raw)
    assert isinstance(p, ParamType)
    assert p.names == ("t0", "tE", "u0", "rho", "piEN", "piEE", "DS")


def test_from_model_spec_accepts_light_curve_like_object():
    p = from_model_spec(SimpleNamespace(spec=ParamType(parallax=True)))
    assert isinstance(p, ParamType)
    assert p.names == ("t0", "tE", "u0", "rho", "piEN", "piEE", "DS")


def test_impl_lom_dispatches_for_binary_parallax_models():
    circular = ParamType(parallax=True, orbital_motion="circular")
    kepler = ParamType(parallax=True, orbital_motion="kepler")
    assert circular.names == BinaryCircularParamType().names
    assert kepler.names == BinaryKeplerParamType().names


@pytest.mark.parametrize("spec,match", [
    (dict(source="binary"), "source='single'"),
    (dict(xallarap="circular_elements"), "xallarap"),
    (dict(lens="triple", parallax=True, orbital_motion="circular"), "binary lenses"),
    (dict(parallax=False, orbital_motion="circular"), "parallax=True"),
    (dict(parallax=True, orbital_motion="circular", distance="marginalize"), "static"),
])
def test_impl_rejects_unsupported_specs(spec, match):
    with pytest.raises(NotImplementedError, match=match):
        ParamType(**spec)


@pytest.mark.parametrize("cls,theta,ctx", [
    (BinaryCircularParamType, _THETA_CIRC, _CTX),
    (BinaryCircularUseThEParamType, _THETA_CIRC_THE, _CTX),
    (BinaryKeplerParamType, _THETA_KEPLER, _CTX),
    (SingleLensParamType, _THETA_SINGLE, _CTX),
    (SingleLensUseThEParamType, _THETA_SINGLE_THE, _CTX),
])
def test_to_physical_returns_finite_scalars(cls, theta, ctx):
    p = cls()
    result = p.to_physical(theta, ctx)
    if cls is BinaryKeplerParamType:
        assert len(result) == 12
    elif cls in {BinaryCircularParamType, BinaryCircularUseThEParamType}:
        assert len(result) == 10
    else:
        assert len(result) == 5
    assert all(np.isfinite(float(v)) for v in result)


def test_circular_orbit_derived_parameters_are_exposed():
    p = ParamType(parallax=True, orbital_motion="circular")
    physical = p.to_physical(_THETA_CIRC, _CTX)
    derived = p.to_derived(_THETA_CIRC, _CTX)

    assert len(physical) == 10
    assert tuple(float(v) for v in physical[5:]) == pytest.approx(
        tuple(float(derived[name]) for name in p.derived_names)
    )
    assert p.derived_names == ("q", "orbital_radi", "cos_i", "Om_NE", "phi0")
    assert set(derived) == set(p.derived_names)
    assert all(np.isfinite(float(v)) for v in derived.values())


def test_kepler_orbit_derived_parameters_are_exposed():
    p = ParamType(parallax=True, orbital_motion="kepler")
    physical = p.to_physical(_THETA_KEPLER, _CTX)
    derived = p.to_derived(_THETA_KEPLER, _CTX)

    assert len(physical) == 12
    assert tuple(float(v) for v in physical[5:]) == pytest.approx(
        tuple(float(derived[name]) for name in p.derived_names)
    )
    assert p.derived_names == ("q", "orbital_radi", "e", "cos_i", "Om_NE", "om", "nu")
    assert set(derived) == set(p.derived_names)
    assert all(np.isfinite(float(v)) for v in derived.values())


def test_kepler_invalid_orbit_is_rejected_before_density():
    class CountingDensity:
        def __init__(self):
            self.calls = 0

        def log_density(self, ML, DL, DS, mu_N, mu_E):
            self.calls += 1
            return 0.0

    density = CountingDensity()
    prior = GalacticModel(
        density,
        param_type=ParamType(parallax=True, orbital_motion="kepler"),
        include_event_rate=False,
    )
    invalid = _THETA_KEPLER.at[13].set(0.1)

    assert prior.log_prob(invalid, context=_CTX) == float("-inf")
    assert density.calls == 0


@pytest.mark.parametrize("cls,theta,ctx", [
    (BinaryCircularParamType, _THETA_CIRC, _CTX),
    (BinaryCircularUseThEParamType, _THETA_CIRC_THE, _CTX),
    (BinaryKeplerParamType, _THETA_KEPLER, _CTX),
    (SingleLensParamType, _THETA_SINGLE, _CTX),
    (SingleLensUseThEParamType, _THETA_SINGLE_THE, _CTX),
])
def test_jacobian_is_finite(cls, theta, ctx):
    p = cls()
    lndet = p.log_abs_det_jacobian(theta, ctx)
    assert np.isfinite(float(lndet))


def test_missing_vEarth_raises():
    p = BinaryCircularParamType()
    with pytest.raises(ValueError, match="vEarth"):
        p.to_physical(_THETA_CIRC, {"thS": 0.5})


def test_missing_thS_raises():
    p = BinaryCircularParamType()
    with pytest.raises(ValueError, match="thS"):
        p.to_physical(_THETA_CIRC, {"vEarth": (28.0, -3.0)})


def test_thE_variant_does_not_need_thS():
    p = BinaryCircularUseThEParamType()
    ctx_no_thS = {"vEarth": (28.0, -3.0)}
    result = p.to_physical(_THETA_CIRC_THE, ctx_no_thS)
    assert len(result) == 10


def test_single_lens_ds_is_in_theta():
    p = SingleLensParamType()
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
    distance = SimpleNamespace(distance_pc=jnp.asarray([2000.0, 4000.0, 6000.0, 8000.0]))

    def log_density(self, ML, DL, DS, mu_N, mu_E):
        return -(ML + DL + DS + 0.01 * jnp.hypot(mu_N, mu_E))

    def log_density_mu(self, ML, DL, DS, mu):
        return -(ML + DL + DS + 0.01 * mu)

    def log_density_theta_mu(self, thetaE, mu, *, include_event_rate=False):
        return -(thetaE + 0.01 * mu)


class _DummyDensity:
    distance = SimpleNamespace(distance_pc=np.asarray([2000.0, 4000.0, 6000.0, 8000.0]))

    def log_density(self, ML, DL, DS, mu_N, mu_E):
        return -(ML + DL + DS + 0.01 * np.hypot(mu_N, mu_E))

    def log_density_mu(self, ML, DL, DS, mu):
        return -(ML + DL + DS + 0.01 * mu)

    def log_density_theta_mu(self, thetaE, mu, *, include_event_rate=False):
        return -(thetaE + 0.01 * mu)


def test_galactic_prior_accepts_impl_selector_directly():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=True),
        include_event_rate=False,
    )
    value = prior.log_prob(_THETA_SINGLE, context=_CTX)
    assert np.isfinite(float(value))


def test_galactic_prior_exposes_physical_and_transform_apis():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=True),
        include_event_rate=False,
    )

    physical = prior.to_physical(_THETA_SINGLE, context=_CTX)
    logp_physical = prior.log_prob_physical(physical)
    log_jacobian = prior.log_abs_det_jacobian(_THETA_SINGLE, context=_CTX)
    logp_theta = prior.log_prob(_THETA_SINGLE, context=_CTX)

    assert len(physical) == 5
    assert all(np.isfinite(float(v)) for v in physical)
    assert np.isfinite(logp_physical)
    assert np.isfinite(float(log_jacobian))
    assert logp_theta == pytest.approx(logp_physical + float(log_jacobian))


def test_galactic_prior_exposes_only_deterministic_values_for_static_parallax_marginalized_distance():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=True, distance="marginalize"),
        include_event_rate=False,
    )

    values = prior.to_deterministic_physical(_THETA_SINGLE[:-1], context=_CTX)

    assert set(values) == {"thetaE", "piE", "ML", "mu_N", "mu_E"}
    assert values["thetaE"] == pytest.approx(float(_CTX["thS"] / _THETA_SINGLE[3]))
    assert values["piE"] == pytest.approx(float(jnp.hypot(_THETA_SINGLE[4], _THETA_SINGLE[5])))
    assert "DL" not in values
    assert "DS" not in values


def test_galactic_prior_exposes_only_deterministic_values_for_no_parallax_marginalized_distance():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=False),
        include_event_rate=False,
    )

    values = prior.to_deterministic_physical(
        _THETA_STATIC_NO_PARALLAX,
        context={"thS": 0.5},
    )

    assert set(values) == {"thetaE", "mu"}
    assert values["thetaE"] == pytest.approx(100.0)
    assert values["mu"] == pytest.approx(730.5)
    assert "ML" not in values
    assert "DL" not in values
    assert "DS" not in values


def test_galactic_prior_samples_orbital_derived_parameters():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=True, orbital_motion="circular"),
        include_event_rate=False,
    )

    physical = prior.to_physical(_THETA_CIRC, context=_CTX)
    derived = prior.to_derived(_THETA_CIRC, context=_CTX)
    draw = prior.sample_physical(_THETA_CIRC, context=_CTX)

    assert len(physical) == 10
    assert tuple(physical[5:]) == pytest.approx(
        tuple(derived[name] for name in ["q", "orbital_radi", "cos_i", "Om_NE", "phi0"])
    )
    assert set(derived) == {"q", "orbital_radi", "cos_i", "Om_NE", "phi0"}
    assert set(derived).issubset(draw)
    assert draw["q"] == pytest.approx(float(_THETA_CIRC[4]))
    assert all(np.isfinite(float(draw[key])) for key in derived)


def test_galactic_prior_parallax_static_can_marginalize_ds():
    p = ParamType(parallax=True, distance="marginalize")
    assert p.names == ("t0", "tE", "u0", "rho", "piEN", "piEE")

    prior = GalacticModel(
        _DummyDensity(),
        param_type=p,
        include_event_rate=False,
    )
    theta = _THETA_SINGLE[:-1]
    value = prior.log_prob(theta, context=_CTX)

    assert np.isfinite(float(value))
    with pytest.raises(TypeError, match="full Jacobian"):
        prior.log_abs_det_jacobian(theta, context=_CTX)


def test_galactic_prior_parallax_static_samples_marginalized_ds():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=True, distance="marginalize"),
        include_event_rate=False,
    )
    rng = np.random.default_rng(1)

    draw = prior.sample_physical(_THETA_SINGLE[:-1], context=_CTX, rng=rng)

    assert set(draw) == {"ML", "DL", "DS", "mu_N", "mu_E"}
    assert draw["ML"] > 0.0
    assert 2.0 <= draw["DS"] <= 8.0
    assert 0.0 < draw["DL"] < draw["DS"]


def test_galactic_prior_parallax_static_samples_marginalized_ds_array():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=True, distance="marginalize"),
        include_event_rate=False,
    )
    theta = np.stack([np.asarray(_THETA_SINGLE[:-1]), np.asarray(_THETA_SINGLE[:-1])])

    draws = prior.sample_physical(theta, context=_CTX, rng=np.random.default_rng(1))

    assert set(draws) == {"ML", "DL", "DS", "mu_N", "mu_E"}
    for values in draws.values():
        assert values.shape == (2,)
    assert np.all(draws["ML"] > 0.0)
    assert np.all((2.0 <= draws["DS"]) & (draws["DS"] <= 8.0))
    assert np.all((0.0 < draws["DL"]) & (draws["DL"] < draws["DS"]))


def test_galactic_prior_without_impl_accepts_physical_values():
    prior = GalacticModel(_DummyDensity(), include_event_rate=False)
    physical = (1.0, 4.0, 8.0, 2.0, -1.0)

    assert prior.to_physical(physical) == physical
    assert prior.log_abs_det_jacobian(physical) == 0.0
    assert prior.log_prob(physical) == pytest.approx(prior.log_prob_physical(*physical))


def test_galactic_prior_no_parallax_marginalizes_distances_by_default():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=False),
        include_event_rate=False,
    )

    physical = prior.to_theta_mu_physical(_THETA_STATIC_NO_PARALLAX, context={"thS": 0.5})
    logp_theta_mu = prior.log_prob_theta_mu(physical)
    log_jacobian = prior.log_abs_det_jacobian(_THETA_STATIC_NO_PARALLAX, context={"thS": 0.5})
    logp_theta = prior.log_prob(_THETA_STATIC_NO_PARALLAX, context={"thS": 0.5})

    assert len(physical) == 2
    assert all(np.isfinite(float(v)) for v in physical)
    assert logp_theta == pytest.approx(logp_theta_mu + float(log_jacobian))
    with pytest.raises(TypeError, match="to_theta_mu_physical"):
        prior.to_physical(_THETA_STATIC_NO_PARALLAX, context={"thS": 0.5})


def test_galactic_prior_no_parallax_samples_marginalized_distances():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=False),
        include_event_rate=False,
    )
    rng = np.random.default_rng(2)

    draw = prior.sample_physical(
        _THETA_STATIC_NO_PARALLAX,
        context={"thS": 0.005},
        rng=rng,
    )

    assert set(draw) == {"ML", "DL", "DS", "mu"}
    assert draw["ML"] > 0.0
    assert 0.0 < draw["DL"] < draw["DS"]
    assert draw["mu"] == pytest.approx(0.005 / 0.005 / 50.0 * 365.25)


def test_galactic_prior_no_parallax_samples_marginalized_distances_array():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=False),
        include_event_rate=False,
    )
    theta = np.stack([
        np.asarray(_THETA_STATIC_NO_PARALLAX),
        np.asarray(_THETA_STATIC_NO_PARALLAX),
    ])

    draws = prior.sample_physical(
        theta,
        context={"thS": 0.005},
        rng=np.random.default_rng(2),
    )

    assert set(draws) == {"ML", "DL", "DS", "mu"}
    for values in draws.values():
        assert values.shape == (2,)
    assert np.all(draws["ML"] > 0.0)
    assert np.all((0.0 < draws["DL"]) & (draws["DL"] < draws["DS"]))
    assert np.allclose(draws["mu"], 0.005 / 0.005 / 50.0 * 365.25)


def test_galactic_prior_sample_physical_rejects_higher_rank_theta():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=False),
        include_event_rate=False,
    )

    with pytest.raises(ValueError, match="one-dimensional or two-dimensional"):
        prior.sample_physical(
            np.zeros((1, 1, len(_THETA_STATIC_NO_PARALLAX))),
            context={"thS": 0.005},
            rng=np.random.default_rng(2),
        )


def test_galactic_prior_no_parallax_can_sample_distances():
    prior = GalacticModel(
        _DummyDensity(),
        param_type=ParamType(parallax=False, distance="sample"),
        include_event_rate=False,
    )

    physical = prior.to_mu_physical(_THETA_STATIC_NO_PARALLAX_SAMPLE_DISTANCE, context={"thS": 0.5})
    logp_mu = prior.log_prob_mu(physical)
    log_jacobian = prior.log_abs_det_jacobian(
        _THETA_STATIC_NO_PARALLAX_SAMPLE_DISTANCE,
        context={"thS": 0.5},
    )
    logp_theta = prior.log_prob(_THETA_STATIC_NO_PARALLAX_SAMPLE_DISTANCE, context={"thS": 0.5})

    assert len(physical) == 4
    assert all(np.isfinite(float(v)) for v in physical)
    assert logp_theta == pytest.approx(logp_mu + float(log_jacobian))


def test_jax_prior_with_binary_impl_is_jittable():
    prior = MappedGalacticModel(
        _DummyJaxDensity(),
        param_type=BinaryCircularParamType(),
        include_event_rate=False,
    )

    @jax.jit
    def lp(theta):
        return prior.log_prob(theta, context=_CTX)

    value = lp(_THETA_CIRC)
    assert np.isfinite(float(value))


def test_jax_galactic_prior_exposes_physical_and_transform_apis():
    prior = MappedGalacticModel(
        _DummyJaxDensity(),
        param_type=ParamType(parallax=True),
        include_event_rate=False,
    )

    @jax.jit
    def pieces(theta):
        physical = prior.to_physical(theta, context=_CTX)
        return (
            prior.log_prob_physical(physical),
            prior.log_abs_det_jacobian(theta, context=_CTX),
            prior.log_prob(theta, context=_CTX),
        )

    logp_physical, log_jacobian, logp_theta = pieces(_THETA_SINGLE)
    assert np.isfinite(float(logp_physical))
    assert np.isfinite(float(log_jacobian))
    assert float(logp_theta) == pytest.approx(float(logp_physical + log_jacobian))


def test_jax_galactic_prior_parallax_static_can_marginalize_ds():
    prior = MappedGalacticModel(
        _DummyJaxDensity(),
        param_type=ParamType(parallax=True, distance="marginalize"),
        include_event_rate=False,
    )

    @jax.jit
    def lp(theta):
        return prior.log_prob(theta, context=_CTX)

    value = lp(_THETA_SINGLE[:-1])
    assert np.isfinite(float(value))


def test_jax_galactic_prior_no_parallax_marginalizes_distances_by_default():
    prior = MappedGalacticModel(
        _DummyJaxDensity(),
        param_type=ParamType(parallax=False),
        include_event_rate=False,
    )

    @jax.jit
    def pieces(theta):
        physical = prior.to_theta_mu_physical(theta, context={"thS": 0.5})
        return (
            prior.log_prob_theta_mu(physical),
            prior.log_abs_det_jacobian(theta, context={"thS": 0.5}),
            prior.log_prob(theta, context={"thS": 0.5}),
        )

    logp_theta_mu, log_jacobian, logp_theta = pieces(_THETA_STATIC_NO_PARALLAX)
    assert np.isfinite(float(logp_theta_mu))
    assert np.isfinite(float(log_jacobian))
    assert float(logp_theta) == pytest.approx(float(logp_theta_mu + log_jacobian))
