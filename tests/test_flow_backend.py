from __future__ import annotations

import jax
import numpy as np
import pytest

from gapmoe import Flow, Isochrone, Model, ParamType
from gapmoe.density import EventKernelFlow
from gapmoe.flow_package import FlowPackage
from gapmoe.priors.parameterized import _KAPPA, _mass_proposal_geometry
from gapmoe.source_selection import CmdCoordinates, CmdPriorTable


def _isochrone(*, selected: bool = False, source_radius: bool = False) -> Isochrone:
    reference_edges = np.linspace(-8.0, 20.0, 57)
    color_edges = np.linspace(-2.0, 8.0, 41)
    density = np.full((11, 56, 40), 1.0 / (28.0 * 10.0))
    mean_log_radius = np.log(8.0)
    variance_log_radius = 0.3**2
    return Isochrone(
        reference_band="Imag",
        color_bands=("Vmag", "Imag"),
        magnitude_range=(15.0, 21.0) if selected else None,
        color_range=(0.5, 3.0) if selected else None,
        table=CmdPriorTable(
            coordinates=CmdCoordinates("Imag", "Vmag", "Imag"),
            reference_edges=reference_edges,
            color_edges=color_edges,
            density_by_component=density,
            log_radius_moment_by_component=(
                density * mean_log_radius if source_radius else np.zeros_like(density)
            ),
            log_radius_square_moment_by_component=(
                density * (mean_log_radius**2 + variance_log_radius)
                if source_radius else np.zeros_like(density)
            ),
            component_indices=np.arange(11),
        ),
    )


def _model(
    param_type=None,
    *,
    source=None,
    release="rate-included-v1",
    extinction=None,
    **options,
):
    return Model(
        param_type or ParamType(parallax=True, distance="sample"),
        l=0.25,
        b=-3.75,
        extinction={} if extinction is None else extinction,
        source=_isochrone() if source is None else source,
        backend=Flow(release),
        **options,
    )


def _distance_sensitive_isochrone(*, selected):
    reference_edges = np.linspace(-8.0, 20.0, 57)
    color_edges = np.linspace(-2.0, 8.0, 41)
    reference = 0.5 * (reference_edges[:-1] + reference_edges[1:])
    color = 0.5 * (color_edges[:-1] + color_edges[1:])
    density = np.exp(
        -0.5 * ((reference[:, None] - 4.0) / 0.4) ** 2
        -0.5 * ((color[None, :] - 1.0) / 0.3) ** 2
    )
    density /= np.sum(density) * np.diff(reference_edges)[0] * np.diff(color_edges)[0]
    density = np.broadcast_to(density, (11, *density.shape)).copy()
    return Isochrone(
        reference_band="Imag",
        color_bands=("Vmag", "Imag"),
        magnitude_range=(18.0, 19.0) if selected else None,
        color_range=(0.5, 1.5) if selected else None,
        table=CmdPriorTable(
            coordinates=CmdCoordinates("Imag", "Vmag", "Imag"),
            reference_edges=reference_edges,
            color_edges=color_edges,
            density_by_component=density,
            component_indices=np.arange(11),
        ),
    )


def test_bundled_flow_runs_the_complete_source_aware_api():
    model = _model(
        source=_isochrone(selected=True),
        release="default",
        extinction={"Imag": 1.2, "Vmag": 2.0},
    )
    prior = model.physical
    theta = np.asarray((0.3, 4.0, 8.0, 3.0, -2.0))
    magnitudes = {"Imag": 18.0, "Vmag": 20.0}

    assert np.isfinite(prior.log_density(theta))
    assert np.isfinite(prior.log_density(theta, magnitudes=magnitudes))
    assert np.isfinite(prior.log_source_density(ds=8.0, magnitudes=magnitudes))
    assert np.isfinite(prior.source_radius(ds=8.0, magnitudes=magnitudes).mean_rsun)

    sample = np.asarray(prior.sample_kernel(jax.random.key(0), ds=8.0, source_group=2))
    assert sample.shape == (5,)
    assert sample[0] > 0.0
    assert 0.0 < sample[1] < sample[2]
    assert sample[2] == 8.0


def test_bundled_flow_density_is_jittable():
    prior = _model(release="default").physical
    compiled = jax.jit(prior.log_density)

    value = compiled(np.asarray((0.3, 4.0, 8.0, 3.0, -2.0)))

    assert np.isfinite(value)


def test_source_brightness_range_reweights_source_distance_without_extinction():
    baseline = _model(source=_distance_sensitive_isochrone(selected=False)).physical
    selected = _model(source=_distance_sensitive_isochrone(selected=True)).physical

    def selection_log_weight(ds):
        theta = np.asarray((0.3, 0.5 * ds, ds, 3.0, -2.0))
        return float(selected.log_density(theta) - baseline.log_density(theta))

    near_source = selection_log_weight(8.0)
    foreground_source = selection_log_weight(3.0)

    assert np.isfinite(near_source)
    assert near_source > foreground_source + 5.0


def test_bundled_flow_parameterizes_and_marginalizes_source_distance():
    prior = _model(
        ParamType(parallax=True, distance="marginalize")
    )
    theta = np.asarray((8000.0, 50.0, 0.1, 0.005, 0.1, 0.05))
    context = {"thS": 0.005, "vEarth": (0.0, 0.0)}

    value = jax.jit(prior.log_density)(theta, context=context)
    physical = prior.to_deterministic_physical(theta, context=context)

    assert prior.names == ("t0", "tE", "u0", "rho", "piEN", "piEE")
    assert np.isfinite(value)
    assert set(physical) == {"thetaE", "piE", "ML", "mu_N", "mu_E"}


def test_parameterized_flow_applies_prior_inside_source_distance_integral():
    baseline = _model(ParamType(parallax=True, distance="marginalize"))
    constrained = _model(ParamType(parallax=True, distance="marginalize"))

    @constrained.prior
    def _(DS, **params):
        del params
        return jax.numpy.where(DS >= 6.0, 0.0, -jax.numpy.inf)

    theta = np.asarray((8000.0, 50.0, 0.1, 0.005, 0.1, 0.05))
    context = {"thS": 0.005, "vEarth": (0.0, 0.0)}
    baseline_value = baseline.log_density(theta, context=context)
    constrained_value = constrained.log_density(theta, context=context)
    draw = constrained.sample_physical(
        theta,
        context=context,
        rng=np.random.default_rng(3),
    )

    assert np.isfinite(constrained_value)
    assert constrained_value < baseline_value
    assert draw["DS"] >= 6.0


def test_parameterized_flow_evaluates_joint_source_photometry():
    prior = _model(
        ParamType(parallax=True, distance="sample"),
        extinction={"Imag": 1.2, "Vmag": 2.0},
    )
    theta = np.asarray((8000.0, 50.0, 0.1, 0.005, 0.1, 0.05, 8.0))
    context = {"thS": 0.005, "vEarth": (0.0, 0.0)}

    value = prior.log_joint_density(
        theta,
        context=context,
        magnitudes={"Imag": 18.0, "Vmag": 20.0},
    )

    assert np.isfinite(value)


def test_source_magnitudes_condition_physical_density_without_cmd_prior():
    physical = _model(
        extinction={"Imag": 1.2, "Vmag": 2.0},
    ).physical
    theta = np.asarray((0.3, 4.0, 8.0, 3.0, -2.0))
    magnitudes = {"Imag": 18.0, "Vmag": 20.0}
    reference, color = physical.isochrone.values_from_magnitudes(magnitudes)

    conditional = physical.log_density(theta, magnitudes=magnitudes)
    joint = physical.log_joint_density(theta, magnitudes=magnitudes)
    log_cmd = physical._conditional_prior.source_prior.log_marginal_density(
        reference,
        color,
    )

    assert conditional == pytest.approx(float(joint - log_cmd), rel=1.0e-6)


def test_no_parallax_flow_marginalizes_distances_with_fixed_importance_points():
    prior = _model(
        ParamType(parallax=False),
        integration_samples=32,
        seed=2,
    )
    theta = np.asarray((8000.0, 50.0, 0.1, 0.005))
    context = {"thS": 0.005}

    first = prior.log_density(theta, context=context)
    second = prior.log_density(theta, context=context)
    draw = prior.sample_physical(
        theta,
        context=context,
        rng=np.random.default_rng(4),
    )

    assert np.isfinite(first)
    assert first == second
    assert draw["ML"] > 0.0
    assert 0.0 < draw["DL"] < draw["DS"]
    assert np.hypot(draw["mu_N"], draw["mu_E"]) == pytest.approx(draw["mu"])


def test_no_parallax_mass_proposal_preserves_the_physical_jacobian():
    prior = _model(
        ParamType(parallax=False),
        integration_samples=32,
        seed=2,
    )
    proposal = prior._model._proposal
    theta_e = 0.5
    mu = 4.0

    ml, dl, ds, combined_jacobian = _mass_proposal_geometry(
        proposal, theta_e, mu
    )
    x = np.asarray(dl / ds)
    pi_rel = 1.0 / np.asarray(dl) - 1.0 / np.asarray(ds)
    old_jacobian = (
        np.log(2.0 * theta_e / (_KAPPA * pi_rel))
        + np.log(np.asarray(ds))
        + np.log(mu)
    )
    dmass_dx = theta_e**2 * np.asarray(ds) / (_KAPPA * (1.0 - x) ** 2)

    assert np.all(np.asarray(ml) > 0.0)
    assert np.all(np.isfinite(np.asarray(proposal.log_q_mass)))
    assert np.all((x > 0.0) & (x < 1.0))
    np.testing.assert_allclose(
        np.asarray(combined_jacobian),
        old_jacobian - np.log(dmass_dx),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_no_parallax_flow_jointly_integrates_isochrone_source_radius():
    prior = _model(
        ParamType(parallax=False),
        source=_isochrone(source_radius=True),
        extinction={"Imag": 1.2, "Vmag": 2.0},
        integration_samples=64,
        seed=2,
    )
    galaxy = prior.physical
    theta = np.asarray((8000.0, 50.0, 0.1, 0.005))
    magnitudes = {"Imag": 18.0, "Vmag": 20.0}

    direct_matched = galaxy.log_theta_star_density(
        theta_star_mas=0.004650467260962157,
        ds=8.0,
        magnitudes=magnitudes,
    )
    direct_mismatched = galaxy.log_theta_star_density(
        theta_star_mas=0.04,
        ds=8.0,
        magnitudes=magnitudes,
    )
    log_theta_center = np.log(0.004650467260962157)
    log_theta_grid = np.linspace(
        log_theta_center - 6.0 * 0.3,
        log_theta_center + 6.0 * 0.3,
        401,
    )
    theta_density = jax.vmap(
        lambda log_theta: jax.numpy.exp(
            galaxy.log_theta_star_density(
                theta_star_mas=jax.numpy.exp(log_theta),
                ds=8.0,
                magnitudes=magnitudes,
            )
        )
    )(jax.numpy.asarray(log_theta_grid))

    proposal = galaxy._theta_star_proposal(magnitudes=magnitudes)
    conditional = prior._isochrone_conditional_terms(
        theta,
        magnitudes={
            band: np.full(prior.integration_samples, value)
            for band, value in magnitudes.items()
        },
    )

    assert proposal.log_center == pytest.approx(log_theta_center, abs=0.1)
    assert proposal.log_sigma >= 0.3
    assert np.isfinite(np.asarray(conditional["log_terms"])).any()
    assert np.asarray(conditional["physical"]["thetaS"]).shape == (64,)
    assert direct_matched > direct_mismatched
    assert np.trapezoid(np.asarray(theta_density), log_theta_grid) == pytest.approx(
        1.0, rel=1.0e-4
    )


@pytest.mark.parametrize(
    "param_type,theta,context,expected",
    [
        (
            ParamType(parallax=True, distance="marginalize"),
            np.asarray((8000.0, 50.0, 0.1, 0.005, 0.1, 0.05)),
            {"vEarth": (0.0, 0.0)},
            {"ML", "DL", "DS", "mu_N", "mu_E", "thetaS"},
        ),
        (
            ParamType(parallax=False, distance="sample"),
            np.asarray((8000.0, 50.0, 0.1, 0.005, 4.0, 8.0)),
            None,
            {"ML", "DL", "DS", "mu_N", "mu_E", "thetaS"},
        ),
        (
            ParamType(parallax=True, orbital_motion="kepler"),
            np.asarray((
                8000.0, 50.0, 0.1, 0.005, 0.1, 1.0, 0.5,
                0.1, 0.05, -0.0001, -0.0001, -0.01, 0.1, 1.1,
            )),
            {"vEarth": (0.0, 0.0)},
            {"ML", "DL", "DS", "mu_N", "mu_E", "thetaS", "e", "cos_i"},
        ),
    ],
)
def test_isochrone_conditional_terms_cover_parameterization_modes(
    param_type, theta, context, expected
):
    prior = _model(
        param_type,
        source=_isochrone(source_radius=True),
        extinction={"Imag": 1.2, "Vmag": 2.0},
        integration_samples=16,
        seed=7,
    )
    result = prior._isochrone_conditional_terms(
        theta,
        magnitudes={"Imag": 18.0, "Vmag": 20.0},
        context=context,
    )

    assert np.isfinite(np.asarray(result["log_terms"])).any()
    assert expected <= set(result["physical"])
    assert all(np.asarray(value).shape == (16,) for value in result["physical"].values())


def test_no_parallax_importance_integral_applies_hidden_physical_prior():
    prior = _model(
        ParamType(parallax=False),
        integration_samples=64,
        seed=3,
    )

    @prior.prior
    def _(DS, **params):
        del params
        return jax.numpy.where(DS >= 6.0, 0.0, -jax.numpy.inf)

    theta = np.asarray((8000.0, 50.0, 0.1, 0.005))
    context = {"thS": 0.005}
    value = prior.log_density(theta, context=context)
    draw = prior.sample_physical(
        theta,
        context=context,
        rng=np.random.default_rng(7),
    )

    assert np.isfinite(value)
    assert draw["DS"] >= 6.0


def test_no_parallax_flow_samples_distances_and_integrates_direction():
    prior = _model(
        ParamType(parallax=False, distance="sample"),
        direction_samples=16,
    )
    theta = np.asarray((8000.0, 50.0, 0.1, 0.005, 4.0, 8.0))
    context = {"thS": 0.005}

    value = prior.log_density(theta, context=context)
    draw = prior.sample_physical(
        theta,
        context=context,
        rng=np.random.default_rng(5),
    )

    assert np.isfinite(value)
    assert draw["DL"] == pytest.approx(4.0)
    assert draw["DS"] == pytest.approx(8.0)
    assert np.hypot(draw["mu_N"], draw["mu_E"]) == pytest.approx(draw["mu"])


def test_bundled_flow_samples_the_full_source_aware_prior():
    prior = _model(
        source=_isochrone(selected=True),
        release="default",
        extinction={"Imag": 1.2, "Vmag": 2.0},
    ).physical

    selected = np.asarray(prior.sample(jax.random.key(1)))
    conditioned = np.asarray(
        prior.sample(jax.random.key(2), magnitudes={"Imag": 18.0, "Vmag": 20.0})
    )

    for sample in (selected, conditioned):
        assert sample.shape == (5,)
        assert sample[0] > 0.0
        assert 0.0 < sample[1] < sample[2]
        assert np.isfinite(sample).all()


def test_bundled_rate_included_flow_runs_without_double_rate_weighting():
    prior = _model().physical
    theta = np.asarray((0.3, 4.0, 8.0, 3.0, -2.0))

    value = prior.log_density(theta)
    sample = np.asarray(prior.sample(jax.random.key(4), num_proposals=1))

    assert np.isfinite(value)
    assert sample.shape == (5,)
    assert np.isfinite(sample).all()


def test_rate_included_flow_cannot_remove_rate_factor():
    with np.testing.assert_raises_regex(ValueError, "cannot remove"):
        _model(include_event_rate=False)


def test_rate_included_flow_loads_source_group_experts():
    package = FlowPackage.bundled("rate-included-v1")
    kernel = EventKernelFlow.load(package.event_kernel_path)

    assert set(kernel.group_overrides) == {3, 4}

    key = jax.random.key(17)
    for group in (2, 3, 4):
        condition = np.concatenate(([0.25, -3.75, 8.0], np.eye(5)[group]))
        expected_model = kernel.group_overrides.get(group, kernel)
        expected = expected_model._sample_single(key, condition)
        np.testing.assert_allclose(kernel.sample(key, condition), expected)
        values = np.asarray((0.3, 4.0, 3.0, -2.0))
        np.testing.assert_allclose(
            kernel.log_density(values, condition),
            expected_model._log_density_single(values, condition),
        )
