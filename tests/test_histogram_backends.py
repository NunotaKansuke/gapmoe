from __future__ import annotations

from math import cos, isfinite, log, sin
from pathlib import Path

import numpy as np
import pytest

from gapmoe.density import HistogramDensity
from gapmoe.density.histogram_tables import HistogramTables
from gapmoe.priors.cmd import CmdGalacticModel
from gapmoe.priors.galactic import GalacticModel
from gapmoe.priors.high_level import GalaxyModel as SourceAwareGalaxyModel, IsochroneModel
from gapmoe.priors.mapped import MappedGalacticModel
from gapmoe.priors.source import EventPrior5D, SourceCmdPrior
from gapmoe.source_selection import CmdCoordinates, CmdPriorTable


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "small_source_default"
POINT_MU_PHI = (0.3, 0.26, 0.6, 5.385164807134504, 0.3805063771123649)


@pytest.fixture(scope="module")
def histogram_density() -> HistogramDensity:
    return HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")


def raw_point() -> tuple[float, float, float, float, float]:
    ml, dl, ds, mu, phi = POINT_MU_PHI
    return ml, dl, ds, mu * cos(phi), mu * sin(phi)


def test_histogram_prior_is_finite(histogram_density: HistogramDensity) -> None:
    ml, dl, ds, mu_n, mu_e = raw_point()
    log_density = histogram_density.log_density(ml, dl, ds, mu_n, mu_e)
    log_prior = histogram_density.log_prior(ml, dl, ds, mu_n, mu_e)
    composed = GalacticModel(histogram_density).log_prob(ml, dl, ds, mu_n, mu_e)

    assert isfinite(log_density)
    assert isfinite(log_prior)
    assert log_prior == pytest.approx(composed)


def test_raw_component_density_has_expected_mu_jacobian(histogram_density: HistogramDensity) -> None:
    ml, dl, ds, mu, phi = POINT_MU_PHI
    mu_n = mu * cos(phi)
    mu_e = mu * sin(phi)

    raw_log_density = histogram_density.log_density(ml, dl, ds, mu_n, mu_e)
    mu_phi_log_density = histogram_density.log_density_mu_phi(ml, dl, ds, mu, phi)

    assert raw_log_density - mu_phi_log_density == pytest.approx(-log(mu))


def test_direction_marginalized_mu_density_is_finite(histogram_density: HistogramDensity) -> None:
    ml, dl, ds, mu, _ = POINT_MU_PHI
    log_density_mu = histogram_density.log_density_mu(ml, dl, ds, mu)

    assert isfinite(log_density_mu)


def test_distance_marginalized_theta_mu_density_is_finite(histogram_density: HistogramDensity) -> None:
    _, _, _, mu, _ = POINT_MU_PHI
    log_density_theta_mu = histogram_density.log_density_theta_mu(1.0, mu)

    assert isfinite(log_density_theta_mu)


def test_histogram_jits(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    ml, dl, ds, mu_n, mu_e = raw_point()
    numpy_log_prob = GalacticModel(histogram_density).log_prob(ml, dl, ds, mu_n, mu_e)
    jax_prior = MappedGalacticModel(histogram_density)
    jax_log_prob = float(jax_prior.log_prob(ml, dl, ds, mu_n, mu_e))

    jit_log_density = float(jax.jit(histogram_density.log_density)(ml, dl, ds, mu_n, mu_e))
    jit_log_prob = float(jax.jit(jax_prior.log_prob)(ml, dl, ds, mu_n, mu_e))

    assert np.isfinite(jax_log_prob)
    assert jax_log_prob == pytest.approx(numpy_log_prob, rel=1e-5, abs=1e-5)
    assert jit_log_density == pytest.approx(float(histogram_density.log_density(ml, dl, ds, mu_n, mu_e)), rel=1e-5, abs=1e-5)
    assert jit_log_prob == pytest.approx(numpy_log_prob, rel=1e-5, abs=1e-5)


def test_jax_cmd_joint_density_matches_numpy_and_jits(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gapmoe.density.histogram_backend import CmdPriorEvaluator

    cmd_prior = CmdPriorTable(
        coordinates=CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=np.asarray([0.0, 1.0]),
        color_edges=np.asarray([0.0, 1.0]),
        density_by_component=np.ones((11, 1, 1)),
    )
    jax_density = histogram_density
    jax_cmd = CmdPriorEvaluator.from_table(cmd_prior)
    values = raw_point()
    def evaluate(theta):
        return jax_density.cmd_joint_density(
            *theta,
            cmd_prior=jax_cmd,
            reference_magnitude=0.5,
            color=0.5,
            magnitude_offsets=jnp.zeros(3),
        )

    value = float(evaluate(jnp.asarray(values)))
    jitted = float(jax.jit(evaluate)(jnp.asarray(values)))
    assert np.isfinite(value)
    assert jitted == pytest.approx(value, rel=1e-5, abs=1e-5)


def test_cmd_galactic_model_extracts_source_photometry_from_mcmc_state(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gapmoe.density.histogram_backend import CmdPriorEvaluator

    cmd_prior = CmdPriorTable(
        coordinates=CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=np.asarray([0.0, 1.0]),
        color_edges=np.asarray([0.0, 1.0]),
        density_by_component=np.ones((11, 1, 1)),
    ).evaluator()
    source = SourceCmdPrior(
        density=histogram_density,
        cmd_prior=cmd_prior,
        offset_calculator=lambda ds, context: jnp.zeros(3),
    )
    model = CmdGalacticModel(
        event_prior=EventPrior5D(histogram_density, source, include_event_rate=False),
        cmd_extractor=lambda theta, context: (theta[5], theta[6]),
    )
    theta = jnp.asarray((*raw_point(), 0.5, 0.5))

    value = float(model.log_prob(theta))
    direct = float(
        model.event_prior.log_density(
            *theta[:5],
            reference_magnitude=theta[5],
            color=theta[6],
        )
    )
    assert value == pytest.approx(direct)
    assert float(jax.jit(model.log_prob)(theta)) == pytest.approx(direct)


def test_source_aware_model_uses_named_magnitudes_for_density_and_radius(histogram_density: HistogramDensity) -> None:
    log_radius = np.log(2.0)
    table = CmdPriorTable(
        coordinates=CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=np.asarray([-20.0, 0.0, 20.0]),
        color_edges=np.asarray([0.0, 1.0]),
        density_by_component=np.ones((11, 2, 1)),
        log_radius_moment_by_component=np.full((11, 2, 1), log_radius),
        log_radius_square_moment_by_component=np.full((11, 2, 1), 2.0 * log_radius**2),
    )
    model = SourceAwareGalaxyModel(
        density=histogram_density,
        isochrone=IsochroneModel("Imag", ("Vmag", "Imag"), table=table),
        l_deg=1.0,
        b_deg=-3.9,
        extinction_at_rc={},
        include_event_rate=False,
    )
    magnitudes = {"Imag": 0.5, "Vmag": 1.0}

    assert np.isfinite(float(model.log_source_density(ds=0.6, magnitudes=magnitudes)))
    estimate = model.source_radius(ds=0.6, magnitudes=magnitudes)
    expected_mean = np.exp(log_radius + 0.5 * log_radius**2)
    assert float(estimate.mean_rsun) == pytest.approx(expected_mean)
    assert float(estimate.std_rsun) > 0.0
    assert float(estimate.median_rsun) == pytest.approx(2.0)
    assert float(estimate.p16_rsun) == pytest.approx(1.0)
    assert float(estimate.p84_rsun) == pytest.approx(4.0)
    jax = pytest.importorskip("jax")
    assert float(jax.jit(lambda ds: model.source_radius(ds=ds, magnitudes=magnitudes).mean_rsun)(0.6)) == pytest.approx(expected_mean)


def test_event_prior_5d_conditions_on_cmd_without_applying_cmd_prior(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    cmd = CmdPriorTable(
        coordinates=CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
        reference_edges=np.asarray([0.0, 1.0]),
        color_edges=np.asarray([0.0, 1.0]),
        density_by_component=np.ones((11, 1, 1)),
    ).evaluator()
    source = SourceCmdPrior(
        density=histogram_density,
        cmd_prior=cmd,
        offset_calculator=lambda ds, context: jnp.zeros(3),
    )
    event = EventPrior5D(histogram_density, source, include_event_rate=False)
    values = raw_point()
    conditional = event.log_density(*values, reference_magnitude=0.5, color=0.5)
    joint = histogram_density.log_cmd_joint_density(
        *values,
        cmd_prior=cmd,
        reference_magnitude=0.5,
        color=0.5,
        magnitude_offsets=jnp.zeros(3),
    )
    marginal = source.log_marginal_density(0.5, 0.5)

    assert float(conditional + marginal) == pytest.approx(float(joint), rel=1e-5)
    assert float(jax.jit(event.log_density)(*values, reference_magnitude=0.5, color=0.5)) == pytest.approx(
        float(conditional), rel=1e-5
    )


def test_jax_histogram_bilinear_murel_is_finite_and_differentiable(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gapmoe.density.histogram_tables import HistogramTables

    _, _, _, mu_n, mu_e = raw_point()
    tables = HistogramTables.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    jax_density = HistogramDensity.from_tables(tables, murel_interpolation="bilinear")

    def log_prob(theta):
        p_mu, p_phi = jax_density.murel.densities(theta[0], theta[1], theta[2], theta[3])
        return jnp.log(p_mu * p_phi)

    theta = jnp.asarray([0.5, 0.9, jnp.hypot(mu_n, mu_e), jnp.arctan2(mu_e, mu_n)])
    value = jax.jit(log_prob)(theta)
    grad = jax.jit(jax.grad(log_prob))(theta)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


def test_histogram_tails_are_positive_normalised_and_match_jax(histogram_density: HistogramDensity) -> None:
    pytest.importorskip("jax")
    tables = HistogramTables.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")
    distance = histogram_density.distance.distance_pc
    mass = histogram_density.mass.log_mass

    for value in (10.0 ** (mass[0] - 0.1), 10.0 ** (mass[-1] + 0.1)):
        numpy_value = tables.mass.density_given_component(value).sum()
        jax_value = histogram_density.mass.density_given_component(value).sum()
        assert numpy_value > 0.0
        assert float(jax_value) == pytest.approx(numpy_value, rel=3e-4)
    for value in ((distance[0] - 1.0) / 1000.0, (distance[-1] + 100.0) / 1000.0):
        assert float(histogram_density.distance.source_pdf(value)) > 0.0

    grid = np.linspace(0.0, distance[-1] / 1000.0 + 50.0, 20_001)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert integrate([float(histogram_density.distance.source_pdf(value)) for value in grid], grid) == pytest.approx(1.0, rel=2e-3)


def test_histogram_physical_boundaries_remain_zero(histogram_density: HistogramDensity) -> None:
    assert float(histogram_density.density_mu_phi(0.0, 0.2, 0.6, 1.0, 0.0)) == 0.0
    assert float(histogram_density.density_mu_phi(0.3, 0.0, 0.6, 1.0, 0.0)) == 0.0
    assert float(histogram_density.density_mu_phi(0.3, 0.6, 0.6, 1.0, 0.0)) == 0.0
    assert float(histogram_density.density_mu_phi(0.3, 0.2, 0.6, 0.0, 0.0)) == 0.0


def test_jax_tail_uses_noncontiguous_positive_bins_like_numpy() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gapmoe.density.histogram_backend import _interp_positive_tail as jax_tail
    from gapmoe.density.histogram_tables import _interp_positive_tail as numpy_tail

    x = np.arange(8.0)
    y = np.asarray([0.0, 0.8, 0.0, 0.4, 0.2, 0.0, 0.1, 0.0])
    evaluate = jax.jit(lambda value: jax_tail(value, jnp.asarray(x), jnp.asarray(y)))
    for value in (-0.5, 6.5):
        assert float(evaluate(value)) == pytest.approx(numpy_tail(value, x, y), rel=3e-5)
