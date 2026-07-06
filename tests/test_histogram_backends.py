from __future__ import annotations

from math import cos, isfinite, log, sin
from pathlib import Path

import numpy as np
import pytest

from gapmoe import GalacticModel, HistogramDensity
from gapmoe.density.histogram import HistogramDensity as CompatHistogramDensity
from gapmoe.density.histogram_numpy import HistogramDensity as NumpyHistogramDensity


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "small_source_default"
POINT_MU_PHI = (0.3, 0.26, 0.6, 5.385164807134504, 0.3805063771123649)


@pytest.fixture(scope="module")
def histogram_density() -> HistogramDensity:
    return HistogramDensity.from_paths(FIXTURE / "mass.dat", FIXTURE / "rho.dat", FIXTURE / "murel.dat")


def raw_point() -> tuple[float, float, float, float, float]:
    ml, dl, ds, mu, phi = POINT_MU_PHI
    return ml, dl, ds, mu * cos(phi), mu * sin(phi)


def test_public_histogram_imports_are_numpy_backend() -> None:
    assert HistogramDensity is NumpyHistogramDensity
    assert CompatHistogramDensity is NumpyHistogramDensity


def test_numpy_histogram_prior_is_finite(histogram_density: HistogramDensity) -> None:
    ml, dl, ds, mu_n, mu_e = raw_point()
    log_density = histogram_density.log_density(ml, dl, ds, mu_n, mu_e)
    log_prior = histogram_density.log_prior(ml, dl, ds, mu_n, mu_e)
    composed = GalacticModel(histogram_density).log_prob(ml, dl, ds, mu_n, mu_e)

    assert isfinite(log_density)
    assert isfinite(log_prior)
    assert log_prior == pytest.approx(composed)


def test_numpy_histogram_array_density_matches_scalar(histogram_density: HistogramDensity) -> None:
    ml, dl, ds, mu_n, mu_e = raw_point()

    log_density = histogram_density.log_density_array(
        np.asarray([ml, ml]),
        np.asarray([dl, dl]),
        np.asarray([ds, ds]),
        np.asarray([mu_n, mu_n]),
        np.asarray([mu_e, mu_e]),
    )

    assert log_density.shape == (2,)
    assert np.allclose(log_density, histogram_density.log_density(ml, dl, ds, mu_n, mu_e))


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


def test_jax_histogram_matches_numpy(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    from gapmoe import JaxGalacticModel, JaxHistogramDensity

    ml, dl, ds, mu_n, mu_e = raw_point()
    jax_density = JaxHistogramDensity.from_numpy(histogram_density)

    numpy_log_density = histogram_density.log_density(ml, dl, ds, mu_n, mu_e)
    numpy_log_density_mu = histogram_density.log_density_mu(ml, dl, ds, (mu_n**2 + mu_e**2) ** 0.5)
    numpy_log_density_theta_mu = histogram_density.log_density_theta_mu(1.0, (mu_n**2 + mu_e**2) ** 0.5)
    jax_log_density = float(jax_density.log_density(ml, dl, ds, mu_n, mu_e))
    jax_log_density_mu = float(jax_density.log_density_mu(ml, dl, ds, (mu_n**2 + mu_e**2) ** 0.5))
    jax_log_density_theta_mu = float(jax_density.log_density_theta_mu(1.0, (mu_n**2 + mu_e**2) ** 0.5))

    numpy_log_prob = GalacticModel(histogram_density).log_prob(ml, dl, ds, mu_n, mu_e)
    jax_prior = JaxGalacticModel(jax_density)
    jax_log_prob = float(jax_prior.log_prob(ml, dl, ds, mu_n, mu_e))

    jit_log_density = float(jax.jit(jax_density.log_density)(ml, dl, ds, mu_n, mu_e))
    jit_log_prob = float(jax.jit(jax_prior.log_prob)(ml, dl, ds, mu_n, mu_e))

    assert np.isfinite(jax_log_density)
    assert np.isfinite(jax_log_density_mu)
    assert np.isfinite(jax_log_density_theta_mu)
    assert np.isfinite(jax_log_prob)
    assert jax_log_density == pytest.approx(numpy_log_density, rel=1e-5, abs=1e-5)
    assert jax_log_density_mu == pytest.approx(numpy_log_density_mu, rel=1e-5, abs=1e-5)
    assert jax_log_density_theta_mu == pytest.approx(numpy_log_density_theta_mu, rel=1e-5, abs=1e-5)
    assert jax_log_prob == pytest.approx(numpy_log_prob, rel=1e-5, abs=1e-5)
    assert jit_log_density == pytest.approx(numpy_log_density, rel=1e-5, abs=1e-5)
    assert jit_log_prob == pytest.approx(numpy_log_prob, rel=1e-5, abs=1e-5)


def test_histogram_tails_are_positive_normalised_and_match_jax(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    from gapmoe import JaxHistogramDensity

    numpy_density = histogram_density
    jax_density = JaxHistogramDensity.from_numpy(numpy_density)
    distance = numpy_density.distance.distance_pc
    mass = numpy_density.mass.log_mass

    # Just outside either finite table edge remains finite and positive.
    for value in (10.0 ** (mass[0] - 0.1), 10.0 ** (mass[-1] + 0.1)):
        assert numpy_density.mass.density_given_component(value).sum() > 0.0
        assert float(jax_density.mass.density_given_component(value).sum()) == pytest.approx(
                numpy_density.mass.density_given_component(value).sum(), rel=3e-4
        )
    for value in ((distance[0] - 1.0) / 1000.0, (distance[-1] + 100.0) / 1000.0):
        assert numpy_density.distance.source_pdf(value) > 0.0
        assert float(jax_density.distance.source_pdf(value)) == pytest.approx(
            numpy_density.distance.source_pdf(value), rel=1e-5
        )
    mu_max = numpy_density.murel.rows[:, 2].max()
    numpy_mu, _ = numpy_density.murel.densities(0.26, 0.6, mu_max + 1.0, 0.0)
    jax_mu, _ = jax_density.murel.densities(0.26, 0.6, mu_max + 1.0, 0.0)
    assert numpy_mu > 0.0
    assert float(jax_mu) == pytest.approx(numpy_mu, rel=3e-4)

    # The source PDF includes both tails in its normalisation.
    grid = np.linspace(0.0, distance[-1] / 1000.0 + 50.0, 20_001)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert integrate([numpy_density.distance.source_pdf(value) for value in grid], grid) == pytest.approx(1.0, rel=2e-3)


def test_histogram_physical_boundaries_remain_zero(histogram_density: HistogramDensity) -> None:
    assert histogram_density.density_mu_phi(0.0, 0.2, 0.6, 1.0, 0.0) == 0.0
    assert histogram_density.density_mu_phi(0.3, 0.0, 0.6, 1.0, 0.0) == 0.0
    assert histogram_density.density_mu_phi(0.3, 0.6, 0.6, 1.0, 0.0) == 0.0
    assert histogram_density.density_mu_phi(0.3, 0.2, 0.6, 0.0, 0.0) == 0.0


def test_jax_tail_uses_noncontiguous_positive_bins_like_numpy() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gapmoe.density.histogram_jax import _interp_positive_tail as jax_tail
    from gapmoe.density.histogram_numpy import _interp_positive_tail as numpy_tail

    x = np.arange(8.0)
    y = np.asarray([0.0, 0.8, 0.0, 0.4, 0.2, 0.0, 0.1, 0.0])
    evaluate = jax.jit(lambda value: jax_tail(value, jnp.asarray(x), jnp.asarray(y)))
    for value in (-0.5, 6.5):
        assert float(evaluate(value)) == pytest.approx(numpy_tail(value, x, y), rel=3e-5)


def test_jax_histogram_bilinear_murel_is_finite_and_differentiable(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gapmoe import JaxHistogramDensity

    _, _, _, mu_n, mu_e = raw_point()
    jax_density = JaxHistogramDensity.from_numpy(histogram_density, murel_interpolation="bilinear")

    def log_prob(theta):
        p_mu, p_phi = jax_density.murel.densities(theta[0], theta[1], theta[2], theta[3])
        return jnp.log(p_mu * p_phi)

    theta = jnp.asarray([0.5, 0.9, jnp.hypot(mu_n, mu_e), jnp.arctan2(mu_e, mu_n)])
    value = jax.jit(log_prob)(theta)
    grad = jax.jit(jax.grad(log_prob))(theta)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))
