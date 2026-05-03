from __future__ import annotations

from math import cos, isfinite, log, sin
from pathlib import Path

import numpy as np
import pytest

from gapmoe import GalacticPrior, HistogramDensity
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
    composed = GalacticPrior(histogram_density).log_prob(ml, dl, ds, mu_n, mu_e)

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


def test_jax_histogram_matches_numpy(histogram_density: HistogramDensity) -> None:
    jax = pytest.importorskip("jax")
    from gapmoe import JaxGalacticPrior, JaxHistogramDensity

    ml, dl, ds, mu_n, mu_e = raw_point()
    jax_density = JaxHistogramDensity.from_numpy(histogram_density)

    numpy_log_density = histogram_density.log_density(ml, dl, ds, mu_n, mu_e)
    jax_log_density = float(jax_density.log_density(ml, dl, ds, mu_n, mu_e))

    numpy_log_prob = GalacticPrior(histogram_density).log_prob(ml, dl, ds, mu_n, mu_e)
    jax_prior = JaxGalacticPrior(jax_density)
    jax_log_prob = float(jax_prior.log_prob(ml, dl, ds, mu_n, mu_e))

    jit_log_density = float(jax.jit(jax_density.log_density)(ml, dl, ds, mu_n, mu_e))
    jit_log_prob = float(jax.jit(jax_prior.log_prob)(ml, dl, ds, mu_n, mu_e))

    assert np.isfinite(jax_log_density)
    assert np.isfinite(jax_log_prob)
    assert jax_log_density == pytest.approx(numpy_log_density, rel=1e-5, abs=1e-5)
    assert jax_log_prob == pytest.approx(numpy_log_prob, rel=1e-5, abs=1e-5)
    assert jit_log_density == pytest.approx(numpy_log_density, rel=1e-5, abs=1e-5)
    assert jit_log_prob == pytest.approx(numpy_log_prob, rel=1e-5, abs=1e-5)
