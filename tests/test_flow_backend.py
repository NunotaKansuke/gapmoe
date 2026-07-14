from __future__ import annotations

import jax
import numpy as np

from gapmoe import Model
from gapmoe.priors.high_level import IsochroneModel
from gapmoe.source_selection import CmdCoordinates, CmdPriorTable


def _isochrone(*, selected: bool = False) -> IsochroneModel:
    reference_edges = np.linspace(-8.0, 20.0, 57)
    color_edges = np.linspace(-2.0, 8.0, 41)
    density = np.full((11, 56, 40), 1.0 / (28.0 * 10.0))
    return IsochroneModel(
        reference_band="Imag",
        color_bands=("Vmag", "Imag"),
        magnitude_range=(15.0, 21.0) if selected else None,
        color_range=(0.5, 3.0) if selected else None,
        table=CmdPriorTable(
            coordinates=CmdCoordinates("Imag", "Vmag", "Imag"),
            reference_edges=reference_edges,
            color_edges=color_edges,
            density_by_component=density,
            log_radius_moment_by_component=np.zeros_like(density),
            log_radius_square_moment_by_component=np.zeros_like(density),
            component_indices=np.arange(11),
        ),
    )


def test_bundled_flow_runs_the_complete_source_aware_api():
    model = Model().set(
        l=0.25,
        b=-3.75,
        extinction={"Imag": 1.2, "Vmag": 2.0},
    ).set_flow()
    prior = model.galactic_model(_isochrone(selected=True))
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
    model = Model().set(l=0.25, b=-3.75).set_flow()
    prior = model.galactic_model(_isochrone())
    compiled = jax.jit(prior.log_density)

    value = compiled(np.asarray((0.3, 4.0, 8.0, 3.0, -2.0)))

    assert np.isfinite(value)


def test_bundled_flow_samples_the_full_source_aware_prior():
    model = Model().set(
        l=0.25,
        b=-3.75,
        extinction={"Imag": 1.2, "Vmag": 2.0},
    ).set_flow()
    prior = model.galactic_model(_isochrone(selected=True))

    selected = np.asarray(prior.sample(jax.random.key(1)))
    conditioned = np.asarray(
        prior.sample(jax.random.key(2), magnitudes={"Imag": 18.0, "Vmag": 20.0})
    )

    for sample in (selected, conditioned):
        assert sample.shape == (5,)
        assert sample[0] > 0.0
        assert 0.0 < sample[1] < sample[2]
        assert np.isfinite(sample).all()
