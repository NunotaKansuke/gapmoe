from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gapmoe.flow_source_grid import FlowSourceDistanceGrid


def test_source_distance_grid_interpolates_sightline_and_distance():
    l = jnp.asarray([0.0, 2.0])
    b = jnp.asarray([-4.0, -2.0])
    distance = jnp.asarray([1000.0, 2000.0])
    values = np.zeros((2, 2, 2, 11))
    for ib in range(2):
        for il in range(2):
            values[ib, il, :, :] = ib * 100.0 + il * 10.0 + np.asarray([[1.0], [3.0]])
    grid = FlowSourceDistanceGrid(l, b, distance, jnp.asarray(values))

    density = grid.at(1.0, -3.0)
    components = density.distance.source_component_values(1.5)

    assert np.allclose(np.asarray(components), 57.0)
    assert density.distance.source_norm == pytest.approx(627.0)


def test_source_distance_grid_rejects_extrapolation():
    grid = FlowSourceDistanceGrid(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([-2.0, -1.0]),
        jnp.asarray([1000.0, 2000.0]),
        jnp.ones((2, 2, 2, 11)),
    )

    with pytest.raises(ValueError, match="outside"):
        grid.at(2.0, -1.5)


def test_source_distance_grid_builds_raw_density_from_rho_nms():
    grid = FlowSourceDistanceGrid.from_rho_profiles(
        l_deg=np.asarray([0.0]),
        b_deg=np.asarray([-2.0]),
        distance_pc=np.asarray([1000.0, 2000.0]),
        nms_by_sightline=np.ones((1, 1, 2, 11)),
    )

    values = np.asarray(grid.source_by_component)
    assert np.allclose(values[0, 0, 0], 1.0)
    assert np.allclose(values[0, 0, 1], 4.0)


def test_flow_samples_keep_lens_strictly_in_front_of_source():
    from gapmoe.density.flow_backend import ResidualTransform

    transform = ResidualTransform(mean=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0))
    values = transform.from_unconstrained(
        jnp.asarray((0.0, 100.0, 0.0, 0.0), dtype=jnp.float32),
        jnp.asarray((0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
    )
    assert float(values[1]) < 8.0
    assert float(1.0 / values[1] - 1.0 / jnp.float32(8.0)) > 0.0
