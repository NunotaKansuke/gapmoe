from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


FLOW_MVP = Path(__file__).resolve().parents[1] / "flow_mvp"
if str(FLOW_MVP) not in sys.path:
    sys.path.insert(0, str(FLOW_MVP))

from genulens_cli_samples import (  # noqa: E402
    physical_and_source_group_from_raw,
    valid_raw_event_mask,
)
from validate_rate_included_joint import (  # noqa: E402
    derived_observables,
    max_rank_correlation_difference,
    rank_columns,
)


def _raw_row(*, weight=1.0, ml=0.3, dl_pc=4000.0, ds_pc=8000.0, pi_e=0.1):
    return [
        weight, ml, dl_pc, ds_pc, 20.0, 0.5, pi_e, 0.06, 0.08,
        5.0, 0.0, 0.0, 99.0, 99.0, 8.0, 0.0, 0.0,
    ]


def test_raw_event_mask_matches_flow_physical_support():
    raw = np.asarray([
        _raw_row(),
        _raw_row(weight=0.0),
        _raw_row(dl_pc=8000.0),
        _raw_row(pi_e=0.0),
    ])

    valid = valid_raw_event_mask(raw)
    physical, weights, groups = physical_and_source_group_from_raw(raw)

    np.testing.assert_array_equal(valid, [True, False, False, False])
    assert physical.shape == (1, 5)
    np.testing.assert_allclose(weights, [1.0])
    np.testing.assert_array_equal(groups, [2])


def test_proper_motion_components_preserve_reported_magnitude():
    raw = np.asarray([_raw_row()])

    physical, _, _ = physical_and_source_group_from_raw(raw)

    assert np.hypot(physical[0, 3], physical[0, 4]) == 5.0


def test_validation_derived_observables_follow_microlensing_relations():
    physical = np.asarray([[1.0, 4.0, 8.0, 3.0, 4.0]])

    derived = derived_observables(physical)

    theta_e = np.sqrt(8.144 * (1.0 / 4.0 - 1.0 / 8.0))
    np.testing.assert_allclose(derived["DL_over_DS"], [0.5])
    np.testing.assert_allclose(derived["mu_rel"], [5.0])
    np.testing.assert_allclose(derived["theta_E"], [theta_e])
    np.testing.assert_allclose(derived["t_E_days"], [theta_e / 5.0 * 365.25])


def test_rank_correlation_validation_handles_ties_and_monotone_scaling():
    values = np.asarray([[1.0, 2.0], [1.0, 4.0], [3.0, 8.0], [4.0, 16.0]])

    np.testing.assert_allclose(rank_columns(values)[:, 0], [0.5, 0.5, 2.0, 3.0])
    assert max_rank_correlation_difference(values, values * [10.0, 0.1]) == 0.0
