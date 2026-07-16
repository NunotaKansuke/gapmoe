from __future__ import annotations

import numpy as np
import pytest

from gapmoe.priors.event_rate_backend import log_event_rate_backend, log_flow_kernel_rate_backend
from gapmoe.priors.source import EventPrior5D


class _BaseDensity:
    def log_density(self, ml, dl, ds, mu_n, mu_e):
        return 0.0


class _FlowDensity(_BaseDensity):
    event_rate_factor_includes_lens_area = True


class _RateIncludedFlowDensity(_FlowDensity):
    event_rate_included = True


def test_flow_rate_factor_excludes_lens_area_already_in_kernel():
    ml, dl, ds, mu = 0.4, 4.5, 8.8, 5.0
    full = float(log_event_rate_backend(ml, dl, ds, mu))
    flow = float(log_flow_kernel_rate_backend(ml, dl, ds, mu))

    assert full - flow == pytest.approx(2.0 * np.log(dl))


def test_event_prior_uses_kernel_compatible_rate_factor_for_flow_density():
    values = (0.4, 4.5, 8.8, 3.0, 4.0)
    flow = EventPrior5D(_FlowDensity(), None, include_event_rate=True)
    histogram = EventPrior5D(_BaseDensity(), None, include_event_rate=True)

    assert float(flow.log_density(*values)) == pytest.approx(
        float(log_flow_kernel_rate_backend(values[0], values[1], values[2], 5.0))
    )
    assert float(histogram.log_density(*values)) == pytest.approx(
        float(log_event_rate_backend(values[0], values[1], values[2], 5.0))
    )


def test_event_prior_does_not_double_weight_rate_included_flow():
    values = (0.4, 4.5, 8.8, 3.0, 4.0)
    prior = EventPrior5D(_RateIncludedFlowDensity(), None, include_event_rate=True)

    assert float(prior.log_density(*values)) == pytest.approx(0.0)
