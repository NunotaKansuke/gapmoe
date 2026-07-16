# gapmoe

gapmoe provides Galactic priors for microlensing event modeling. The public
interface is deliberately small: configure a sightline with `Model`, build an
isochrone source model, and obtain a five-dimensional event prior.

The canonical parameter order is:

```text
ML, DL, DS, mu_N, mu_E
```

Mass is in solar masses, distances are in kpc, and proper motions are in
mas/yr.

## Flow release

The bundled `default` Flow release needs no local genulens checkout:

```python
import gapmoe

model = gapmoe.Model()
model.set(l=1.0, b=-3.9, extinction={"Imag": 1.2, "Vmag": 2.0})
model.set_flow()

isochrone = model.isochrone(
    reference_band="Imag",
    color_bands=("Vmag", "Imag"),
    magnitude_range=(15.0, 21.0),
    color_range=(0.5, 3.0),
)
prior = model.galactic_model(isochrone)

logp = prior.log_density(theta)
logp_given_photometry = prior.log_density(
    theta, magnitudes={"Imag": i_s, "Vmag": v_s}
)
sample = prior.sample(key)
sample_given_photometry = prior.sample(
    key, magnitudes={"Imag": i_s, "Vmag": v_s}
)
```

The release covers `-5 <= l <= 5` and `-6 <= b <= -2` degrees with
`REMNANT=0` and `BINARY=0`. It models
`p(ML, DL, mu_E, mu_N | DS, source_group, l, b)`. The packaged source-distance
measure matches the unselected genulens proposal used to train this release:
`nMS * sqrt(DS / 8000) * 1e-3` times the integrated `DL**2`-weighted total
lens-number-density column to `DS` (the default `gammaDs=0.5`). The Flow
supplies lens parameters conditional on that source and already retains the
`DL**2` lens-area factor; inference applies only `thetaE * mu_rel`. CMD
selection and supplied source photometry are applied at inference, so neither
requires retraining.

`sample_kernel(key, ds=..., source_group=...)` is the fixed-`DS`, fixed-group
diagnostic sampler. Source-group indices are thin, thick, bulge, NSD, and halo
in that order (0 through 4). `sample()` draws a source distance/group first;
with the default event-rate option it uses importance resampling. This
partially rate-removed release is intended for prior evaluation and modest diagnostic
sampling; do not use `sample()` as the bulk proposal for high-precision,
rate-weighted or exponentially tilted Monte Carlo.

For high-precision population Monte Carlo, use the separately validated
rate-included release:

```python
model.set_flow(release="rate-included-v1")
prior = model.galactic_model(isochrone)
sample = prior.sample(key)
```

Its source-group conditional experts were trained directly on raw genulens
`wtj`, and its component-resolved source grid is the matching event-rate
`(DS, component)` marginal. The NSD and halo use the balanced rare-group
expert; the three major groups use the full-grid raw-`wtj` kernel. The complete
package is released against independent genulens holdouts:
the joint five-dimensional state, derived `DL/DS`, `mu_rel`, `theta_E`, and
`t_E`, source-group fractions, and rank-correlation structure all pass the
limits recorded in its manifest. Consequently `log_density()` and `sample()`
do not apply a second rate factor or importance correction.
`include_event_rate=False` is rejected for this release because the factor
cannot be removed after training. The coverage and `REMNANT=0`, `BINARY=0`
restrictions are the same as `default`.

The executed
[`flow_galactic_model.ipynb`](example/flow_galactic_model.ipynb) example covers
initialization and density evaluation, `emcee` sampling versus matching
weighted genulens rows, and integration with source magnitude/color data.

`log_density(..., magnitudes=...)` conditions the event prior on the supplied
photometry, but does not include the photometry as an additional source prior.
Use `log_source_density(ds=..., magnitudes=...)` when that factor belongs in an
analysis. `source_radius(ds=..., magnitudes=...)` provides the corresponding
source-radius summary.

## Histogram backend

The legacy event-local histogram workflow is also available through the same
public API. It requires the installed `genulens` package:

```python
import gapmoe

model = gapmoe.Model()
model.set(
    l=1.0,
    b=-3.9,
    extinction={"Imag": 1.2, "Vmag": 2.0},
    dm_rc=14.45,
)
model.prepare("runs/event-001")

isochrone = model.isochrone(
    reference_band="Imag",
    color_bands=("Vmag", "Imag"),
    magnitude_range=(15.0, 21.0),
    color_range=(0.5, 3.0),
)
prior = model.galactic_model(isochrone)
```

`Model().resume("runs/event-001")` reuses prepared artifacts. A source
checkout is only needed while developing genulens' CLI helpers:

```bash
make -C ../genulens pre_gapmoe
```

## Source populations

The default isochrone population reproduces genulens' broken-power-law IMF and
component-dependent age-metallicity mixtures. Override only the desired part:

```python
from gapmoe import AgeMetallicityPoint, SourcePopulation

population = SourcePopulation(
    imf={"alpha2": -1.3},
    age_metallicity_by_component={
        8: (AgeMetallicityPoint(log_age=9.9, metallicity_mh=0.1, weight=1.0),),
    },
)
```

For V/I work, `model.set(ai_rc=..., evi_rc=...)` is equivalent to supplying
the matching per-band RC extinctions. Use `extinction` for other bands.

## Install

```bash
pip install gapmoe
```

For local development:

```bash
pip install -e ".[dev]"
pytest -q
```

## Public API

- `gapmoe.Model`
- `gapmoe.SourcePopulation`
- `gapmoe.AgeMetallicityPoint`
