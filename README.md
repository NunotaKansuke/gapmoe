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

To express the same physical density in microlensing light-curve parameters,
attach an independent parameterization:

```python
params = gapmoe.ParamType(
    lens="binary",
    parallax=True,
    orbital_motion="static",
    distance="marginalize",
)
light_curve_prior = prior.parameterize(params)

context = {"thS": 0.005, "vEarth": (v_north, v_east)}
logp = light_curve_prior.log_density(theta, context=context)
```

Priors that depend on hidden physical integration variables belong inside the
parameterized Galactic prior:

```python
import jax.numpy as jnp

@light_curve_prior.prior
def physical_bounds(ML, DL, DS, **_):
    valid = (ML > 0.01) & (DL > 0.0) & (DL < DS)
    return jnp.where(valid, 0.0, -jnp.inf)
```

These callables are evaluated inside distance marginalization and must be
JAX-compatible for Flow-backed models.

The parameterized object owns its physical transform, Jacobian, and hidden
distance integration. It remains independent of any sampler and exposes the
small `names`/`log_density` protocol expected by downstream inference tools.

Without parallax, Flow-backed models use deterministic importance integration:

```python
prior = galaxy.parameterize(
    gapmoe.ParamType(parallax=False, distance="marginalize"),
    integration_samples=512,
    seed=0,
)
```

The proposal draws `DS` from the packaged source-distance measure, uses a
full-support `DL/DS` proposal fitted from fast Flow samples, and samples the
proper-motion direction uniformly. Fixed Halton points are reused at every
evaluation, so the MCMC target is deterministic. With `distance="sample"`,
`DL` and `DS` are explicit parameters and only the proper-motion direction is
integrated; `direction_samples=32` controls that quadrature.

Circular and Kepler orbital-motion parameterizations use lcbinint's canonical
names `g1`, `g2`, `g3`, `lom_szs`, and `lom_ar` directly.

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

The same isochrone radius information can participate in downstream
light-curve inference jointly with source distance. The parameterized prior
exposes a small provider protocol for paired-QMC consumers; users do not need
to configure a separate thetaS prior on the gapmoe object.

Within each CMD bin and source component, the stored first two
`log(R/Rsun)` moments define a log-normal radius approximation. Paired-QMC
consumers keep component weights inside the Flow or histogram event-density
sum instead of forming a Cartesian thetaS-by-distance grid. For diagnostics,
`prior.log_theta_star_density(theta_star_mas=..., ds=..., magnitudes=...)`
returns `log p(log(thetaS) | DS, magnitudes)` directly.

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
- `gapmoe.ParamType`
- `gapmoe.SourcePopulation`
- `gapmoe.AgeMetallicityPoint`
- `gapmoe.calc_vEarth`
