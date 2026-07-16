# gapmoe

gapmoe provides sampler-independent Galactic priors for microlensing event
modeling. A `Model` is complete at construction: it owns the physical Galactic
density, light-curve parameterization, Jacobian, and hidden-variable
integration.

## Flow Model

```python
import gapmoe

param_type = gapmoe.ParamType(
    lens="binary",
    parallax=True,
    orbital_motion="static",
    distance="marginalize",
)

source = gapmoe.Isochrone(
    reference_band="Imag",
    color_bands=("Vmag", "Imag"),
    magnitude_range=(15.0, 21.0),
    color_range=(0.5, 3.0),
)

galaxy = gapmoe.Model(
    param_type,
    l=1.0,
    b=-3.9,
    extinction={"Imag": 1.2, "Vmag": 2.0},
    source=source,
    backend=gapmoe.Flow("rate-included-v1"),
    integration_samples=512,
)
```

`magnitude_range` and `color_range` are hard source selections. gapmoe
integrates the isochrone population inside those apparent-photometry bounds at
each source distance and component, then uses that evidence to reweight the
source-distance prior. Downstream packages receive the already-selected model;
they do not need a separate source-photometry API. The registration call below
belongs to lcbinint, not to gapmoe; ``lc_model`` denotes the downstream
light-curve model.

```python
lc_model.galactic_prior(galaxy)
```

Distance modulus is always included in this selection, including when
`extinction` is omitted or all extinctions are zero.

The model is not tied to lcbinint or to a particular sampler:

```python
context = {"thS": 0.005, "vEarth": (v_north, v_east)}

logp = galaxy.log_density(theta, context=context)
values = galaxy.log_density_batch(theta_batch, context=context)
physical = galaxy.to_physical(theta, context=context)
draw = galaxy.sample_physical(theta, context=context, rng=rng)
```

Priors involving hidden physical variables are evaluated inside the same
distance integral:

```python
import jax.numpy as jnp

@galaxy.prior
def physical_bounds(ML, DL, DS, **_):
    valid = (ML > 0.01) & (DL > 0.0) & (DL < DS)
    return jnp.where(valid, 0.0, -jnp.inf)
```

## Physical Model

The lower-level physical density remains available without changing the main
inference API. Its canonical coordinates are
`(ML, DL, DS, mu_N, mu_E)`, with mass in solar masses, distances in kpc, and
proper motions in mas/yr.

```python
logp = galaxy.physical.log_density((ML, DL, DS, mu_N, mu_E))
sample = galaxy.physical.sample(key, magnitudes={"Imag": i_s, "Vmag": v_s})
```

## Parameterizations

`ParamType` determines both the visible parameter names and which physical
variables are integrated internally.

- `parallax=True, distance="sample"`: `DS` is explicit.
- `parallax=True, distance="marginalize"`: `DS` is integrated.
- `parallax=False, distance="sample"`: `DL` and `DS` are explicit; the
  proper-motion direction is integrated.
- `parallax=False, distance="marginalize"`: `DL`, `DS`, and the
  proper-motion direction are integrated.
- `orbital_motion="circular"` and `"kepler"` enable binary-lens orbital
  mappings.

Flow-backed hidden-variable integration uses deterministic fixed QMC points.
`integration_samples=512` is the default. `direction_samples=32` controls the
direction-only quadrature used by sampled-distance no-parallax models.
For difficult short-timescale or isochrone-conditioned no-parallax events,
repeat representative evaluations with a larger value such as 2048 and check
that posterior summaries are stable.

## Histogram Backend

An existing event-local histogram uses the same complete `Model` API:

```python
galaxy = gapmoe.Model(
    param_type,
    l=1.0,
    b=-3.9,
    extinction={"Imag": 1.2, "Vmag": 2.0},
    source=source,
    backend=gapmoe.Histogram.open("runs/event-001"),
)
```

Histogram generation is a precomputation concern, separate from the inference
model itself. An ordinary no-parallax Histogram model uses its deterministic
precomputed `DL x DS` quadrature. Dynamic source-magnitude/isochrone
conditioning and hidden physical priors would require a different importance
proposal and are intentionally unsupported for a parallax-free Histogram
model. Use the Flow backend or sample distances explicitly for those cases.

## Supported Inference Boundary

- Flow supports sampled and marginalized distances, parallax-free hidden
  physical integration, dynamic source magnitudes, isochrone thetaS, and
  physical priors.
- Histogram supports its direct parameter transforms and deterministic
  distance quadrature.
- Histogram does not fall back to a generic hidden-physical QMC proposal for
  dynamic parallax-free conditioning. Such a request raises an explanatory
  error instead of returning a poorly controlled approximation.
- `sample_physical()` draws marginalized quantities conditionally from the
  same density or finite-QMC approximation used by `log_density()`.

## Downstream Integration

Downstream inference packages can consume `Model` through a small protocol:
`names`, `log_density()`, `is_valid()`, and `sample_physical()`. gapmoe does not
import lcbinint.

Supplying source magnitudes to `log_density()` changes the Galactic model
conditionally:

```text
p(physical | source magnitudes)
```

The marginal CMD density is divided out, so this does not define an additional
prior on a fitted, sampled, or marginalized source flux. `log_joint_density()`
remains a separate low-level operation for analyses that explicitly want the
joint CMD density.

The following examples use lcbinint integration hooks. They are not methods on
``gapmoe.Model`` itself; ``lc_model`` is the downstream light-curve model.

```python
lc_model.galactic_prior(galaxy)
```

For joint isochrone, source-flux, and distance integration, use the built
source model owned by `galaxy`:

```python
@lc_model.theta_star(isochrone=galaxy.isochrone)
def source_magnitudes(fluxes):
    return {
        "Imag": flux_to_mag(fluxes["I"]["Fs"]),
        "Vmag": flux_to_mag(fluxes["V"]["Fs"]),
    }
```

## Source Populations

The default isochrone population reproduces genulens' broken-power-law IMF and
component-dependent age-metallicity mixtures. It can be customized with
`SourcePopulation` and `AgeMetallicityPoint`.

## Migration from the builder API

This branch intentionally replaces the mutable
``Workspace().set(...).set_flow().galactic_model(...)`` inference workflow.
Construct ``gapmoe.Model(param_type, ..., backend=...)`` directly instead.
``Workspace`` remains internal to histogram precomputation and cache handling.

``GalaxyModel.parameterize(...)`` was removed because the parameterization is
now supplied when the complete ``gapmoe.Model`` is constructed. The former
``MagnitudeMeasurement``, ``ColorMeasurement``, and ``SourcePhotometry``
wrappers were also removed. Pass a band-to-magnitude mapping through
``Model.log_density(..., magnitudes=...)`` for conditional photometry, or use
``log_joint_density`` when the marginal CMD density belongs in the target.

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
- `gapmoe.Flow`
- `gapmoe.Histogram`
- `gapmoe.Isochrone`
- `gapmoe.SourcePopulation`
- `gapmoe.AgeMetallicityPoint`
- `gapmoe.calc_vEarth`
