# gapmoe

gapmoe provides Galactic prior tools for microlensing event modeling. The
current public API builds event-local histogram products with the `genulens`
`pre_gapmoe` Python API, loads those products in Python, and evaluates Galactic
density/prior terms for physical microlensing parameters.

The canonical public parameter order is:

```text
ML, DL, DS, mu_N, mu_E
```

where `ML` is in solar masses, `DL` and `DS` are in kpc, and proper motions are
in mas/yr.

## Recommended API

`Model.prepare(directory)` writes its fixed artifact names
(`mass.dat`, `rho.dat`, `murel.dat`, `source_evidence.npz`, `manifest.json`)
there, together with `gapmoe.json`. The latter records the sightline and
gapmoe settings, so `Model().prepare(directory)` later restores them.

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

logp = prior.log_density(theta)
logp_at_source_magnitudes = prior.log_density(
    theta,
    magnitudes={"Imag": i_s, "Vmag": v_s},
)

logp_source = prior.log_source_density(
    ds=theta[2],
    magnitudes={"Imag": i_s, "Vmag": v_s},
)
radius = prior.source_radius(
    ds=theta[2],
    magnitudes={"Imag": i_s, "Vmag": v_s},
)
radius_rsun = radius.mean_rsun
```

To reuse an existing prepared directory without running genulens:

```python
model = gapmoe.Model().resume("runs/event-001")
```

Without `magnitudes`, the optional ranges in `isochrone()` define the source
selection. With named apparent magnitudes, the prior instead conditions the
source-distance distribution on that current photometry; it does not apply the
hard range a second time. `log_source_density(ds=..., magnitudes=...)` adds the
source-photometry prior itself, while `source_radius(...)` returns the corresponding
source-population radius summary. If no ranges are supplied, all sources are
used.

The default isochrone source population exactly uses genulens' broken-power-law
IMF and its component-dependent age-metallicity mixtures. Override only the
parts that matter for a systematic test:

```python
from gapmoe import AgeMetallicityPoint, SourcePopulation

population = SourcePopulation(
    imf={"alpha2": -1.3},
    age_metallicity_by_component={
        8: (AgeMetallicityPoint(log_age=9.9, metallicity_mh=0.1, weight=1.0),),
    },
)
isochrone = model.isochrone(
    reference_band="Imag",
    color_bands=("Vmag", "Imag"),
    population=population,
)
```

Unspecified components retain the genulens default mixture.

For V/I work, `model.set(ai_rc=..., evi_rc=...)` is equivalent to supplying
the matching per-band RC extinctions. For other bands, use `extinction`.

## Install

From PyPI:

```bash
pip install gapmoe
```

For local development:

```bash
pip install -e ".[dev]"
```

For the core NumPy backend only:

```bash
pip install -e .
```

Optional extras:

- `.[jax]`: JAX histogram/prior backend.
- `.[examples]`: notebook plotting/sampling dependencies.
- `.[dev]`: JAX, examples, and pytest.

## genulens Dependency

gapmoe depends on `genulens>=2.0.0a3`. A normal `pip install gapmoe` installs
`genulens`, including the bundled `pre_gapmoe` helper executables used by
`genulens.pre_gapmoe`.

Check what backend `PreRunner` will use:

```python
from gapmoe import PreRunner

runner = PreRunner()
env = runner.check_environment()
print(runner.backend)
print(env.ok)
print(env.backend)
```

By default, `PreRunner(backend="auto")` uses the installed `genulens.pre_gapmoe`
Python API. For development against a source checkout, pass `backend="cli"` or
`genulens_root=...` to run the local CLI helpers instead:

```bash
git clone https://github.com/nkoshimoto/genulens.git ../genulens
make -C ../genulens/pre_gapmoe
```

```python
runner = PreRunner(genulens_root="../genulens")
```

The CLI backend resolves a checkout from `genulens_root=...`,
`GAPMOE_GENULENS_ROOT`, `GENULENS_ROOT`, or nearby default candidates. It can
also try `make` automatically with `PreRunner(..., auto_build=True)`.

## Minimal Usage

```python
from gapmoe import GalacticModel, HistogramDensity, PreRunner

runner = PreRunner(
    output_dir="example/pre_runner_outputs",
)

pre_run = runner.run(
    ra_deg=270.0,
    dec_deg=-30.0,
    run_name="demo",
)

density = HistogramDensity.from_pre_run(pre_run)
prior = GalacticModel(density)

logp = prior.log_prob(
    0.3,  # ML [Msun]
    5.0,  # DL [kpc]
    8.0,  # DS [kpc]
    5.0,  # mu_N [mas/yr]
    2.0,  # mu_E [mas/yr]
)
```

If histogram files already exist:

```python
density = HistogramDensity.from_paths("mass.dat", "rho.dat", "murel.dat")
prior = GalacticModel(density)
```

## JAX Backend

`HistogramDensity` is the JAX-native histogram evaluator:

```python
from gapmoe import HistogramDensity, MappedGalacticModel

density = HistogramDensity.from_paths("mass.dat", "rho.dat", "murel.dat")
prior = MappedGalacticModel(density)
```

The JAX backend is currently intended for batched evaluation workflows such as
`jax.vmap`. Histogram lookup is piecewise and uses nearest `(DS, DL)` murel
blocks, so differentiability has not been validated. A future normalizing-flow
backend is the better target for smooth gradients.

## Source Evidence

`rho.dat` stores `nMS[i]` and a historical `rhoD_S[i]` column. The canonical
`HistogramDensity` ignores `rhoD_S[i]`, reconstructs the forward-source base
factor `nMS[i] * 1e-6 * D_S**2`, and multiplies it by
`p(source data | i, D_S)`. A hard selection is one special case of this
factor. Apply a source-evidence table after loading the histograms:

```python
density = HistogramDensity.from_pre_run(pre_run)
conditioned_density = density.with_source_evidence(source_evidence)
```

Pass `source_evidence=source_evidence` to `from_paths` or `from_pre_run` to
apply it while loading. The JAX loader accepts the same argument. The input
`PreRunner` writes a raw `rho.dat` with `SOURCE=0`, a source-group conditional
`murel.dat`, and `source_evidence.npz` in the run directory. When a CMD table
is passed, it also saves `cmd_prior.npz`. `HistogramDensity.from_pre_run(...)`
automatically load that table. With no explicit selector, the table has
unit evidence and uses the physical forward-source `nMS * D_S**2` base.

For a magnitude or colour selector, pass a `GenulensSourceModel`. The
source-data evidence is then evaluated from
genulens' isochrone model on the same distance grid:

```python
from gapmoe.source_selection import (
    ExponentialDustOffsets,
    GenulensSourceModel,
    MagnitudeCut,
    SourceSelection,
)

source_model = GenulensSourceModel(
    bands=("F146mag",),
    source_data=SourceSelection(cuts=(MagnitudeCut("F146mag", 15.0, 21.0),)),
    offset_provider=ExponentialDustOffsets(
        l_deg=1.0,
        b_deg=-3.9,
        extinction_at_reference={"F146mag": 1.0},
    ),
)
pre_run = runner.run(l=1.0, b=-3.9, source_model=source_model)
density = HistogramDensity.from_pre_run(pre_run)
```

When a CMD table is also used for the event prior, pass it to `PreRunner`.
The hard-cut evidence is then integrated from that same table rather than from
a separate selection model:

```python
cmd_prior = source_model.build_cmd_prior(
    CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
    reference_edges=np.linspace(-5.0, 15.0, 401),
    color_edges=np.linspace(-1.0, 6.0, 281),
)
pre_run = runner.run(l=1.0, b=-3.9, source_model=source_model, cmd_prior=cmd_prior)
```

Replace the hard-cut `SourceSelection` with `SourcePhotometry` when actual
source photometry is available. The same evidence calculation updates the
distance/component prior. Angular-source-radius inference remains in the
separate isochrone source model, so a theta-star estimate derived from this
photometry is not double-counted here:

```python
from gapmoe.source_selection import (
    MagnitudeMeasurement,
    SourcePhotometry,
)

source_model = GenulensSourceModel(
    bands=("F146mag",),
    source_data=SourcePhotometry(
        magnitudes=(MagnitudeMeasurement("F146mag", value=19.2, error=0.03),),
    ),
    offset_provider=ExponentialDustOffsets(
        l_deg=1.0,
        b_deg=-3.9,
        extinction_at_reference={"F146mag": 1.0},
    ),
)
```

Apparent magnitudes and colours always require an `offset_provider`; it must
provide the distance modulus and the extinction prescription for every used
band. Intrinsic absolute-magnitude constraints may instead set
`apparent=False`.

`PreRunner` no longer exposes the old luminosity-function selection options.
All source cuts are represented by `SourceSelection` and evaluated through the
forward isochrone model.

## CMD Joint Prior

For an MCMC that samples source fluxes or photometric parameters directly, use
an intrinsic component-conditional CMD table instead of conditioning a
five-dimensional prior on fixed measurements. The table represents
`p(m_reference, colour | DS, source_component, l, b)` after the current
distance and extinction shift. It therefore gives a joint density in event and
source-photometry variables at every MCMC step.

```python
import numpy as np

from gapmoe.source_selection import CmdCoordinates, GenulensSourceModel

cmd_prior = GenulensSourceModel(
    bands=("Imag", "Vmag"), samples_per_population_point=4096
).build_cmd_prior(
    CmdCoordinates(reference_band="Imag", blue_band="Vmag", red_band="Imag"),
    reference_edges=np.linspace(-5.0, 15.0, 401),
    color_edges=np.linspace(-1.0, 6.0, 281),
)

joint_density = density.cmd_joint_density(
    ML, DL, DS, mu_N, mu_E,
    cmd_prior=cmd_prior,
    reference_magnitude=i_source,
    color=v_source - i_source,
    offset_provider=dust_offsets,
)
```

For an MCMC parameter vector, wrap this direct evaluation with
`CmdGalacticModel`. The callbacks only need to extract the current apparent CMD
coordinates and JAX-compatible dust offsets:

```python
from gapmoe import CmdGalacticModel, EventPrior5D, ExponentialDustModel, SourceCmdPrior

dust = ExponentialDustModel.from_exponential(dust_offsets, cmd_prior.coordinates)
source_prior = SourceCmdPrior(
    density=density,
    cmd_prior=cmd_prior.evaluator(),
    offset_calculator=lambda ds_kpc, context: dust.offsets(ds_kpc),
)
prior = CmdGalacticModel(
    event_prior=EventPrior5D(density, source_prior),
    cmd_extractor=lambda theta, context: (theta[5], theta[6]),
)
log_prior = prior.log_prob(theta)
```

The result has density measure
`dML dDL dDS dmu_N dmu_E dm_reference dcolour`. For a two-flux parameterization
`(F_blue, F_red)` with `reference_band == red_band`, use
`cmd_joint_density_from_fluxes(...)`; it includes the required flux-to-CMD
Jacobian.

Convert the table once with `cmd_prior.evaluator()` and pass the current
offsets as a JAX array ordered as `(reference, blue, red)`. This keeps the
lookup and the source-component reweighting compatible with `jax.jit` and
`jax.vmap`:

```python
import jax.numpy as jnp

from gapmoe import HistogramDensity

cmd_prior_evaluator = cmd_prior.evaluator()
joint_density = density.cmd_joint_density(
    ML, DL, DS, mu_N, mu_E,
    cmd_prior=cmd_prior_evaluator,
    reference_magnitude=i_source,
    color=v_source - i_source,
    magnitude_offsets=jnp.asarray([offset_i, offset_v, offset_i]),
)
```

## Examples

Current notebooks:

- `example/pre_runner.ipynb`: generate event-local histogram files.
- `example/emcee_physical_params.ipynb`: sample the physical-parameter Galactic
  prior with `emcee` and compare with raw genulens Monte Carlo output in the
  final corner plot.

Generated pre-run files under `example/pre_runner_outputs/` are intentionally
ignored by git.

## Tests

```bash
pytest -q
```

The test suite uses a small committed histogram fixture under
`tests/fixtures/small_source_default/` and mocks the `genulens.pre_gapmoe`
integration path.

## Public API

- `PreRunner`
- `HistogramDensity`
- `GalacticModel` / `MappedGalacticModel`
- `gapmoe.param_types`
