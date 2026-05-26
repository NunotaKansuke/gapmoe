# GAPMOE

GAPMOE provides Galactic prior tools for microlensing event modeling. The
current public API builds event-local histogram products with external Genulens
`pre_gapmoe` tools, loads those products in Python, and evaluates Galactic
density/prior terms for physical microlensing parameters.

The canonical public parameter order is:

```text
ML, DL, DS, mu_N, mu_E
```

where `ML` is in solar masses, `DL` and `DS` are in kpc, and proper motions are
in mas/yr.

## Install

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

## External Genulens Dependency

GAPMOE does not vendor Genulens. To generate histogram inputs, provide a local
Genulens checkout containing `pre_gapmoe` with these executables:

- `calc_mass_dist`
- `calc_rho_profile`
- `calc_murel_dist`

`PreRunner` resolves the checkout from `genulens_root=...`,
`GAPMOE_GENULENS_ROOT`, `GENULENS_ROOT`, or nearby default candidates.

Build Genulens separately before running `PreRunner`. For a sibling checkout,
the usual shape is:

```bash
git clone <genulens-url> ../genulens
make -C ../genulens/pre_gapmoe
export GAPMOE_GENULENS_ROOT="$(realpath ../genulens)"
```

Check what GAPMOE sees:

```python
from gapmoe import PreRunner

env = PreRunner(genulens_root="../genulens").check_environment()
print(env.ok)
print(env.pre_gapmoe_dir)
print(env.missing_tools)
```

If `env.ok` is false, build `pre_gapmoe` or point `genulens_root` at the right
checkout. GAPMOE can also try `make` automatically with `PreRunner(...,
auto_build=True)`.

## Minimal Usage

```python
from gapmoe import GalacticPrior, HistogramDensity, PreRunner

runner = PreRunner(
    genulens_root="../genulens",
    output_dir="example/pre_runner_outputs",
)

pre_run = runner.run(
    ra_deg=270.0,
    dec_deg=-30.0,
    run_name="demo",
)

density = HistogramDensity.from_pre_run(pre_run)
prior = GalacticPrior(density)

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
prior = GalacticPrior(density)
```

## JAX Backend

The JAX backend mirrors the NumPy histogram semantics:

```python
from gapmoe import HistogramDensity, JaxGalacticPrior, JaxHistogramDensity

np_density = HistogramDensity.from_paths("mass.dat", "rho.dat", "murel.dat")
jax_density = JaxHistogramDensity.from_numpy(np_density)
jax_prior = JaxGalacticPrior(jax_density)
```

Histogram lookup is piecewise and uses nearest `(DS, DL)` murel blocks, so it is
not intended as a smooth density model. For gradient-heavy workflows, validate
the behavior carefully; a future normalizing-flow backend is the better target
for smooth gradients.

## Source Selection

Normal GAPMOE usage assumes `rho.dat` was generated with Genulens source
selection enabled. `PreRunner` passes `SOURCE=1` to `calc_rho_profile` by
default. `HistogramDensity.from_paths(...)` requires source-density columns by
default and uses `rhoD_S_tot` for the source-distance factor.

## Examples

Current notebooks:

- `example/pre_runner.ipynb`: generate event-local histogram files.
- `example/emcee_physical_params.ipynb`: sample the physical-parameter Galactic
  prior with `emcee` and compare with raw Genulens Monte Carlo output in the
  final corner plot.

Generated pre-run files under `example/pre_runner_outputs/` are intentionally
ignored by git.

## Tests

```bash
pytest -q
```

The test suite uses a small committed histogram fixture under
`tests/fixtures/small_source_default/` and does not run external Genulens
binaries.

## Legacy Modules

`gapmoe.gapmoe`, `gapmoeJax.py`, and older parameter-conversion modules remain
for compatibility while the public API stabilizes. New code should prefer:

- `PreRunner`
- `HistogramDensity` / `JaxHistogramDensity`
- `GalacticPrior` / `JaxGalacticPrior`
- `gapmoe.parameterizations`
