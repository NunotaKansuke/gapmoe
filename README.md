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
from gapmoe import GalacticPrior, HistogramDensity, PreRunner

runner = PreRunner(
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

The JAX backend is currently intended for batched evaluation workflows such as
`jax.vmap`. Histogram lookup is piecewise and uses nearest `(DS, DL)` murel
blocks, so differentiability has not been validated. A future normalizing-flow
backend is the better target for smooth gradients.

## Source Selection

Normal gapmoe usage assumes `rho.dat` was generated with genulens source
selection enabled. `PreRunner` passes `SOURCE=1` to `calc_rho_profile` by
default. `HistogramDensity.from_paths(...)` requires source-density columns by
default and uses `rhoD_S_tot` for the source-distance factor.

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

## Legacy Modules

`gapmoe.gapmoe`, `gapmoeJax.py`, and older parameter-conversion modules remain
for compatibility while the public API stabilizes. New code should prefer:

- `PreRunner`
- `HistogramDensity` / `JaxHistogramDensity`
- `GalacticPrior` / `JaxGalacticPrior`
- `gapmoe.parameterizations`
