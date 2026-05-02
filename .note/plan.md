# GAPMOE Main Design Plan

Date: 2026-05-02

## Goal

Refactor the main GAPMOE code from one large legacy `gapmoe` class into a package-level design that can support:

- histogram-based Galactic density, as implemented today;
- future density models such as normalizing flows;
- flexible microlensing parameterizations chosen by users;
- per-event preprocessing through external Genulens `pre_gapmoe`;
- NumPy/JAX backends without duplicating the whole model by hand.

## Current Problem

`gapmoe` is the overall project/package name. It should not be treated as the name of one specific model object in the long-term API.

`src/gapmoe/gapmoe.py` currently mixes several responsibilities:

- sky coordinate handling;
- locating histogram files;
- loading and normalizing histogram tables;
- evaluating density terms;
- computing the event-rate factor `Gamma`;
- exposing the final Galactic prior;
- legacy assumptions about precomputed 0.2 degree histogram grids.

`src/gapmoe/gapmoeJax.py` duplicates much of the same logic in JAX form.

`src/gapmoe/parametrics.py` contains parameter transformations, Jacobians, and several model-specific assumptions in one file. These are likely to vary by user and by light-curve model.

## Design Direction

Use composition instead of one large model class.

The high-level object should coordinate three separate pieces:

1. A density model:
   - answers `log_density(physical_params)`;
   - can be histogram-based now and flow-based later.
2. A parameterization:
   - maps user/light-curve parameters to physical parameters;
   - optionally returns a log-Jacobian term.
3. A prior/evaluator:
   - combines density, event-rate terms, parameterization Jacobian, and user-specific extra priors.

This keeps the Galactic density model independent from how a user chooses to parameterize a light curve.

## Proposed Package Layout

Potential target layout:

```text
src/gapmoe/
  __init__.py
  coordinates.py
  pre_runner.py
  constants.py

  density/
    __init__.py
    base.py
    histogram.py
    flow.py
    tables.py

  parameterizations/
    __init__.py
    base.py
    binary_circular.py
    binary_kepler.py
    single_lens.py

  priors/
    __init__.py
    galactic.py
    event_rate.py

  backends/
    __init__.py
    numpy.py
    jax.py
```

This does not need to be created all at once. It is a target shape for migration.

## Core Concepts

### Physical Parameters

Create a canonical physical-parameter representation shared by all density backends.

Candidate fields:

- `ML`: lens mass, solar masses.
- `DL`: lens distance, kpc or pc; unit must be made explicit.
- `DS`: source distance, kpc or pc.
- `mu_N`: heliocentric relative proper motion north, mas/yr.
- `mu_E`: heliocentric relative proper motion east, mas/yr.
- Derived:
  - `mu = sqrt(mu_N^2 + mu_E^2)`.
  - `phi = atan2(mu_E, mu_N)`.

Open issue:

- Current histogram code mixes kpc-like calls (`DL <= 16` warning) with preprocessor outputs in pc. The refactor should choose one internal unit and convert at boundaries.

### Density Interface

Proposed base API:

```python
class DensityModel:
    def log_density(self, params):
        ...

    def component_fraction(self, params):
        ...
```

`HistogramDensity` would implement current behavior:

- load `mass.dat`;
- load `rho.dat`;
- load `murel.dat`;
- evaluate:
  - `p(M_L | D_L)`;
  - `p(D_L | D_S)`;
  - `p(D_S)`;
  - `p(mu | D_L, D_S)`;
  - `p(phi | D_L, D_S)`.

`FlowDensity` could later implement:

- `log p(ML, DL, DS, mu_N, mu_E)` directly;
- optionally conditional flow variants, e.g. conditioned on sky position or source constraints.

The rest of GAPMOE should not care whether the density came from histograms or a flow.

### Histogram Tables

Split parsing/storage from probability logic.

Possible objects:

- `MassHistogram`
- `DistanceDensityTable`
- `MurelHistogram`
- `HistogramDensity`

This avoids keeping all parsing code inside one class and makes it testable.

### Parameterization Interface

Parameterization should be swappable.

Proposed API:

```python
class Parameterization:
    names: tuple[str, ...]

    def to_physical(self, theta, context):
        ...

    def log_abs_det_jacobian(self, theta, context):
        ...
```

`context` can hold values like:

- `theta_star`;
- `vEarth`;
- event time metadata;
- optional fixed source distance;
- user-supplied constants.

Concrete parameterizations:

- `BinaryCircularParameterization`
- `BinaryKeplerParameterization`
- `SingleLensParameterization`
- user-defined classes following the same protocol.

Open issue:

- Some users may already have physical parameters and do not need a transformation. Support an identity parameterization.

### Prior Composition

The top-level model should combine terms rather than own all details.

Candidate API:

```python
class GalacticPrior:
    def __init__(self, density, parameterization=None, include_event_rate=True):
        ...

    def log_prob(self, theta_or_params, context=None):
        ...
```

If a parameterization is supplied:

```text
log_prob(theta) =
    density.log_density(parameterization.to_physical(theta))
  + event_rate.log_gamma(physical_params)
  + parameterization.log_abs_det_jacobian(theta)
  + optional user priors
```

If no parameterization is supplied:

```text
log_prob(physical_params) =
    density.log_density(physical_params)
  + event_rate.log_gamma(physical_params)
```

## Relationship With PreRunner

`PreRunner` should remain a preprocessing tool, not a density evaluator.

Flow:

1. User creates `PreRunner`.
2. `PreRunner.run(...)` creates event-local files:
   - `mass.dat`
   - `rho.dat`
   - `murel.dat`
   - `manifest.json`
3. `HistogramDensity.from_run(result)` or `HistogramDensity.from_paths(...)` loads those files.
4. `GalacticPrior` uses the density object.

This separates external C/C++ execution from Python-side probability evaluation.

## Source Selection And rho.dat Semantics

Decision:

- GAPMOE's intended prior is conditioned on an observed source.
- Therefore `p(DS)` should come from `rhoD_S_tot` in `rho.dat`, not from a raw stellar number-density column.
- `PreRunner` should pass `SOURCE=1` to `calc_rho_profile` by default, even when the user does not provide explicit source-selection options.
- With no explicit source-selection options, this follows the default fallback behavior in `genulens.cpp`: use the source-distance weighting controlled by `gammaDs` (default `0.5`) rather than no source conditioning.

Column meanings from `calc_rho_profile`:

- `nMS[0..10]`: main-sequence number density by Galactic component.
- `n[0..10]`: main-sequence plus white-dwarf number density by Galactic component.
- `rhoD_S[0..10]`: source-distance density after source weighting/selection, by Galactic component.

Current histogram-density rule:

- Lens distance/component weights use `nMS`, because the current mass and murel preprocessor outputs are normalized with main-sequence component weights.
- Source distance uses `rhoD_S_tot`; missing source-density columns are treated as an invalid normal input file.
- Legacy `rho.dat` files without `SOURCE=1` can only be loaded by explicitly opting out in test/debug code.

## NumPy vs JAX

Avoid maintaining two full copies of the same model.

Options:

1. Start with NumPy-only `HistogramDensity`.
2. Add a JAX implementation only for parts that need gradients.
3. Define backend helper functions later if duplication becomes real.

Pragmatic recommendation:

- First refactor into clean NumPy classes.
- Keep old `gapmoeJax.py` as legacy until the API stabilizes.
- Then port the stable density interface to JAX if needed.

## Migration Plan

### Step 1: Preserve Behavior

- Keep current `gapmoe` class as compatibility wrapper.
- Add new classes beside it.
- Make the wrapper load `HistogramDensity` internally.
- Do not change external behavior yet.

### Step 2: Introduce HistogramDensity

- Move file parsing and normalization out of `gapmoe.py`.
- Add `HistogramDensity.from_paths(mass_path, rho_path, murel_path)`.
- Add `HistogramDensity.from_pre_run(pre_run_result)`.
- Add tests with the small `/tmp/gapmoe_prerunner_smoke/small` outputs or a committed tiny fixture.

### Step 3: Introduce PhysicalParams

- Use a dataclass or lightweight mapping for physical parameters.
- Make units explicit.
- Make old method signatures call into the new object.

### Step 4: Introduce Parameterization Classes

- Move functions from `parametrics.py` into named classes or wrappers.
- Keep pure functions available for backward compatibility.
- Add `IdentityParameterization`.

### Step 5: Introduce GalacticPrior

- Compose:
  - density;
  - parameterization;
  - event-rate term;
  - optional extra priors.
- Make this the recommended public API.

### Step 6: Future FlowDensity

- Add `FlowDensity` with the same `log_density(params)` interface.
- It should be a separate backend, not a special case inside `HistogramDensity`.

## Naming Ideas

Top-level public names:

- `PreRunner`
- `HistogramDensity`
- `FlowDensity`
- `GalacticPrior`
- `PhysicalParams`
- `Parameterization`

Potential compatibility wrapper:

- Keep the current lowercase `gapmoe` class only as a compatibility wrapper for old scripts.
- Do not introduce a new main class named `GapMoe`; `gapmoe` is the package/project name.
- Prefer descriptive class names such as `GalacticPrior`, `HistogramDensity`, and `PreRunner`.

## Questions To Resolve

- What should the canonical internal distance unit be: pc or kpc?
- Should `mu_N/mu_E` or `mu/phi` be the canonical density input?
- Should the event-rate factor `Gamma` always be included, or should users opt into it?
- Do normalizing flows model the same physical parameter set as histograms, or a transformed space such as log mass/log distance?
- Should `PreRunner` always regenerate all three files, or cache/reuse mass tables when model options are unchanged?
- How much of JAX support is needed for the first public release?

## Current Recommendation

Do the next implementation in this order:

1. Add `density/base.py`, `density/histogram.py`, and `priors/event_rate.py`.
2. Move only the histogram-loading and density-evaluation logic from `gapmoe.py`.
3. Keep `gapmoe.py` as a wrapper for now.
4. Add `GalacticPrior` only after `HistogramDensity` is stable.
5. Delay `FlowDensity` implementation, but keep the interface flow-ready from the start.

## Implementation Status

2026-05-02:

- Added `src/gapmoe/physical.py` with `PhysicalParams`.
  - Canonical distance unit is currently pc.
  - Canonical proper-motion components are heliocentric `mu_N`, `mu_E` in mas/yr.
  - `mu` and `phi` are derived properties.
- Added `src/gapmoe/priors/event_rate.py` with `log_event_rate`.
- Added `src/gapmoe/density/base.py` with `DensityModel`.
- Added `src/gapmoe/density/histogram.py` with:
  - `MassHistogram`
  - `DistanceDensityTable`
  - `MurelHistogram`
  - `HistogramDensity`
- `HistogramDensity` reads current `PreRunner` outputs: `mass.dat`, `rho.dat`, `murel.dat`.
- Added public lazy exports for `PhysicalParams` and `HistogramDensity`.
- Parameterization code was intentionally not touched in this step.
- Committed this density layer as `bb93116 Add histogram density model`.

2026-05-02, next step:

- Added `src/gapmoe/priors/galactic.py` with `GalacticPrior`.
- `GalacticPrior` composes:
  - a `DensityModel`;
  - optional event-rate factor;
  - optional parameterization hook;
  - optional extra user prior.
- The implemented default path is intentionally conservative: without a parameterization, `log_prob(...)` accepts only `PhysicalParams`.
- The parameterization interface is currently a thin protocol only. No concrete light-curve parameter transformations have been moved yet.
- Added public lazy export for `GalacticPrior`.

2026-05-02, compatibility wrapper:

- Added `src/gapmoe/model.py` with `GalacticModel`.
- `GalacticModel` is the public wrapper name. Avoid using `gapmoe.gapmoe` as the main user-facing name because `gapmoe` is the package/project name.
- Replaced `src/gapmoe/gapmoe.py` with a small compatibility shim that exports `GalacticModel` and the deprecated lowercase `gapmoe` alias.
- The wrapper can be built from:
  - an existing `HistogramDensity`;
  - a `PreRunResult`;
  - explicit `mass/rho/murel` paths;
  - `ra_deg` and `dec_deg`, in which case it runs `PreRunner`.
- Legacy method names such as `get_joint_log_density`, `log_galactic_prior`, and `get_density_M_given_DL` now delegate to the new density/prior objects.

2026-05-02, examples:

- Added `example/emcee_physical_params.ipynb`.
- The notebook demonstrates the current canonical workflow:
  - run `PreRunner` for one sky position;
  - build `GalacticModel` from the generated `mass.dat`, `rho.dat`, and `murel.dat`;
  - evaluate `PhysicalParams(ML, DL, DS, mu_N, mu_E)`;
  - sample those five physical parameters with `emcee`.
- The notebook samples the Galactic prior itself. Real event likelihoods should be added inside its `log_probability` function.
- Generated pre-run files go under `example/pre_runner_outputs/` and should not be treated as source files.

2026-05-02, murel performance:

- The first `MurelHistogram` implementation searched `rows` by boolean masks on every density evaluation.
- This became slow for large grids such as `DL=[0,10000]`, `DS=[0,16000]`, `step=100`, where `murel.dat` can be tens of MB.
- Updated `MurelHistogram` to build `(DS, DL) -> row slice` block indices at load time.
- It also parses the `# Grid: DL ...` and `# Grid: DS ...` header metadata for inspection/debugging.
- Density evaluation now looks only at the small murel/phi block for the relevant neighboring grid cells, instead of scanning the whole table.

Smoke checks:

- `py_compile` passed for new modules.
- `HistogramDensity.from_paths(...)` loaded `/tmp/gapmoe_prerunner_smoke/small/{mass,rho,murel}.dat`.
- For `PhysicalParams(ML=0.3, DL=250, DS=600, mu_N=5, mu_E=2)`, `HistogramDensity` returned finite `log_density` and `log_prior`.
- `GalacticPrior(HistogramDensity).log_prob(PhysicalParams(...))` matches `HistogramDensity.log_prior(...)` exactly on `/tmp/gapmoe_prerunner_smoke/small_source_default`.
- `GalacticModel.from_paths(...)` loads `/tmp/gapmoe_prerunner_smoke/small_source_default` and returns the same `log_galactic_prior` as `log_prob`.
- `from gapmoe.gapmoe import GalacticModel, gapmoe` remains import-compatible.
- `example/emcee_physical_params.ipynb` passes JSON validation and all code cells compile.
- On `example/pre_runner_outputs/emcee_demo/murel.dat` with 788400 rows and 10950 `(DS, DL)` blocks, load time was about 2.9 s and 1000 repeated `log_prob` evaluations took about 0.22 s.

Important caveats:

- The current histogram interpretation is a first extraction from the legacy `gapmoe` behavior plus the new `pre_gapmoe` output format. It needs scientific validation.
- `rho.dat` source density uses `rhoD_S_tot` for normal GAPMOE usage. Files without source-density columns are legacy/debug inputs and require an explicit opt-out.
- Lens density uses `nMS[0..10]` by component.
- `murel.dat` lookup uses distance-weighted neighboring `(DS, DL)` histogram blocks when bracketing blocks exist, with nearest-block fallback outside the tabulated grid.
- Normalizing-flow support is not implemented yet; the interface is only prepared for it.

## Density Validation Notes

2026-05-02:

- Checked `../genulens/pre_gapmoe/calc_rho_profile.cpp` output definition.
  - Columns are `D`, `nMS[0..10]`, `nMS_tot`, `n[0..10]`, `n_tot`, and optionally `rhoD_S[0..10]`, `rhoD_S_tot`.
  - `rhoD_S` is only present when source selection / extinction weighting is active.
- Checked `../genulens/pre_gapmoe/calc_mass_dist.cpp` output definition.
  - Its PDMF is normalized to main-sequence counts `n0MS`.
- Checked `../genulens/pre_gapmoe/calc_murel_dist.cpp`.
  - Lens/source component weights use `n0MS_arr[i] * rho_i`.
  - `murel.dat` grid rows are centered at `(DS_c, DL_c)`.

Decision:

- Use `nMS[0..10]` for lens component fractions and `p(DL | DS)` in `HistogramDensity`.
- Use `rhoD_S_tot` for the source-distance factor because GAPMOE is intended to condition on the observed source.
- `PreRunner` passes `SOURCE=1` by default. If no source-selection options are supplied, this matches the `genulens.cpp` default fallback weighting with `gammaDs=0.5`.
- Do not silently fall back to `nMS_tot` for normal `HistogramDensity` loading; old files without `rhoD_S` need an explicit debug/legacy opt-out.
- Do not use `n[0..10]` / `n_tot` for the current histogram density because current mass and murel preproducts are main-sequence normalized.
- Replace nearest-only murel block lookup with distance-weighted neighboring grid-block lookup plus nearest fallback.

Clarification:

- `nMS` is main-sequence number density.
- `n` is MS+WD number density. It includes white dwarfs, but not the same source population as `nMS`.
- `rhoD_S` is not just a density. It is a source-selection weight derived from `nMS_i`, detection/luminosity-function weighting, and a distance-volume factor (`D^2`) or fallback source-distance weight.
- Therefore `rhoD_S_tot` is the right `p(DS)` input for the observed-source-conditioned prior that GAPMOE should represent.

Smoke checks:

- Parsed `/tmp/gapmoe_prerunner_smoke/small/rho.dat`; first-row `nMS[0..10]` sum matches `nMS_tot`.
- `PhysicalParams(ML=0.3, DL=260, DS=600, mu_N=5, mu_E=2)` returns finite density after the `nMS` mapping change.
- For `DL=500`, `DS=1000`, weighted murel lookup differs from nearest lookup, confirming it is interpolating across neighboring blocks.
- For exact grid-center input, weighted murel lookup equals nearest lookup.
- Re-ran `PreRunner` without explicit source-selection options at `/tmp/gapmoe_prerunner_smoke/small_source_default`.
  - The manifest shows `calc_rho_profile ... SOURCE 1`.
  - The generated `rho.dat` has 37 columns, including `rhoD_S_tot`.
  - `HistogramDensity.from_paths(...)` loads it with default `require_source_selection=True`.
