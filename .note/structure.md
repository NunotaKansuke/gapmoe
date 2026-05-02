# GAPMOE repository structure notes

Date: 2026-05-02

## Note policy

- Keep `.note/` updated whenever the repository structure, public-readiness plan, or implementation direction changes.
- Record the current state, decisions made, open issues, and next steps as the cleanup proceeds.

## Current state

- This directory was not a git repository before this cleanup pass.
- Main Python code lives under `src/gapmoe/`.
- The bundled old Genulens C preprocessing copy under `src/genulens/` has been removed from the working tree.
- Exploratory scripts and notebooks live under `test_tool/`.
- Public-facing examples live under `example/`; `example/emcee_physical_params.ipynb` currently demonstrates the NumPy histogram backend by building `HistogramDensity` and `GalacticPrior` explicitly before sampling with `emcee` and `corner`.
- Automated tests live under `tests/`; `pytest.ini` restricts default discovery there, and `tests/fixtures/small_source_default/` contains a small committed histogram fixture for NumPy/JAX backend parity checks.
- Runtime histogram data is expected at absolute paths under `/moao38_7/nunota/gapmoe/`, but those data directories are not present in this checkout.
- Initial git commit exists: `6d4d8dd Initial GAPMOE baseline`.
- `src/gapmoe/pre_runner.py` has been added but is not committed yet.
- Existing local histogram data was found outside this checkout under `/moao38_7/nunota/gapmoe/`.
- Design change: public GAPMOE should not depend on a precomputed 0.2 degree histogram grid. `PreRunner` should generate event-specific histograms each time.

## `src/gapmoe`

- `gapmoe.py` is now a compatibility shim around `GalacticModel`; the old monolithic model should not be the long-term public API.
- `model.py` contains `GalacticModel`, the compatibility-facing wrapper that builds a density and prior from pre-run outputs or explicit histogram paths.
- `density/histogram_numpy.py` contains the NumPy histogram density backend.
  - `density/histogram.py` remains as a compatibility re-export for older imports.
- `density/histogram_jax.py` contains `JaxHistogramDensity`, the JAX histogram density backend using the same event-local files and raw physical-parameter API.
- `priors/galactic.py` and `priors/galactic_jax.py` compose density backends with event-rate and optional parameterization hooks.
- `gapmoeJax.py` remains a legacy copy of the old hard-coded 0.2 degree grid model. New JAX work should happen in backend-specific modules such as `density/histogram_jax.py`.
- `parametrics.py`, `parametrics2.py`, and `parametrics_old.py` contain transformations between microlensing light-curve parameters and physical parameters.
- `EarthMotion.py`, `EarthMotion_tmp.py`, and `calc_vEarth.py` handle Earth velocity/motion support used by the parameter transformations.
- `pre_runner.py` is the first wrapper around external Genulens `pre_gapmoe`.
  - It accepts RA/Dec and converts to galactic `(l, b)` without snapping to the existing 0.2 degree histogram grid.
  - Coordinate aliases are supported:
    - Equatorial: `ra_deg/dec_deg` or `ra/dec`.
    - Galactic: `l_deg/b_deg`, `l/b`, `glon/glat`, `gal_l/gal_b`, or `galactic_l/galactic_b`.
    - Equatorial and Galactic inputs must not be mixed in the same call.
  - It calls external `calc_mass_dist`, `calc_rho_profile`, and `calc_murel_dist`.
  - It writes run-local files: `mass.dat`, `rho.dat`, `murel.dat`, and `manifest.json`.
  - Source magnitude/extinction options are represented by `SourceSelection`.
  - Genulens location is user-dependent. `PreRunner` resolves it from `genulens_root=...`, `GAPMOE_GENULENS_ROOT`, `GENULENS_ROOT`, or nearby default candidates.
- The Genulens path may point either to the Genulens repository root or directly to its `pre_gapmoe/` directory.
  - `calc_murel_dist` has no `t0` or Earth-velocity option; its murel output is treated as heliocentric relative proper motion from the Galactic lens/source kinematics.

## PreRunner smoke test

2026-05-02:

- Ran `PreRunner.run()` with `ra_deg=270.0`, `dec_deg=-30.0`, `run_name="small"`, and reduced grids/MC settings.
- This executed all three external binaries through `subprocess.run`: `calc_mass_dist`, `calc_rho_profile`, and `calc_murel_dist`.
- Output directory: `/tmp/gapmoe_prerunner_smoke/small`
- Outputs:
  - `mass.dat`: 1006 lines
  - `rho.dat`: 9 lines
  - `murel.dat`: 225 lines
  - `manifest.json`: 68 lines
- The manifest recorded unsnapped coordinates `l=0.6739000406866593`, `b=-3.236225767885934`.

2026-05-02 coordinate alias check:

- Confirmed `PreRunner` accepts `ra/dec`, `ra_deg/dec_deg`, `l/b`, `glon/glat`, `gal_l/gal_b`, and `galactic_l/galactic_b`.
- Confirmed `glon > 180` is normalized to Genulens-style negative/positive longitude by subtracting 360.
- Confirmed validation errors for mixed RA/Dec plus l/b input, missing Dec, and missing Galactic latitude.

2026-05-02 coordinate value format check:

- Coordinate system selection is explicit from argument names; there is no automatic inference between RA/Dec and Galactic l/b beyond the provided names.
- RA values accept numeric degrees, degree strings such as `270 deg`, and hour-angle strings such as `18:00:00` or `18h00m00s`.
- Dec/l/b values accept numeric degrees, degree strings such as `-30 deg`, and sexagesimal degree strings such as `-30:00:00` or `-30d00m00s`.
- Duplicate aliases are allowed only when they resolve to the same numeric value, e.g. `ra_deg=270.0` and `ra="18:00:00"`.
- Conflicting aliases, mixed coordinate systems, and incomplete coordinate pairs raise `ValueError`.

Important public-readiness issue:

- `gapmoe.py` and `gapmoeJax.py` hard-code `/moao38_7/nunota/gapmoe` data paths. Public usage needs configurable data roots.
- There is no package metadata yet (`pyproject.toml`, install requirements, README usage path, etc.).

## Existing local histogram paths

Current `gapmoe.py` and `gapmoeJax.py` read these paths:

- Mass histogram: `/moao38_7/nunota/gapmoe/ML_hist/ML_hist.dat`
- Density histograms: `/moao38_7/nunota/gapmoe/rhos_hist/rhos_hist_{l:.1f}_{b:.1f}.dat`
- Relative proper-motion histograms: `/moao38_7/nunota/gapmoe/murel_hist/murel_hist_{l:.1f}_{b:.1f}.dat`

`{l, b}` are computed from RA/Dec and snapped to the nearest 0.2 degree grid point. A local scan found 13,334 files across these three histogram directories.

## Removed `src/genulens`

This directory used to contain old C sources, object files, and compiled executables:

- Sources: `calc_mass_distribution_each.c`, `calc_rhon_at_Dlb.c`, `murel_sampling.c`, `murel_sampling_2d.c`, `genulens_helio.c`, `option.c`, `random.c`.
- Headers: `option.h`, `random.h`.
- Build script: `compile.sh`.
- Generated artifacts: `*.o` and compiled binaries.

It was removed because public GAPMOE should use external Genulens `pre_gapmoe` through `PreRunner`, not vendor a stale local Genulens copy.

## External Genulens source

There is a sibling repository at `../genulens`, with remote:

- `origin`: `git@github.com:NunotaKansuke/genulens.git`
- `upstream`: `https://github.com/nkoshimoto/genulens.git`

The relevant replacement implementation is `../genulens/pre_gapmoe/`.

`pre_gapmoe` contains C++ tools and a Makefile:

- `calc_rho_profile`
- `calc_mass_dist`
- `calc_murel_dist`
- Shared code: `galactic_model.*`, `galactic_kinematics.cpp`, `option.*`
- Build dependency: GSL via `gsl-config`

Current `../genulens` has uncommitted changes, so GAPMOE should not overwrite or vendor from it blindly until the ownership boundary is decided.

For public usage, users should provide their own Genulens checkout path rather than relying on this local sibling path.

## Recommended direction

1. Keep GAPMOE as the Python package that generates and consumes per-event histograms.
2. Move preprocessing responsibility to external Genulens `pre_gapmoe`.
3. Continue hardening `PreRunner`:
   - points to an external `pre_gapmoe` checkout or installed command directory;
   - validates the three required executables exist;
   - runs `calc_rho_profile`, `calc_mass_dist`, and `calc_murel_dist`;
   - writes outputs into a configurable run directory;
   - records command lines and versions for reproducibility.
4. Replace hard-coded data paths in `gapmoe.py` and `gapmoeJax.py` with direct paths returned by `PreRunner`.
5. Add package metadata and a minimal smoke test before publishing.

## Initial commit policy used here

The initial commit should track source and working notebooks/scripts, but ignore generated local artifacts:

- Python bytecode and cache directories.
- Jupyter checkpoints.
- C/C++ object files and compiled binaries.
- Local generated data/output directories such as `ML_hist/`, `murel_hist/`, `rhos_hist/`, `test_data/`, `test_result/`, and `result/`.
