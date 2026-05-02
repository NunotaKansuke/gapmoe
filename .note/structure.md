# GAPMOE repository structure notes

Date: 2026-05-02

## Current state

- This directory was not a git repository before this cleanup pass.
- Main Python code lives under `src/gapmoe/`.
- A local copy of old Genulens C preprocessing tools lives under `src/genulens/`.
- Exploratory scripts and notebooks live under `test_tool/`.
- Runtime histogram data is expected at absolute paths under `/moao38_7/nunota/gapmoe/`, but those data directories are not present in this checkout.

## `src/gapmoe`

- `gapmoe.py` is the NumPy implementation of the galactic prior lookup.
  - It converts input RA/Dec to galactic `(l, b)`.
  - It snaps `(l, b)` to 0.2 degree bins.
  - It loads three generated histogram products: mass, density along distance, and relative proper motion.
  - It exposes density lookups and `log_galactic_prior`.
- `gapmoeJax.py` mirrors the same model with JAX arrays and jitted prior/gradient methods.
- `parametrics.py`, `parametrics2.py`, and `parametrics_old.py` contain transformations between microlensing light-curve parameters and physical parameters.
- `EarthMotion.py`, `EarthMotion_tmp.py`, and `calc_vEarth.py` handle Earth velocity/motion support used by the parameter transformations.

Important public-readiness issue:

- `gapmoe.py` and `gapmoeJax.py` hard-code `/moao38_7/nunota/gapmoe` data paths. Public usage needs configurable data roots.
- There is no package metadata yet (`pyproject.toml`, install requirements, module `__init__.py`, README usage path, etc.).

## `src/genulens`

This directory currently contains old C sources, object files, and compiled executables:

- Sources: `calc_mass_distribution_each.c`, `calc_rhon_at_Dlb.c`, `murel_sampling.c`, `murel_sampling_2d.c`, `genulens_helio.c`, `option.c`, `random.c`.
- Headers: `option.h`, `random.h`.
- Build script: `compile.sh`.
- Generated artifacts: `*.o` and compiled binaries.

The public cleanup should remove generated artifacts from git and eventually remove this embedded copy entirely.

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

## Recommended direction

1. Keep GAPMOE as the Python package that consumes generated histograms.
2. Move preprocessing responsibility to external Genulens `pre_gapmoe`.
3. Add a small Python runner class in GAPMOE, tentatively `PreRunner`, that:
   - points to an external `pre_gapmoe` checkout or installed command directory;
   - validates the three required executables exist;
   - runs `calc_rho_profile`, `calc_mass_dist`, and `calc_murel_dist`;
   - writes outputs into a configurable GAPMOE data directory;
   - records command lines and versions for reproducibility.
4. Replace hard-coded data paths in `gapmoe.py` and `gapmoeJax.py` with a configurable data root.
5. Add package metadata and a minimal smoke test before publishing.

## Initial commit policy used here

The initial commit should track source and working notebooks/scripts, but ignore generated local artifacts:

- Python bytecode and cache directories.
- Jupyter checkpoints.
- C/C++ object files and compiled binaries.
- Local generated data/output directories such as `ML_hist/`, `murel_hist/`, `rhos_hist/`, `test_data/`, `test_result/`, and `result/`.
