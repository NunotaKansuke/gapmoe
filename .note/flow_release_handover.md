# Flow Release Handover

Date: 2026-07-16

## Shipped State

The bundled `default` Flow release is ready for use through the public
`Model` API. It covers:

- `-5 <= l <= 5` deg;
- `-6 <= b <= -2` deg;
- `REMNANT=0`, `BINARY=0`.

The packaged model is an event-kernel Flow for

```text
p(ML, DL, mu_N, mu_E | DS, source_group, l, b).
```

`DS` and source-group weights remain in the source-distance/isochrone layer.
This keeps CMD selection independent of Flow training. The default
`GalaxyModel` adds the event-rate factor to `log_density`.

The bundled `rate-included-v1` release covers the same sky and model options,
but targets the complete genulens event measure. It is the release for
high-precision population Monte Carlo and exponential tilting. Its kernel is
trained directly from raw `wtj`, and its component-resolved source-distance
grid is the matching `wtj` marginal. No rate factor is applied at inference.

## Public API

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
    theta,
    magnitudes={"Imag": i_s, "Vmag": v_s},
)
sample = prior.sample(key)
sample_given_photometry = prior.sample(
    key,
    magnitudes={"Imag": i_s, "Vmag": v_s},
)
```

Select the population-MC release explicitly:

```python
model.set_flow(release="rate-included-v1")
prior = model.galactic_model(isochrone)
sample = prior.sample(key)  # direct event-measure draw
```

The canonical parameter order is `(ML, DL, DS, mu_N, mu_E)`, with masses in
solar masses, distances in kpc, and proper motions in mas/yr.

`prior.log_density(theta, magnitudes=...)` conditions the 5D prior on source
photometry but deliberately does not add a photometric prior. Add
`prior.log_source_density(ds=theta[2], magnitudes=...)` only when the analysis
wants to include the photometry itself as a source prior.

`prior.source_radius(ds=..., magnitudes=...)` returns the source-radius summary.
`prior.sample_kernel(key, ds=..., source_group=...)` remains the low-level
fixed-`DS`, fixed-group diagnostic sampler. Source groups are thin, thick,
bulge, NSD, halo in indices 0 through 4.

`prior.sample(...)` first samples `(DS, source_group)` from either the hard
CMD selection or supplied photometry, then samples the Flow kernel. With the
default event-rate weighting it uses 256 Flow proposals and importance
resampling by the same event-rate factor as `log_density`, approximating the
event-rate-weighted distribution. `num_proposals` can be increased for
diagnostic sampling, but this partially rate-removed release is not an efficient bulk
proposal for high-precision, rate-weighted or exponentially tilted Monte
Carlo. If `include_event_rate=False` was passed to `galactic_model`, it draws
directly from the base kernel.

The Flow path intentionally does not require a local genulens checkout. The
histogram path remains:

```python
model.prepare("runs/event-001")
```

and needs the installed genulens Python API or a local checkout for the
preprocessing step. `Model().resume(directory)` restores a prepared histogram
run.

For `rate-included-v1`, `include_event_rate=False` is invalid: the event-rate
factor is part of both learned factors and cannot be removed after training.

## Packaged Artifacts

- `src/gapmoe/data/flows/default/event_kernel/flow.eqx`
- `src/gapmoe/data/flows/default/event_kernel/config.json`
- `src/gapmoe/data/flows/default/source_distance_grid.npz`
- `src/gapmoe/data/flows/default/manifest.json`
- `src/gapmoe/data/flows/rate-included-v1/event_kernel/flow.eqx`
- `src/gapmoe/data/flows/rate-included-v1/event_kernel/config.json`
- `src/gapmoe/data/flows/rate-included-v1/source_distance_grid.npz`
- `src/gapmoe/data/flows/rate-included-v1/manifest.json`

`pyproject.toml` includes these files as package data. `Model` creates the
genulens runner lazily when no `genulens_root` is explicitly supplied, so a
wheel-only Flow installation works.

## Validation

Training used 15 million balanced samples, three million per source group.
The source data used `NSD=1`, `SMALLGAMMA=1`, `REMNANT=0`, and `BINARY=0`.
The training weights removed `thetaE * mu_rel`; the conditional kernel retains
the genulens `DL**2` lens-area factor. `GalaxyModel` consequently applies only
`thetaE * mu_rel` at inference time.

The companion source-distance grid is deliberately sampler-compatible rather
than a physical volume-density table: for the unselected `gammaDs=0.5`
configuration it is `nMS * sqrt(DS / 8000) * 1e-3` times the integrated
`DL**2`-weighted total lens-number-density column to `DS`. This is the measure
that remains after removing `thetaE * mu_rel` from genulens event weights.

Independent midpoint validation used new genulens seeds:

- normal thin/thick/bulge components: 27 cells, KS medians `0.044-0.051`,
  maximum `0.110`;
- NSD: four midpoint cells, 5,000 samples per cell, KS maximum `0.044`;
- halo: four midpoint cells, 5,000 samples per cell, KS maximum `0.046`.

The raw validation products are intentionally outside git under:

```text
flow_mvp/runs/release_v2_base_20260713_135000/independent_validation/
flow_mvp/runs/rate_included_v1_galaxy_validation_20260716.json
flow_mvp/runs/rate_included_v1_conditional_validation_20260716/
```

### `rate-included-v1`

The release is a source-group expert model, with no inference-time correction
weights. Thin disk, thick disk, and bulge use the original raw-`wtj` kernel,
whose training table actually covers all 189 grid points over `l=[-5,5]` and
`b=[-6,-2]`. Earlier downstream notes describing it as `l=+-4` were incorrect.
NSD and halo use the 15-million-sample balanced raw-`wtj` expert (three million
per source group), because the original table contained no NSD examples. Both
experts are normalized conditional densities; selecting one by source group
preserves the joint event measure and is not importance reweighting.

The matching source grid was regenerated from independent raw `wtj` tables
with `NSD=1`, `SMALLGAMMA=1`, `REMNANT=0`, and `BINARY=0`. The complete
source-grid × expert comparison on 270,000 independent genulens events at nine
midpoint sightlines gives
maximum KS values: DS 0.03310, ML 0.02204, DL 0.02047, mu_E 0.01687, and mu_N
0.01834. Derived `DL/DS`, `mu_rel`, `theta_E`, and `t_E` have maximum KS
0.02464; the maximum Spearman-correlation-matrix difference is 0.03025 and the
largest source-group fraction difference is 0.01034.

Conditional holdouts test the learned kernel separately from the source grid.
For thin disk, thick disk, and bulge over nine held-out sightlines, the maxima
are 0.04337 (marginal KS), 0.04070 (derived-observable KS), and 0.07121
(rank-correlation difference). Independent forced-component holdouts give
0.03730/0.03860/0.03297 for NSD and 0.03130/0.03790/0.05693 for halo. All are
below the predeclared 0.05/0.05/0.10 limits. The complete package therefore
passes directly as a Flow approximation to the genulens Galactic event model.

N24 was useful downstream for exposing the earlier event-measure mismatch,
but it is not a release dependency or the acceptance definition for gapmoe.
The shipped evidence is the independent direct comparison above. The
fresh `asinh(mu)` and globally balanced candidates remain reproducible rejected
runs: neither improves the complete direct validation enough to replace the
full-grid main kernel plus the more accurate rare-group expert.

## Verification Performed

- `pytest -q`: 129 passed.
- A wheel was built and installed into an isolated target directory.
- From that wheel, both `rate-included-v1` experts loaded, the nested rare-group
  artifact was present, and the public model evaluated a finite density and
  produced a kernel sample.

## Follow-up Work

- Extend the Flow coverage or publish separate releases for `REMNANT=1` and
  `BINARY=1` as needed.
- Revisit the finite-proposal event-rate sampler if it becomes part of a
  precision population simulation workflow.
- The isochrone latent model and `theta_*` inference are deliberately outside
  this release.
