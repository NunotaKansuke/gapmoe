# Flow Release Handover

Date: 2026-07-14

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

## Packaged Artifacts

- `src/gapmoe/data/flows/default/event_kernel/flow.eqx`
- `src/gapmoe/data/flows/default/event_kernel/config.json`
- `src/gapmoe/data/flows/default/source_distance_grid.npz`
- `src/gapmoe/data/flows/default/manifest.json`

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
```

## Verification Performed

- `pytest -q`: 116 passed.
- A wheel was built and installed into an isolated target directory.
- From that wheel, `Model().set(...).set_flow().galactic_model(...)` loaded
  the artifact, evaluated a finite density, and produced a kernel sample.

## Follow-up Work

- Extend the Flow coverage or publish separate releases for `REMNANT=1` and
  `BINARY=1` as needed.
- Revisit the finite-proposal event-rate sampler if it becomes part of a
  precision population simulation workflow.
- The isochrone latent model and `theta_*` inference are deliberately outside
  this release.
