# MB21333: spurious luminous-giant prior from an isochrone discontinuity

## Status

**Blocking correctness issue.**  Do not use a source-CMD prior built through
`GenulensCmdPriorBuilder` for quantitative Galactic inference until the
underlying `genulens` isochrone lookup is corrected and the CMD table is
rebuilt.  This was found while analysing MB21333 in July 2026.

The problem is in the stellar-isochrone lookup used by both the ordinary
random source sampler and the deterministic `imf_quadrature` API.  The latter
made the problem reproducible; it did not create it.

## Symptom in MB21333

The unconditioned rate-included Flow source-distance measure at
`(l,b)=(-4.1508,-3.1648)` is sensible: 71.5% of its mass lies at
`7 < DS < 10 kpc`, with median `DS = 9.25 kpc`.  After multiplying by the
source CMD density in the observed bright-source region, a false distant
giant mode appeared around `DS = 12--15 kpc`.

For the intrinsic CMD rectangle used during diagnosis,

```
0.9 < (V-I)_0 < 2.3
12.3 < I_0 < 13.9
```

the model gave the following conditional probabilities:

| DS (kpc) | P(CMD rectangle | DS) |
| ---: | ---: |
| 9 | 1.11e-4 |
| 14 | 9.81e-3 |
| 15 | 1.41e-2 |

The factor of about 100 enhancement at large distance is not a physical
luminosity-function effect.  It is driven by the thick-disk CMD component.
For example, at 14 kpc that component assigns 3.52% probability to this
very bright rectangle.

## Reproduction

With the current `genulens` development binding, draw deterministic IMF
quadrature points for the first thick-disk age/metallicity population:

```python
generator = genulens.ForwardSourceGenerator.load_default_for_bands(["Vmag", "Imag"])
point = genulens.SourcePopulationPrior.points_for_component(7)[0]
query = genulens.ForwardSourceQuery()
query.component_index = 7
query.distance_pc = 10.0
query.min_initial_mass_msun = 0.09
query.max_initial_mass_msun = 1.0
query.use_default_log_age = False
query.log_age = point.log_age
query.use_default_metallicity = False
query.metallicity_mh = point.metallicity_mh
rows = generator.imf_quadrature(query, 8192).to_numpy()
```

Near the old-population turnoff, the selected isochrone contains a large
initial-mass discontinuity.  Representative returned rows are:

| quadrature index | returned initial mass (Msun) | M_I | radius (Rsun) |
| ---: | ---: | ---: | ---: |
| 7600 | 0.8072 | 2.827 | 1.79 |
| 7800 | 0.8222 | -3.103 | 43.96 |
| 8000 | 0.8222 | -3.103 | 43.96 |
| 8100 | 1.0948 | 29.024 | 2.27e-5 |

About 5.27% of the 8192 equal-IMF-probability draws land in
`-4 < M_I < -3`.  This is an artificial concentration at the RGB tip.

## Root cause

`genulens::model::IsochroneGrid::lookup` brackets a requested initial mass by
adjacent isochrone rows.  When `continuous_isochrone_segment(lo, hi)` is
false, it chooses `lo` or `hi` by a midpoint rule:

```cpp
if (!continuous_isochrone_segment(*lo, *hi)) {
    t = (t < 0.5) ? 0.0 : 1.0;
}
```

For the RGB-tip-to-white-dwarf discontinuity, this assigns a broad interval
of initial masses to the final luminous RGB row.  Sampling the IMF over that
interval therefore treats a short-lived evolutionary state as a several-%
stellar population.

## Required genulens fix

The fix belongs in `genulens`, not in FlowJAX or the gapmoe likelihood:

1. Treat non-continuous isochrone gaps explicitly when mapping an initial
   mass to a stellar state; do not midpoint-assign their mass measure to
   either endpoint.
2. Define and test the intended handling of post-turnoff/remnant masses.
3. Add a regression test for component 7 at its old-population age verifying
   that the RGB-tip mass fraction is not inflated by the adjacent WD gap.
4. Rebuild the source CMD table and rerun MB21333 only after the new binding
   passes this test.

## Consequences for existing results

All MB21333 products that use the current source-CMD prior are invalid for
quantitative interpretation, including the source-only CMD comparison and
the direct Flow Galactic MCMC/corner plots.  The separately trained
light-curve likelihood flows are unaffected.
