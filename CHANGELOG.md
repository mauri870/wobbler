# Changelog

## 0.1.0 - 2026-11-24

First release.

- `simulate_fit`, `observe` and `fit_five_parameter`: the batched forward model of Gaia's five-parameter fit for stars with a companion, on a CPU or a GPU. It gives the RUWE, the excess noise and the fitted parallax of many orbits at once, in the DR3 and DR2 windows.
- The unit weight error is calibrated to Gaia DR3 by default: sqrt(f(G)^2 + B(G)^2 (w^2 - 1)) for a raw value w, with `wobbler.uwe_factor` (f, from DR3 single stars), `wobbler.orbit_factor` (B, from 60,940 DR3 orbit stars) and `wobbler.calibrated_uwe`. `uwe_factor=1` gives the raw simulation.
- `simulate_fit` with `device=None` runs on the GPU from 700 cases a call when PyTorch sees one, and on the CPU below.
- `Scan.from_gost`, `Scan.from_position`, `wobbler.sources.scan_from_source_id` and `wobbler.sources.scan_from_epoch_astrometry` (provisional, for the draft DR4 data model).
- A companion that gives light: `flux_ratio` and `primary_mass`.
- `wobbler.light`: orbit and mass functions in NumPy, with no PyTorch.
- `wobbler.joker`: rejection sampling of orbits from radial velocities.
- `wobbler.epoch_astrometry.grid_search`: a search for an astrometric orbit in along-scan positions.
- `wobbler.results.to_table`: the results as a table.
- `simulate_fit(..., parameters=True)` adds the fitted proper motion (`pmra_mas_yr`, `pmdec_mas_yr`) and the formal errors of the parallax and the proper motion (`parallax_error_mas`, `pmra_error_mas_yr`, `pmdec_error_mas_yr`) to the result of each window. The default result is unchanged.
- gaiamock's unbinned mode, off by default: `simulate_fit(..., binned=False, position=(ra, dec))`, `fit_five_parameter(..., binned=False)`, `Scan.from_position(..., binned=False)` and `wobbler.unbinned` (one measurement per CCD, the epoch errors rescaled by the median RUWE of the sky position). The map `healpix_16_med_ruwe.npz` is not shipped; `wobbler.unbinned` says how to fetch it. With `binned=False` the default `uwe_factor` is `wobbler.unbinned_factor(G)`, the median ratio of Gaia DR3's RUWE to the unbinned mode's in bands of G from 9 to 16, fitted on 19,600 DR3 single stars; `uwe_factor=1` gives the raw simulation. The mode is for single stars and weak orbits.
