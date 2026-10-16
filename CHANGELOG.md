# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor versions).

## [0.1.0] — unreleased

First release of the rewritten library (`import panda_md`). The method follows Semenchuk, Kondratyuk and
Kopanichuk, Colloids Surf. A 708 (2025) 135994.

### Added

- Core: a pore with its shapes and a single parameter description (`Parameter`, `FitSpec`): free / fixed /
  bounds / initial for every parameter, units of the user (θ in degrees, δ in units of the pore size), checks
  with readable errors; one generic fit (multistart, L1 or L2 loss, normalization by the bulk density or by the
  mean density, wetting regimes kept apart, problem flags in `FitResult`).
- Slit pore with a rectangular base L × kL and six shapes (droplet, doughnut, worm, roll, perforation, layer);
  triangular pore (12 shapes) and square pore (7 shapes); stability diagrams by the capillary functional.
- New shapes and pores are added by a decorator without changes to the core (`examples/05_custom_shape.py`).
- Data path (extra `md`, MDAnalysis): slit dimensions from the wall atoms across periodic boundaries, density
  profiles centred on the pore or on the liquid, time blocks, per-molecule profiles, wall offset, 2D density maps
  with axial centering for shapes with a symmetry axis.
- Analysis: fit of every block (mean ± error), comparison of all shapes of a pore, statistical inefficiency.
- Synthetic profiles and point clouds for tests and validation; accuracy study of the method
  (`validation/REPORT.md`).
- Plots (extra `plot`, matplotlib): fit, blocks, density map, stability diagram, accuracy.
- Command line `panda`: `shapes`, `profile`, `fit`, `run CONFIG.toml`; exit codes 0 / 1 / 2.

### Not yet available

- Formulas of the wetting regime (θ < 90°) for the slit shapes; the fit flags a result pressed to 90°.
- Surface areas of the doughnut and the perforation (they do not enter stability diagrams).
- Cylindrical pore; 3D shapes (`contains`) of the triangular and square pores; wall detection in polygonal pores.

## [0.0.1] — 2026-10-02

Name reservation on PyPI; no code.
