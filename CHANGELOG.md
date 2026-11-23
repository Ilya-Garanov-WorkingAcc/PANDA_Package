# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) (0.x: the API may still change between minor versions).

## [0.1.1] — 2026-11-23

### Fixed

- The command line no longer fails with `UnicodeEncodeError` when its output goes to an encoding without θ, ≈ or →
  (on Windows, output redirected to a file or a pipe in the ANSI code page, e.g. cp1251): such characters are written
  as escapes (`θ`). Set `PYTHONUTF8=1` to get them as UTF-8.

### Changed

- The repository is renamed to `PANDA_Package`; the project links point to the new address.

## [0.1.0] — 2026-11-23

First release of the rewritten library (`import panda_md`). The method follows Semenchuk, Kondratyuk and
Kopanichuk, Colloids Surf. A 708 (2025) 135994.

### Added

- Revised polygon morphologies (G. Vandyshev): the square pore gains `filament_w4c0`–`filament_w4c3`,
  `filament_w3c2`, `filament_w2c0_opp`, `filament_w2c1`, `capsule_w4c4` and `perforation` (16 shapes); the
  triangular pore gains `droplet_w2c1` and `capsule_w3c3` (14 shapes), and every triangle shape except `droplet_w3c0`
  now has a density profile and can be fitted.
- Cylindrical pore `CylinderPore` (`cylinder`) with the shape `capsule` — a plug with two spherical menisci, θ > 90°
  (formulas: alpha.py `rollx`); `CylinderPore.from_trajectory` measures a (multi-walled) tube from the wall atoms; the
  command line measures any pore that can be measured (`Pore.measure_from_trajectory`), and the per-frame wall check
  is a method of the pore (`Pore.wall_size_check_nm`).
- Core: a pore with its shapes and a single parameter description (`Parameter`, `FitSpec`): free / fixed /
  bounds / initial for every parameter, units of the user (θ in degrees, δ in units of the pore size), checks
  with readable errors; one generic fit (multistart, L1 or L2 loss, normalization by the bulk density or by the
  mean density, wetting regimes kept apart, problem flags in `FitResult`).
- Slit pore with a rectangular base L × kL and six shapes (droplet, doughnut, worm, roll, perforation, layer);
  triangular pore (14 shapes) and square pore (16 shapes); stability diagrams by the capillary functional.
- New shapes and pores are added by a decorator without changes to the core (`examples/05_custom_shape.py`).
- Data path (extra `md`, MDAnalysis): slit dimensions from the wall atoms across periodic boundaries, density
  profiles centred on the pore or on the liquid, time blocks, per-molecule profiles, wall offset, 2D density maps
  with axial centering for shapes with a symmetry axis.
- Analysis: fit of every block (mean ± error), comparison of all shapes of a pore, statistical inefficiency.
- Synthetic profiles and point clouds for tests and validation; accuracy study of the method
  (`validation/REPORT.md`).
- Plots (extra `plot`, matplotlib): density profile with the error over blocks, fit, blocks, density map, stability
  diagram, accuracy.
- Bulk density of a liquid from its 2D map (`DensityMap.bulk_density`, robust to layering peaks and to menisci
  smeared by a moving shape); pores print their dimensions readably.
- `load_trajectory` opens LAMMPS runs (`.data` of `atom_style atomic` + `.lammpstrj`) and passes MDAnalysis
  options through (`dt`, `atom_style`, `format`, …); a file MDAnalysis cannot read gives an error with a hint
  (e.g. a `.tpr` from a newer GROMACS → use the `.gro`).
- `examples/demo.ipynb`: the whole analysis of a real trajectory, step by step, with figures.
- Command line `panda`: `shapes`, `profile`, `fit`, `run CONFIG.toml`, `--version`; exit codes 0 / 1 / 2.
- Reproducibility: the PANDA version is stored in every `FitResult` and in saved `.npz` / `.csv` files.
- Shape comparison marks shapes whose losses differ by less than 2 % as indistinguishable by the profile.

### Not yet available

- Formulas of the wetting regime (θ < 90°) for the slit shapes and the cylinder capsule; the fit flags a result
  pressed to 90°.
- Surface areas of the slit doughnut and perforation (they do not enter stability diagrams).
- Density profile of the triangular `droplet_w3c0` (stability diagrams only).
- 3D shapes (`contains`) of the triangular and square pores; wall detection in polygonal pores.

### Changed

- Shape names of the polygon pores: `filament_w0c0_straight` is now `filament_w0c0_square` in the square pore and
  `filament_w0c0_triangle` in the triangular pore; the old name is an error. In the triangular pore
  `filament_w3c1` and `filament_w3c2` swapped their geometry so that the name counts the occupied corners
  (`w3c1`: one corner filled, two empty).

### Known limitations

- At exactly the boundary angle between two shapes of a polygonal pore (135° in the square, 120° in the
  triangle) neither neighbouring shape exists: the existence conditions of the source formulas are strict.
- In a narrow slit cell (k·l < π/4 at θ = 180°, δ = 0) the hole of a perforation never fits the cell: the
  shape does not exist there. The published lower volume-fraction limit is a spurious root in that case
  (hole wider than the cell, negative density); PANDA reports an empty interval.
- Near θ = 180° the profile depends weakly on θ: report the spread over time blocks (`fit_blocks`).

## [0.0.1] — 2026-10-02

Name reservation on PyPI; no code.
