# Tests

```bash
make test          # all tests (uv run pytest)
make test-env      # the same in a clean environment: the package with [md, plot] and pytest only
```

Tests that need optional data or packages are skipped with a reason (for example, contract checks of quantities a
shape does not derive). Run time: about a minute.

| Suite | Files | What is checked |
|-------|-------|-----------------|
| parameters, specification | `unit/test_parameters.py` | parameter description and bounds, degrees ↔ radians, free / fixed / bounds / initial rules, readable errors |
| profiles | `unit/test_profile.py` | array checks, normalization by ρ_bulk and by the mean density, save / load (.npz, .csv), version in files |
| registry, models | `unit/test_pore_registry.py`, `unit/test_model.py`, `unit/test_model_guards.py` | shapes register without core changes, unknown names, regimes and their boundaries, formulas never called outside their domain |
| contract | `contract/test_shapes.py` | for **every** registered shape: ∫ρ dz = φ, 0 ≤ ρ ≤ 1, centre of mass, 3D shape vs profile, positive surface areas |
| slit with base L × kL | `unit/test_slit_rectangular_cell.py` | the six slit shapes against the closed formulas of Shistko et al. (2025) |
| regression | `regression/test_reference_formulas.py` | profiles, φ limits, existence conditions, areas and stability diagrams equal the reference implementations (the previous PANDA, commit 93c6dbb; PANDA_Polygon, commit 38ff491) to 1e-12; reference numbers in `regression/data/` |
| fitting | `unit/test_fitting.py`, `unit/test_fitting_details.py` | one optimizer for any number of free parameters, problem flags, wetting regimes, barrier, φ from the data, profile / pore mismatch |
| polygon pores | `unit/test_polygon_shapes.py` | shapes with defects in the source are not ported; strict boundaries |
| stability diagrams | `unit/test_stability.py` | the shape with the lowest capillary energy wins; regimes without formulas; parameter bounds |
| data path | `unit/test_trajectory.py`, `unit/test_density_profile.py`, `unit/test_molecules_and_offset.py`, `unit/test_pbc_and_time.py` | reading trajectories, slit from the wall atoms across periodic boundaries, pore and liquid centering, time blocks, per-molecule profiles, wall offset, warnings |
| 3D shapes and maps | `unit/test_shapes_3d.py`, `synthetic/test_point_clouds.py` | `contains` against the profile, symmetry axes, axial centering, 2D maps, bulk density from the map |
| analysis | `unit/test_analysis.py` | fit per block, statistical inefficiency, shape comparison and indistinguishable shapes |
| synthetic data, accuracy, plots | `unit/test_synthetic_profiles.py`, `unit/test_accuracy_and_plots.py` | synthetic profiles behave like MD histograms; accuracy study; every plot function |
| recovery | `synthetic/test_recovery.py`, `synthetic/test_data_path.py` | shape → profile with noise → fit recovers θ (16 cases over the three pores; trajectories of point clouds) |
| real MD | `md/test_real_systems.py` | saved profiles and single frames of three MD systems (graphite / water + decane, benzene on SOLD, calcite) in `md/data/`: walls found as in the manual calculation, water θ ≈ 171°, wetting benzene flagged |
| command line | `cli/test_cli.py` | `panda shapes / profile / fit / run` give the same answer as the API; exit codes 0 / 1 / 2; readable errors |
| pre-release audit | `unit/test_audit_minors.py` and tests marked “Pre-release audit” | regressions of the issues found before 0.1.0 |
| examples, layering, packaging | `test_examples.py`, `unit/test_layers.py`, `test_package.py` | examples run; notebooks match their builders; the core imports no pores, trajectories or plotting; `import panda_md` loads neither MDAnalysis nor matplotlib |

The reference data in `regression/data/` and `md/data/` were produced once from the reference implementations and
from the MD trajectories (which are not part of the repository); the tests only read them.
