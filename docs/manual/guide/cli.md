# Command line

`panda --help` and `panda COMMAND --help` list every option. The command line builds the same `FitSpec` and
calls the same functions as the Python API.

| Command | Does |
|---------|------|
| `panda shapes [--pore slit]` | pore types, their fields, shapes, parameters, bounds, regimes, axes |
| `panda profile TOP TRAJ... -o profiles.npz` | density profiles by time blocks (needs `panda-md[md]`) |
| `panda fit PROFILE` | fit with one shape or `--shape auto` (every shape, best first); `--blocks` also fits every block |
| `panda run CONFIG.toml` | profile + fit from one settings file |

## Pore

`--pore-file pore.json` (a JSON of the pore fields, optionally with `"type"`) and/or `-g FIELD=VALUE`
(`-g width_nm=8 -g length_nm=16`). Without dimensions `panda profile` measures the slit from the wall atoms
(`--walls "selection"`) and `--pore-output pore.json` saves it for `panda fit`. `--walls z1,z2` fixes the wall
positions instead (the pore dimensions are then required).

## Parameters

The same for every shape: `--optimize theta,delta`, `--fix phi=0.5`, `--bounds theta=100:180`,
`--rho-bulk 33.1`, `--regime auto|wetting|non_wetting`, `--normalization`, `--loss`.

## Output

`-o result.json` holds `{"fit": …, "blocks": …, "comparison": …}`: the fit of the (block-averaged) profile, the
fits of the blocks with `--blocks`, the ranking with `--shape auto`. `--plot fit.png` saves the figure
(needs `panda-md[plot]`). A `.csv` profile from another program needs no header: columns are the coordinate (nm),
the density (nm⁻³) and optionally its error.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | input error (options, files, settings) — a one-line message, nothing computed |
| 2 | the fit finished but raised problem flags (see Fitting) |
| 130 | interrupted |

## Settings file

```toml
--8<-- "examples/panda.toml"
```

Paths are relative to the settings file. `[profile]` takes either `input` (a saved profile) or `topology`,
`trajectory` and `select`; `[fit]` takes every `FitSpec` field plus `shape`, `blocks`, `output`, `plot`.
