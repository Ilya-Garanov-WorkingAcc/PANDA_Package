# PANDA

**P**redicting **A**ngle from **N**anoscale **D**ensity **A**nalysis — a Python library that determines the
contact angle and the interface shape of liquids confined in pores from one-dimensional density profiles
of molecular simulations.

> **Status: alpha (0.1.0).** Slit pore (six shapes, rectangular base L × kL), triangular pore (12 shapes) and
> square pore (7 shapes); fitting, the data path from MD trajectories (MDAnalysis, extra `md`), analysis over
> time blocks, stability diagrams, figures (extra `plot`) and the command line `panda`. Limitations: no formulas
> for the wetting regime θ < 90° of the slit shapes; see [CHANGELOG.md](https://github.com/Ilya-Garanov-WorkingAcc/PANDA_2.0/blob/main/CHANGELOG.md).

Documentation: `make docs` builds it from `docs/manual/` (mkdocs); examples: [examples/](https://github.com/Ilya-Garanov-WorkingAcc/PANDA_2.0/blob/main/examples/).

```python
import panda_md as panda
from panda_md.synthetic.profiles import synthetic_profile
pore = panda.SlitPore(width_nm=8.0, length_nm=16.0)
roll = pore.interface("roll")
profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=30.0)
result = roll.fit(profile, optimize=["theta", "delta"], density_bulk_per_nm3=30.0)
print(result)
```

From an MD trajectory (`pip install 'panda-md[md]'`):

```python
import panda_md as panda
traj = panda.load_trajectory("system.tpr", "system.xtc")
pore = panda.SlitPore.from_trajectory(traj, walls="resname GRAPHITE")
profiles = panda.density_profile(traj, pore, "resname WATER and name OW", start="2 ns", block="1 ns")
result = pore.interface("roll").fit(profiles.mean(), density_bulk_per_nm3=33.1)
spec = panda.FitSpec(density_bulk_per_nm3=33.1)
print(panda.fit_blocks(pore.interface("roll"), profiles, spec))  # θ по блокам, среднее ± ошибка
print(panda.compare_shapes(pore, profiles.mean(), spec))  # все формы поры по ошибке фита
water_map = panda.density_map(traj, pore, "resname WATER and name OW", shape="roll")  # ось валика в x = 0
```

Plots (`pip install 'panda-md[plot]'`): `from panda_md.plotting import plot_fit, plot_density_map` — and the
accuracy of the method on synthetic data is in `validation/REPORT.md`.

## Command line

The same pipeline from the shell (`panda --help`, `panda COMMAND --help`):

```bash
panda shapes --pore slit      # pore fields, shapes, their parameters, bounds, regimes, symmetry axes
# profiles by time blocks; without -g/--pore-file the slit is measured from the wall atoms
panda profile system.tpr system.xtc --walls "resname GRAPHITE" --select "resname WATER and name OW" \
      --start "2 ns" --block "1 ns" -o water.npz --pore-output pore.json
panda fit water.npz --pore-file pore.json --shape roll --rho-bulk 33.1 --blocks -o result.json --plot fit.png
panda fit water.npz --pore-file pore.json --shape auto --rho-bulk 33.1    # every shape of the pore, best first
panda run panda.toml          # profile + fit from one settings file
```

Model parameters are set the same way for every shape: `--optimize theta,delta`, `--fix phi=0.5`,
`--bounds theta=100:180`. Pore dimensions: `--pore-file pore.json` or `-g width_nm=8 -g length_nm=16`.
`-o result.json` holds `{"fit": …, "blocks": …, "comparison": …}`: the fit of the (block-averaged) profile, the
fit of every block with `--blocks`, the shape ranking with `--shape auto`. A `.csv` profile from another program
needs no header: columns are coordinate (nm), density (nm⁻³) and, optionally, its error.
Exit code: 0 — success, 1 — input error, 2 — the fit finished but raised problem flags (at a bound, outside the
shape's existence domain, θ at 90°, optimizer failure).

`panda.toml` (paths are relative to the file):

```toml
[pore]
type = "slit"
wall_selection = "resname GRAPHITE"   # no dimensions: measured from the walls

[profile]
topology = "system.tpr"
trajectory = ["system.xtc"]
select = "resname WATER and name OW"
start = "2 ns"
block = "1 ns"
output = "water.npz"
pore_output = "pore.json"

[fit]                                 # every FitSpec field, plus shape, blocks, output, plot
shape = "roll"
density_bulk_per_nm3 = 33.1
blocks = true
output = "result.json"
```

## Method

Two immiscible liquids in a pore form one of a few interface shapes (droplet, doughnut, worm, roll,
perforation, layer). For each shape the density profile across the pore is known analytically as a
function of the contact angle θ, the wetting-layer thickness δ and the volume fraction φ. PANDA fits this
analytical profile to the profile measured in a simulation, so the position of the interface never has to
be located explicitly.

References:

- A. Semenchuk, N. Kondratyuk, I. Kopanichuk, *PANDA: Predicting angle from nanoscale density analysis*,
  Colloids Surf. A 708 (2025) 135994. https://doi.org/10.1016/j.colsurfa.2024.135994
- A. Semenchuk, N. Kondratyuk, K. Gerke, I. Kopanichuk, *Towards reproducible wetting studies: Automated
  contact angle determination by molecular simulations*, Colloids Surf. A 725 (2025) 137586.
  https://doi.org/10.1016/j.colsurfa.2025.137586

## Installation

```bash
pip install panda-md
```

```python
import panda_md
print(panda_md.__version__)
```

## Development

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
make install     # create .venv with the package and the dev tools
make check       # black + ruff + mypy + pytest (the same checks CI runs)
make test-env    # run tests in a separate clean environment (package + test group only)
make build       # build sdist and wheel into dist/
make wheel-check # install the built wheel into an isolated environment and import it
make help        # all commands
```

Dependency groups in `pyproject.toml`: `test` (pytest), `lint` (black, ruff, mypy), `dev` (both, installed by
default). They are not part of the published package.

## Repository layout

```
src/panda_md/   the package (typed, ships py.typed)
tests/          pytest tests (what each suite checks: tests/README.md)
examples/       runnable scripts and a settings file for `panda run`
validation/     accuracy of the method on synthetic data (REPORT.md, figures)
docs/manual/    user documentation (mkdocs)
docs/           design documents: audit of the previous version, architecture specification
smoke/          exploratory scripts run against the previous PANDA version
ROADMAP.md      project plan
```

## Previous version

The original research code lives at https://github.com/FlufffyMelon/PANDA.

## How to cite
Cite the method paper (Semenchuk, Kondratyuk, Kopanichuk, Colloids Surf. A 708 (2025) 135994) and the software;
metadata for both are in [CITATION.cff](https://github.com/Ilya-Garanov-WorkingAcc/PANDA_2.0/blob/main/CITATION.cff).

MIT, see [LICENSE](https://github.com/Ilya-Garanov-WorkingAcc/PANDA_2.0/blob/main/LICENSE).
