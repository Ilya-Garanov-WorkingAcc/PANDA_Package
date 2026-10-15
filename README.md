# PANDA

**P**redicting **A**ngle from **N**anoscale **D**ensity **A**nalysis — a Python library that determines the
contact angle and the interface shape of liquids confined in pores from one-dimensional density profiles
of molecular simulations.

> **Status: pre-alpha.** The core (parameters, pore/shape contract, generic fit) works for the slit pore
> (all six shapes: droplet, doughnut, worm, roll, perforation, layer; rectangular base L × kL), the triangular pore (14 shapes) and the square pore (7 shapes); stability diagrams by the capillary functional. Trajectory processing and CLI are
> not implemented yet; the published 0.0.1 contains no code.
```python
import panda_md as panda
from panda_md.synthetic.profiles import synthetic_profile
pore = panda.SlitPore(width_nm=8.0, length_nm=16.0)
roll = pore.interface("roll")
profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=30.0)
result = roll.fit(profile, optimize=["theta", "delta"], density_bulk_per_nm3=30.0)
print(result)
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
tests/          pytest tests
docs/           design documents: audit of the previous version, architecture specification
smoke/          exploratory scripts run against the previous PANDA version
ROADMAP.md      project plan
```

## Previous version

The original research code lives at https://github.com/FlufffyMelon/PANDA.

## License

MIT, see [LICENSE](LICENSE).
