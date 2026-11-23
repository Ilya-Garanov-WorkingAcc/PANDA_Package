# Installation

PANDA needs Python ≥ 3.10.

```bash
pip install panda-md              # core: pores, shapes, fitting, analysis, command line
pip install 'panda-md[md]'        # + reading MD trajectories (MDAnalysis)
pip install 'panda-md[plot]'      # + figures (matplotlib)
pip install 'panda-md[md,plot]'   # everything
```

The package is imported as `panda_md`; the command line is `panda` (also `python -m panda_md`).

```python
import panda_md
print(panda_md.__version__)
```

Without the `md` extra, reading a trajectory raises a clear error asking to install it; the same holds for
figures without `plot`. Importing `panda_md` never loads MDAnalysis or matplotlib.

## From the repository

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Ilya-Garanov-WorkingAcc/PANDA_Package
cd PANDA_Package
make install    # .venv with the package, all extras and the development tools
make check      # black + ruff + mypy + pytest
make docs       # this documentation into site/
```
