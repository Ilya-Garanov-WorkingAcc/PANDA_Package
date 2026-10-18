# Quick start

## A synthetic profile

```python
import panda_md as panda
from panda_md.synthetic.profiles import synthetic_profile

pore = panda.SlitPore(width_nm=8.0, length_nm=16.0)           # H = 8 nm between the walls, base 16 × 16 nm
roll = pore.interface("roll")                                  # the pore + one of its shapes
profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=30.0)
result = roll.fit(profile, density_bulk_per_nm3=30.0)          # θ and δ free, φ from the integral of ρ
print(result)                                                  # values, loss, problem flags
print(result.values["theta"])
```

## An MD trajectory

```python
import panda_md as panda

trajectory = panda.load_trajectory("system.tpr", "system.xtc")
pore = panda.SlitPore.from_trajectory(trajectory, walls="resname GRAPHITE", liquid="resname WATER")
profiles = panda.density_profile(trajectory, pore, "resname WATER and name OW", start="2 ns", block="1 ns")
spec = panda.FitSpec(density_bulk_per_nm3=33.1)

print(panda.compare_shapes(pore, profiles.mean(), spec))       # every shape of the pore, best loss first
print(panda.fit_blocks(pore.interface("roll"), profiles, spec))   # θ in every block, mean ± error
```

## The same from the shell

```bash
panda profile system.tpr system.xtc --walls "resname GRAPHITE" --select "resname WATER and name OW" \
      --start "2 ns" --block "1 ns" -o water.npz --pore-output pore.json
panda fit water.npz --pore-file pore.json --shape roll --rho-bulk 33.1 --blocks -o result.json --plot fit.png
```

The whole analysis of a real trajectory step by step, with figures, is `examples/demo.ipynb`. More in `examples/` of the repository: a full pipeline script, saved profiles of water in graphite, a stability
diagram and a user-defined shape.
