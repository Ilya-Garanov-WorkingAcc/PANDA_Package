# PANDA

**P**redicting **A**ngle from **N**anoscale **D**ensity **A**nalysis — a Python library that determines the
contact angle θ and the interface shape of a liquid confined in a pore from the one-dimensional density profile
of a molecular simulation.

Two immiscible liquids in a pore form one of a few interface shapes: a droplet, a roll bridging the walls, a
worm, a doughnut, a perforated film, a flat layer. For every shape the density profile across the pore is known
analytically as a function of θ, of the thickness δ of the layer the other liquid keeps at the walls and of the
volume fraction φ. PANDA fits the analytical profile to the measured one, so the interface never has to be
located explicitly.

```python
import panda_md as panda

trajectory = panda.load_trajectory("system.tpr", "system.xtc")
pore = panda.SlitPore.from_trajectory(trajectory, walls="resname GRAPHITE")
profiles = panda.density_profile(trajectory, pore, "resname WATER and name OW", start="2 ns", block="1 ns")
print(panda.fit_blocks(pore.interface("roll"), profiles, panda.FitSpec(density_bulk_per_nm3=33.1)))
```

What is inside:

- three pore types — slit (rectangular base L × kL, six shapes), triangular (12 shapes) and square (7 shapes);
- one fit for every shape: free, fixed and bounded parameters by name, wetting regimes kept apart, problem
  flags in the result;
- the data path from an MD trajectory (MDAnalysis): pore dimensions from the wall atoms, centering, time
  blocks, per-molecule profiles, 2D density maps with axial centering;
- analysis: θ ± its error over time blocks, comparison of all shapes, stability diagrams;
- the command line `panda` and a settings file for a whole run;
- new shapes and pores added from a user script, without changes to the library.

The method: A. Semenchuk, N. Kondratyuk, I. Kopanichuk, *PANDA: Predicting angle from nanoscale density
analysis*, Colloids Surf. A 708 (2025) 135994,
[doi:10.1016/j.colsurfa.2024.135994](https://doi.org/10.1016/j.colsurfa.2024.135994).

