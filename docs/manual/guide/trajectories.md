# From a trajectory

Reading trajectories needs `pip install 'panda-md[md]'` (MDAnalysis); any format MDAnalysis reads works.
Selections use the MDAnalysis syntax.

```python
trajectory = panda.load_trajectory("system.tpr", "part1.xtc", "part2.xtc")
pore = panda.SlitPore.from_trajectory(trajectory, walls="resname GRAPHITE", liquid="resname WATER", time="2 ns")
profiles = panda.density_profile(
    trajectory, pore, "resname WATER and name OW",
    start="2 ns", stop=None, block="1 ns", stride=1,
    bins=200,                 # or bin_width_nm=0.05
    centering="pore",         # or "liquid"
    per="atom",               # or "molecule": centres of mass of molecules
)
profiles.save("water.npz")
```

## Pore dimensions

`SlitPore.from_trajectory` measures H as the empty gap between the wall atoms (across periodic boundaries; with
`liquid=` the gap that holds the liquid, useful when the box has a vacuum slab), L and k from the box edges.
`wall_offset_nm` moves each wall surface away from the centres of its outermost atoms (H = gap − 2·offset).

## Centering

By default (`centering=None`) the profile is centred on the pore if its walls (`wall_selection`) or its
centre (`pore_center_nm`) are known, otherwise on the liquid, with a warning.

- `pore`: 0 is the middle between the walls, found from `pore.wall_selection` in every frame (or a
  fixed `pore_center_nm`); bins are aligned with the walls. The profile is flipped z → −z if the liquid sits at
  the upper wall (`orientation="keep"` disables it).
- `liquid`: 0 is the centre of the selected atoms (circular mean across the box); bins cover the whole box.

Shapes with a symmetry axis (droplet, worm, roll, doughnut, perforation) can also be centred along the walls; this matters
for 2D maps, not for the 1D profile:

```python
density = panda.density_map(trajectory, pore, "resname WATER and name OW", shape="roll", start="2 ns")
```

## Time blocks

`block="1 ns"` splits the frames into blocks; an incomplete last block is dropped. `profiles.mean()` is the
profile to fit, `profiles.block(i)` one block, `profiles.sem()` the error of the mean per bin. Fitting every
block (`fit_blocks`) gives the spread of θ; it is an honest error only if blocks are longer than the correlation
time (`statistical_inefficiency`).

## Warnings

The data path warns (does not stop) when something looks wrong: particles outside the pore, wall gap different
from H, box base different from the pore base, a box that changes along the profile axis (NPT), empty blocks.
