# Pores and shapes

A **pore** is a concrete pore with dimensions; a **shape** is one interface shape of that pore type; their pair
is an `InterfaceModel` (`pore.interface("roll")`). `panda shapes` prints the up-to-date list with parameters,
bounds, regimes and symmetry axes.

## Pore types

| Pore | Class | Fields | Profile axis | Characteristic size |
|------|-------|--------|--------------|---------------------|
| slit | `SlitPore` | `width_nm` (H), `length_nm` (L), `base_aspect_ratio` (k: base L × kL), `wall_offset_nm` | normal to the walls | H |
| triangle | `TrianglePore` | `side_nm` (a), `length_nm` (L), `layer_width_nm` | across the channel | a |
| square | `SquarePore` | `side_nm` (a), `length_nm` (L), `layer_width_nm` | across the channel | a |
| cylinder | `CylinderPore` | `diameter_nm` (D), `length_nm` (L: box edge along the tube), `layer_width_nm`, `axis`, `wall_offset_nm` | across the tube | D |

All pores also have `normal_axis` (box axis of the profile, default `z`) and `wall_selection` (atoms of the
walls, used to find the pore centre in every frame). The coordinate of the profile is measured in units of the
characteristic size, δ too.

## Parameters

| Name | Symbol | Unit | Meaning |
|------|--------|------|---------|
| `theta` | θ | degree | contact angle of the analysed liquid |
| `delta` | δ | pore size | thickness of the layer of the other liquid at the walls |
| `phi` | φ | — | volume fraction of the analysed liquid; by default taken from the data (∫ρ / ρ_bulk) |
| `shift` | — | pore size | shift of the profile along its axis (error of the wall positions); fixed at 0 by default |

## Slit shapes

| Shape | Picture | θ regime | Symmetry axis |
|-------|---------|----------|---------------|
| `droplet` | spherical segment on one wall | 90°–180° | vertical, liquid centre |
| `roll` | cylindrical segment bridging the walls | 90°–180° | horizontal, liquid centre |
| `worm` | cylindrical segment on one wall | 90°–180° | horizontal, liquid centre |
| `doughnut` | liquid ring around a column of the other liquid | 90°–180° | vertical, liquid centre |
| `perforation` | film between the walls with a hole | 90°–180° | vertical, centre of the hole |
| `layer` | flat layer in the middle, no contact angle | — | — |

Formulas follow Semenchuk et al. (2025) for the square base and Shistko et al., Mol. Simul. (2025),
[doi:10.1080/08927022.2025.2574643](https://doi.org/10.1080/08927022.2025.2574643) for the rectangular base L × kL.

!!! warning "Wetting regime"
    Formulas for θ < 90° are not derived for the slit shapes. A liquid that wets the walls is fitted at the
    regime boundary and the result carries the flag `at_regime_boundary`.

## Triangular and square pores

Shape names encode how the liquid sits in the pore: `w` — number of walls it touches, `c` — number of corners it
occupies, e.g. `droplet_w2c1` is a droplet touching two walls and occupying one corner. Formulas come from PANDA_Polygon
(G. Vandyshev); some shapes have no density profile yet and take part only in stability diagrams.

## Cylindrical pore

A tube of diameter D whose axis (`axis`, default `x`) runs through the periodic box, as a nanotube does. One shape,
`capsule`: a liquid plug filling the cross-section, closed by two spherical menisci; formulas for θ > 90° only
(convex menisci). Conventions: the profile runs across the tube (z = 0 on the axis, walls at ±D/2) and is normalized
to the box layer L × L_y, where L is the box edge along the tube and L_y (`layer_width_nm`) the box edge along the
third axis; φ is relative to the whole tube π D² L / 4. At the largest φ the menisci touch the periodic image of the
plug. `CylinderPore.from_trajectory` measures the tube from the wall atoms (see “From a trajectory”); the walls must
be shells with empty space outside the tube, and the tube must run through the box.

The profile across a short plug constrains θ weakly: θ enters only through the two caps, while the cylindrical part
dominates the profile. On a synthetic plug in an 8 nm tube (l = 2) a per-bin noise of 0.02 ρ_bulk already spreads the
fitted θ by about ±10°, and layering of the liquid at the wall shifts it as well. Judge θ by its spread over time
blocks (`fit_blocks`) rather than by a single fit.

## Feasibility

Every shape exists only in part of the parameter space (e.g. a droplet cannot hold φ = 0.4 of a short pore).
`model.is_feasible(theta=…, delta=…, phi=…)` and `model.phi_limits(theta=…, delta=…)` answer it; the fit only
visits feasible points and reports `feasible = False` if it ends outside.
