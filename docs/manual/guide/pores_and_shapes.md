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
occupies, e.g. `droplet_w2c1` is a droplet touching two walls and occupying one corner. A filament is uniform along
the channel, a droplet sits on the walls, a capsule spans the whole cross-section and ends in two menisci. Formulas
come from PANDA_Polygon and its revised model (G. Vandyshev). Every shape has surface areas for the stability
diagrams; a shape without a density profile takes part only in them.

Square pore — 16 shapes:

| Shape | Picture | θ regime | Profile |
|-------|---------|----------|---------|
| `filament_w0c0` | circular cylinder along the axis, no walls touched; no θ | — | yes |
| `filament_w0c0_square` | cord of square cross-section along the axis, no walls touched; no θ | — | yes |
| `filament_w1c0` | cylindrical segment on one wall | 0°–90°, 90°–180° | yes |
| `filament_w2c1` | fills the corner between two adjacent walls | 45°–90°, 90°–135° | yes |
| `filament_w2c0_opp` | bridge between two opposite walls, two menisci | 90°–180° | yes |
| `filament_w3c2` | touches three walls, fills two corners, one meniscus | 90°–180° | yes |
| `filament_w4c0` … `filament_w4c3` | fills the section, touches all four walls; 0–3 corners filled, the rest empty | 135°–180° | yes |
| `droplet_w1c0` | spherical cap on one wall | 90°–180° | yes |
| `droplet_w2c0` | droplet touching two adjacent walls, corner empty | 135°–180° | yes |
| `droplet_w2c1` | droplet filling the corner between two adjacent walls | 90°–135° | yes |
| `capsule_w4c0` | bridge touching four walls, corners empty, menisci at the ends | 135°–180° | yes |
| `capsule_w4c4` | plug filling the whole square, two spherical menisci | 90°–135° | yes |
| `perforation` | liquid filling the channel except an axisymmetric hole through two opposite walls | 90°–180° | yes |

Triangular pore — 14 shapes:

| Shape | Picture | θ regime | Profile |
|-------|---------|----------|---------|
| `filament_w0c0` | circular cylinder along the axis, no walls touched; no θ | — | yes |
| `filament_w0c0_triangle` | cord of triangular cross-section along the axis, no walls touched; no θ | — | yes |
| `filament_w1c0` | cylindrical segment on the base | 0°–90°, 90°–180° | yes |
| `filament_w2c0` | in the corner between two walls, corner empty | 120°–180° | yes |
| `filament_w2c1` | fills the corner between two walls | 90°–120° | yes |
| `filament_w3c0` | middle of the section, all three corners empty | 120°–180° | yes |
| `filament_w3c1` | touches three walls, one corner filled, two empty | 120°–180° | yes |
| `filament_w3c2` | touches three walls, two corners filled, one empty | 120°–180° | yes |
| `droplet_w1c0` | spherical cap on the base | 90°–180° | yes |
| `droplet_w2c0` | droplet touching two walls, corner empty | 120°–180° | yes |
| `droplet_w2c1` | droplet wetting the edge between two walls | 90°–120° | yes |
| `droplet_w3c0` | droplet touching three walls; its size is fixed by the section | 120°–180° | no |
| `capsule_w3c0` | bridge touching three walls, corners empty, menisci at the ends | 120°–180° | yes |
| `capsule_w3c3` | plug filling the whole triangle, two spherical menisci | 90°–120° | yes |

The hole of the square `perforation`, like that of the slit perforation, has a circular generating arc: an
approximation, not a surface of constant mean curvature.

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
