# Adding a shape or a pore

Shapes and pores are classes registered by a decorator; the core, the fit, `compare_shapes`, the command line
and the synthetic generators find them through the registry, so nothing else has to change.

## A new shape

```python
from typing import ClassVar
import numpy as np
from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore


@SlitPore.register
class BottomFilm(InterfaceShape[SlitGeometry]):
    name: ClassVar[str] = "bottom_film"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}       # no contact angle
    frame: ClassVar[Frame] = Frame.PORE                               # z = 0 in the middle of the pore
    default_optimize: ClassVar[tuple[str, ...]] = ("delta",)

    def feasible(self, geometry, values):
        return values[ParameterName.WALL_LAYER_THICKNESS] + values[ParameterName.VOLUME_FRACTION] <= 1.0

    def profile(self, coordinate, geometry, values):
        bottom = -0.5 + values[ParameterName.WALL_LAYER_THICKNESS]
        return np.where((coordinate > bottom) & (coordinate < bottom + values[ParameterName.VOLUME_FRACTION]), 1.0, 0.0)
```

The full script is `examples/05_custom_shape.py`. What a shape declares:

| Member | Required | Meaning |
|--------|----------|---------|
| `name`, `parameters`, `regimes`, `frame` | yes | name in the registry, its parameters (θ, δ, φ, …), θ ranges with derived formulas, frame of the formulas |
| `feasible(geometry, values)` | yes | does the shape exist at these values (internal units: θ in radians) |
| `profile(coordinate, geometry, values)` | to be fitted | ρ / ρ_bulk on the dimensionless coordinate |
| `phi_limits` | recommended | feasible φ interval — better starting points |
| `surface_terms` | for stability diagrams | S_LV and S_SL |
| `contains(points, …)` | for 3D | which points are liquid (synthetic point clouds, 3D checks) |
| `symmetry_axis` | for axial centering | `SymmetryAxis(orientation, locate)` |
| `center_of_mass` | optional | analytic centre of mass; numeric by default |

Every registered shape is checked by the contract tests (`tests/contract`): ∫ρ dz = φ, 0 ≤ ρ ≤ 1, consistency
of the centre of mass, of `contains` and of the surface areas.

## A new pore type

Subclass `panda_md.core.pore.Pore` with a `name`, its fields (pydantic) and `geometry()`,
`characteristic_size_nm()`, `volume_nm3()`, `bin_cross_section_nm2(box)`; add the class to
`panda_md.pores.registry.PORE_TYPES` to make it available by name in the command line. Its shapes register with
`@NewPore.register`; every pore type has its own registry.
