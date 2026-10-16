"""Add your own interface shape without touching the library.

A shape is a class with its parameters, regimes and formulas, registered with a pore type by a decorator. Here: a
liquid film lying on the lower wall of a slit pore (thickness φ·H above a wall layer δ). After registration the
shape takes part in fitting, `compare_shapes`, `panda shapes` and synthetic data like the built-in ones.

    python examples/05_custom_shape.py
"""

from typing import ClassVar

import numpy as np

import panda_md as panda
from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SlitGeometry, SlitPore
from panda_md.synthetic.profiles import synthetic_profile

LOWER_WALL = -0.5  # walls of the slit in units of H: z = ±0.5


@SlitPore.register
class BottomFilm(InterfaceShape[SlitGeometry]):
    """Flat film on the lower wall: ρ/ρ_bulk = 1 for −0.5 + δ < z < −0.5 + δ + φ."""

    name: ClassVar[str] = "bottom_film"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}  # no contact angle in this shape
    frame: ClassVar[Frame] = Frame.PORE  # formulas are written with z = 0 in the middle of the pore
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.WALL_LAYER_THICKNESS.value,)

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        return values[ParameterName.WALL_LAYER_THICKNESS] + values[ParameterName.VOLUME_FRACTION] <= 1.0

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        return 0.0, 1.0 - values[ParameterName.WALL_LAYER_THICKNESS]

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        bottom = LOWER_WALL + values[ParameterName.WALL_LAYER_THICKNESS]
        top = bottom + values[ParameterName.VOLUME_FRACTION]
        return np.where((coordinate > bottom) & (coordinate < top), 1.0, 0.0)


def main() -> None:
    pore = panda.SlitPore(width_nm=6.0, length_nm=12.0)
    film = pore.interface("bottom_film")
    profile = synthetic_profile(film, {"delta": 0.05, "phi": 0.3}, density_bulk_per_nm3=30.0, bin_count=300)
    print(film.fit(profile, density_bulk_per_nm3=30.0))
    print(panda.compare_shapes(pore, profile, panda.FitSpec(density_bulk_per_nm3=30.0)))


if __name__ == "__main__":
    main()
