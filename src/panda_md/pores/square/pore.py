"""Square pore: a channel of square cross-section with side a and length L along the axis.

Conventions as in PANDA_Polygon (`docs/model/square.py`, commit 38ff491): a = 1, z = 0 at the cross-section
centre, the profile runs across one wall, normalization to a layer of width ly (in units of a), φ is relative
to the original cross-section.
"""

from typing import ClassVar

from panda_md.pores.polygon import PolygonGeometry, PolygonPore

SQUARE_HALF_SIDE = 0.5  # square walls in the dimensionless coordinate: z = ±0.5
SQUARE_SECTION_AREA = 1.0  # area of the original cross-section at a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2δ — side of the square accessible to the liquid."""
    return 1.0 - 2.0 * wall_layer


class SquareGeometry(PolygonGeometry):
    section_area: ClassVar[float] = SQUARE_SECTION_AREA

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SQUARE_HALF_SIDE, SQUARE_HALF_SIDE)


class SquarePore(PolygonPore):
    name: ClassVar[str] = "square"
    geometry_class: ClassVar[type[PolygonGeometry]] = SquareGeometry

    def geometry(self) -> SquareGeometry:
        return SquareGeometry(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )
