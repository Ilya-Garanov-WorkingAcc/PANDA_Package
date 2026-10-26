"""Cylindrical pore: a tube of diameter D whose axis runs through the periodic box.

Conventions as in alpha.py (`rollx`): D = 1, the profile runs across the tube (z = 0 on the axis, walls at ±1/2),
the tube length l = L / D is the box edge along the axis (the tube is periodic), the density is normalized to a box
layer of width ly = L_y / D along the third axis, φ is relative to the whole tube π D² L / 4.
"""

import math
from typing import ClassVar

from pydantic import Field, model_validator

from panda_md.core.enums import AXIS_ORDER, Axis
from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry
from panda_md.core.types import FloatArray

CYLINDER_HALF_DIAMETER = 0.5  # tube walls in the dimensionless coordinate: z = ±0.5
CYLINDER_SECTION_AREA = math.pi / 4.0  # area of the tube cross-section at D = 1


def accessible_radius(wall_layer: float) -> float:
    """R′ = 1/2 − δ — tube radius accessible to the analysed liquid."""
    return CYLINDER_HALF_DIAMETER - wall_layer


class CylinderGeometry(PoreGeometry):
    """Dimensionless tube geometry: length l and normalization layer width ly, both in units of D."""

    pore_length_ratio: float = Field(gt=0.0)  # l = L / D
    layer_width_ratio: float = Field(gt=0.0)  # ly = L_y / D

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-CYLINDER_HALF_DIAMETER, CYLINDER_HALF_DIAMETER)

    def profile_integral(self) -> float:
        return CYLINDER_SECTION_AREA / self.layer_width_ratio

    def layer_area(self) -> float:
        """l · ly — area of the box layer by which the liquid cross-section is divided."""
        return self.pore_length_ratio * self.layer_width_ratio


class CylinderPore(Pore):
    """Tube of diameter D along `axis`; the profile runs along `normal_axis`, the layer spans the third axis."""

    name: ClassVar[str] = "cylinder"

    diameter_nm: float = Field(gt=0.0)  # D — tube diameter (wall surface to wall surface)
    length_nm: float = Field(gt=0.0)  # L — box edge along the tube axis (the tube is periodic)
    layer_width_nm: float = Field(gt=0.0)  # L_y — box edge along the third axis: width of the normalization layer
    axis: Axis = Axis.X  # box axis along the tube
    wall_offset_nm: float = Field(default=0.0, ge=0.0)  # the wall surface lies beyond the innermost atom centres

    @model_validator(mode="after")
    def _check_axes_and_layer(self) -> "CylinderPore":
        if self.axis is self.normal_axis:
            raise ValueError(
                f"tube axis {self.axis.value} coincides with the profile axis; choose another axis or normal_axis"
            )
        if self.layer_width_nm < self.diameter_nm:
            raise ValueError(
                f"layer width {self.layer_width_nm} nm is smaller than the tube diameter {self.diameter_nm} nm; "
                "set layer_width_nm to the box edge across the tube"
            )
        return self

    @property
    def layer_axis(self) -> Axis:
        """Box axis across the tube and across the profile: the normalization layer spans it."""
        (third,) = (axis for axis in AXIS_ORDER if axis not in (self.axis, self.normal_axis))
        return third

    def geometry(self) -> CylinderGeometry:
        return CylinderGeometry(
            pore_length_ratio=self.length_nm / self.diameter_nm,
            layer_width_ratio=self.layer_width_nm / self.diameter_nm,
        )

    def characteristic_size_nm(self) -> float:
        return self.diameter_nm

    def volume_nm3(self) -> float:
        return CYLINDER_SECTION_AREA * self.diameter_nm**2 * self.length_nm

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Box layer L × L_y the profile is normalized to (does not depend on the box)."""
        return self.length_nm * self.layer_width_nm
