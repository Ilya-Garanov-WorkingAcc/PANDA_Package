"""Cylindrical pore: a tube of diameter D whose axis runs through the periodic box.

Conventions as in alpha.py (`rollx`): D = 1, the profile runs across the tube (z = 0 on the axis, walls at ±1/2),
the tube length l = L / D is the box edge along the axis (the tube is periodic), the density is normalized to a box
layer of width ly = L_y / D along the third axis, φ is relative to the whole tube π D² L / 4.
"""

import math
from typing import ClassVar

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import AXIS_ORDER, Axis
from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import empty_gaps, wrap_centered
from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.core.types import FloatArray

CYLINDER_HALF_DIAMETER = 0.5  # tube walls in the dimensionless coordinate: z = ±0.5
CYLINDER_SECTION_AREA = math.pi / 4.0  # area of the tube cross-section at D = 1
MIN_TUBE_DIAMETER_NM = 0.5  # a narrower empty channel along the axis is the spacing inside a solid, not a tube
MAX_SHELL_SPREAD_FRACTION = 0.5  # wall atoms farther from the axis than 1.5 inner radii: not concentric shells
MIN_LIQUID_INSIDE_FRACTION = 0.5  # less of the liquid inside the tube — the walls do not form a tube around it


def accessible_radius(wall_layer: float) -> float:
    """R′ = 1/2 − δ — tube radius accessible to the analysed liquid."""
    return CYLINDER_HALF_DIAMETER - wall_layer


def tube_axis_coordinate(wall_coordinates_nm: FloatArray, period_nm: float) -> float:
    """Coordinate of the tube axis along one box axis, in [0, L): opposite the widest empty gap of the wall atoms.
    Projected on a box axis, the shells of a tube fill [c − R_out, c + R_out]; the widest empty gap is the space
    outside the tube, centred at c + L/2. A circular mean fails here: when the tube nearly fills the box, the two
    sides of a shell are close across the boundary and the mean may point to the opposite side.
    """
    start, end = empty_gaps(wall_coordinates_nm, period_nm)[0]
    return (0.5 * (start + end) + 0.5 * period_nm) % period_nm


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
    wall_size_label: ClassVar[str] = "tube diameter"

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

    def expected_wall_gap_nm(self) -> float:
        """D + 2 · wall_offset: the wall surface lies inwards from the innermost atom centres."""
        return self.diameter_nm + 2.0 * self.wall_offset_nm

    def locate_center_nm(self, wall_coordinates_nm: FloatArray, period_nm: float) -> float:
        """Tube axis along the profile axis, from the wall atoms (see tube_axis_coordinate)."""
        return tube_axis_coordinate(wall_coordinates_nm, period_nm)

    def wall_size_check_nm(
        self, wall_positions_nm: FloatArray, box_nm: FloatArray, center_nm: float
    ) -> tuple[float, float]:
        """(2 × distance from the axis to the nearest wall atom, D + 2 · wall_offset)."""
        columns = (self.normal_axis.column, self.layer_axis.column)
        center = np.array(
            [center_nm, tube_axis_coordinate(wall_positions_nm[:, columns[1]], float(box_nm[columns[1]]))]
        )
        inner_radius = float(np.min(_radial_distances(wall_positions_nm, box_nm, columns, center)))
        return 2.0 * inner_radius, self.expected_wall_gap_nm()

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        walls: str,
        axis: Axis | str = Axis.X,
        normal_axis: Axis | str = Axis.Z,
        time: TimeValue = 0.0,
        liquid: str | None = None,
        wall_offset_nm: float = 0.0,
    ) -> "CylinderPore":
        """Build the tube from a trajectory frame: the axis is found from the wall atoms (tube_axis_coordinate),
        D — twice the distance from the axis to the nearest wall atom minus 2 · wall_offset_nm, L — box edge along
        `axis`, L_y — box edge along the third axis.

        With several concentric shells (a multi-walled nanotube) D is measured by the inner one. If `liquid` is
        given, at least half of it must lie inside the tube.
        """
        axis, normal_axis = Axis(axis), Axis(normal_axis)
        if axis is normal_axis:
            raise InvalidTrajectoryError(
                f"tube axis {axis.value} coincides with the profile axis; choose another axis or normal_axis"
            )
        (layer_axis,) = (other for other in AXIS_ORDER if other not in (axis, normal_axis))
        columns = (normal_axis.column, layer_axis.column)
        groups = [trajectory.select(walls)] + ([] if liquid is None else [trajectory.select(liquid)])
        frame = trajectory.frame_at(time, groups)
        box = frame.box_nm
        wall_positions = frame.positions_nm[0]
        center = np.array([tube_axis_coordinate(wall_positions[:, column], float(box[column])) for column in columns])
        radii = _radial_distances(wall_positions, box, columns, center)
        inner, outer = float(np.min(radii)), float(np.max(radii))
        if 2.0 * inner < MIN_TUBE_DIAMETER_NM or outer - inner > MAX_SHELL_SPREAD_FRACTION * inner:
            raise InvalidTrajectoryError(
                f"not a tube along {axis.value}: the wall atoms lie {inner:.3f}–{outer:.3f} nm from the axis, not on "
                "concentric shells around an empty channel; check the wall selection and the axis"
            )
        if liquid is not None:
            inside = np.count_nonzero(_radial_distances(frame.positions_nm[1], box, columns, center) < inner)
            inside_fraction = inside / frame.positions_nm[1].shape[0]
            if inside_fraction < MIN_LIQUID_INSIDE_FRACTION:
                raise InvalidTrajectoryError(
                    f"not a tube along {axis.value} around the liquid: only {inside_fraction:.0%} of it lies inside "
                    f"the wall shells (radius {inner:.3f} nm); check the wall selection and the axis"
                )
        diameter = 2.0 * inner - 2.0 * wall_offset_nm
        if diameter <= 0.0:
            raise InvalidTrajectoryError(
                f"wall_offset_nm = {wall_offset_nm} nm is not less than the inner tube radius {inner:.3f} nm; "
                "reduce wall_offset_nm"
            )
        return cls(
            diameter_nm=diameter,
            length_nm=float(box[axis.column]),
            layer_width_nm=float(box[layer_axis.column]),
            axis=axis,
            normal_axis=normal_axis,
            wall_offset_nm=wall_offset_nm,
            wall_selection=walls,
        )


def _radial_distances(
    positions_nm: FloatArray, box_nm: FloatArray, columns: tuple[int, int], center_nm: FloatArray
) -> FloatArray:
    """Distance of each point from the tube axis in the cross-section spanned by `columns`, with PBC."""
    selected = list(columns)
    offsets = wrap_centered(positions_nm[:, selected] - center_nm, box_nm[selected])
    distances: FloatArray = np.hypot(offsets[:, 0], offsets[:, 1])
    return distances
