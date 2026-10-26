"""Slit pore: two flat walls a distance H apart, rectangular base L × kL (0 < k ≤ 1).

The parameter k (squeeze parameter) is from Shistko et al., Mol. Simul. (2025), DOI 10.1080/08927022.2025.2574643.
"""

import math
import warnings
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from pydantic import Field

from panda_md.core.enums import AXIS_ORDER, Axis
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.pbc import empty_gaps, wrap_centered
from panda_md.core.pore import Pore, field_as_float
from panda_md.core.shape import PoreGeometry
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.core.types import FloatArray

SECOND_GAP_WARNING_NM = 1.0  # gaps inside substrates < 0.5 nm (graphite, SOLD, calcite); a second one > 1 nm — vacuum?
MIN_SLIT_GAP_NM = 0.5  # a narrower widest gap is the spacing inside a solid (a tube, a channel), not a slit
MIN_LIQUID_INSIDE_FRACTION = 0.5  # less of the liquid in the chosen gap — the walls do not form a slit
ASPECT_RATIO_TOLERANCE = 1e-6  # base edges equal up to float noise — a square
SLIT_HALF_WIDTH = 0.5  # slit walls in the dimensionless coordinate: z = ±0.5


class SlitGeometry(PoreGeometry):
    """Dimensionless slit geometry: l = L / H — long side of the base, k — short side / long side."""

    pore_length_ratio: float = Field(gt=0.0)  # l = L / H
    base_aspect_ratio: float = Field(default=1.0, gt=0.0, le=1.0)  # k: base L × kL

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SLIT_HALF_WIDTH, SLIT_HALF_WIDTH)

    def profile_integral(self) -> float:
        return 1.0  # ∫ρ dz = φ: the profile is normalized to the whole base area

    def base_area(self) -> float:
        """k l² — cell base area in units of H²."""
        return self.base_aspect_ratio * self.pore_length_ratio**2

    def short_side_ratio(self) -> float:
        """k l — short side of the base in units of H."""
        return self.base_aspect_ratio * self.pore_length_ratio

    def equivalent_square(self) -> "SlitGeometry":
        """Return the square cell with side kL.

        An axisymmetric shape that does not touch the lateral cell boundaries "sees" only the short side: in an
        L × kL cell it is the same as in a square kL × kL cell with the volume fraction rescaled to the base area.
        """
        return SlitGeometry(pore_length_ratio=self.short_side_ratio())


class SlitPore(Pore):
    name: ClassVar[str] = "slit"
    trajectory_fields: ClassVar[tuple[str, ...]] = ("wall_selection", "length_axis", "normal_axis", "wall_offset_nm")

    width_nm: float = Field(gt=0.0)  # H — distance between the walls
    length_nm: float = Field(gt=0.0)  # L — long side of the cell base
    base_aspect_ratio: float = Field(default=1.0, gt=0.0, le=1.0)  # k — short side / long side
    length_axis: Axis = Axis.X  # box axis along the long side of the base (for 2D maps and axial centering)
    wall_offset_nm: float = Field(default=0.0, ge=0.0)  # the wall surface lies beyond the outermost atom centres

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=self.length_nm / self.width_nm, base_aspect_ratio=self.base_aspect_ratio)

    def characteristic_size_nm(self) -> float:
        return self.width_nm

    def volume_nm3(self) -> float:
        return self.base_aspect_ratio * self.length_nm**2 * self.width_nm

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Box base area — the product of the two edges perpendicular to the profile axis."""
        return math.prod(float(box_nm[axis.column]) for axis in AXIS_ORDER if axis is not self.normal_axis)

    def base_area_nm2(self) -> float:
        """k L² — cell base; must match the box base."""
        return self.base_aspect_ratio * self.length_nm**2

    def expected_wall_gap_nm(self) -> float:
        """H + 2 · wall_offset: the wall surfaces are shifted inwards from the outermost atom centres."""
        return self.width_nm + 2.0 * self.wall_offset_nm

    def locate_center_nm(self, wall_coordinates_nm: FloatArray, period_nm: float) -> float:
        """Return the middle of the empty gap between wall atoms whose width is closest to H, in [0, L).

        The gap may cross the box boundary; a vacuum layer of a different width is not chosen.
        """
        start, end = min(
            empty_gaps(wall_coordinates_nm, period_nm),
            key=lambda gap: abs(gap[1] - gap[0] - self.expected_wall_gap_nm()),
        )
        return (0.5 * (start + end)) % period_nm

    @classmethod
    def measure_from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        walls: str,
        time: TimeValue,
        liquid: str | None,
        fields: Mapping[str, object],
    ) -> "SlitPore":
        return cls.from_trajectory(
            trajectory,
            walls=walls,
            length_axis=Axis(fields.get("length_axis", Axis.X)),
            normal_axis=Axis(fields.get("normal_axis", Axis.Z)),
            wall_offset_nm=field_as_float(fields, "wall_offset_nm", 0.0),
            time=time,
            liquid=liquid,
        )

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        walls: str,
        length_axis: Axis = Axis.X,
        normal_axis: Axis = Axis.Z,
        time: TimeValue = 0.0,
        liquid: str | None = None,
        wall_offset_nm: float = 0.0,
    ) -> "SlitPore":
        """Build the slit from a trajectory frame: H — gap between wall atoms (surfaces at the outermost atom centres),
        L — box edge along length_axis, k — second base edge / L.

        The gap is the largest one or, if a liquid selection `liquid` is given, the one containing most of it (needed
        when the box has a vacuum layer). wall_offset_nm — how far the surface of each wall lies from the centres of
        its outermost atoms (e.g. the van der Waals radius): H = gap − 2 · wall_offset_nm.
        """
        if length_axis is normal_axis:
            raise InvalidTrajectoryError(
                f"length axis {length_axis.value} coincides with the profile axis; choose another length_axis"
            )
        groups = [trajectory.select(walls)] + ([] if liquid is None else [trajectory.select(liquid)])
        frame = trajectory.frame_at(time, groups)
        box = frame.box_nm
        period = float(box[normal_axis.column])
        gaps = empty_gaps(frame.positions_nm[0][:, normal_axis.column], period)
        if liquid is None:
            start, end = gaps[0]
            if len(gaps) > 1 and gaps[1][1] - gaps[1][0] >= SECOND_GAP_WARNING_NM:
                warnings.warn(
                    f"two wide gaps between the walls ({end - start:.3f} and {gaps[1][1] - gaps[1][0]:.3f} nm), "
                    f"the box may contain vacuum; the largest one is used; pass liquid=<selection> to choose the gap "
                    f"with the liquid",
                    PandaWarning,
                    stacklevel=2,
                )
        else:
            heights = frame.positions_nm[1][:, normal_axis.column]

            def liquid_inside(gap: tuple[float, float]) -> int:
                offsets = wrap_centered(heights - 0.5 * (gap[0] + gap[1]), period)
                return int(np.count_nonzero(np.abs(offsets) < 0.5 * (gap[1] - gap[0])))

            start, end = max(gaps, key=liquid_inside)
            inside_fraction = liquid_inside((start, end)) / heights.size
            if inside_fraction < MIN_LIQUID_INSIDE_FRACTION:
                raise InvalidTrajectoryError(
                    f"not a slit along {normal_axis.value}: at most {inside_fraction:.0%} of the liquid lies between "
                    f"two wall layers ({end - start:.3f} nm gap); the walls may form a tube or a channel — check the "
                    "wall selection and normal_axis"
                )
        if end - start < MIN_SLIT_GAP_NM:
            raise InvalidTrajectoryError(
                f"not a slit along {normal_axis.value}: the widest gap between wall atoms is {end - start:.3f} nm "
                f"(< {MIN_SLIT_GAP_NM} nm); the walls may form a tube or a channel — check the wall selection and "
                "normal_axis"
            )
        (other_axis,) = (axis for axis in AXIS_ORDER if axis not in (length_axis, normal_axis))
        length = float(box[length_axis.column])
        aspect_ratio = float(box[other_axis.column]) / length
        if aspect_ratio > 1.0 + ASPECT_RATIO_TOLERANCE:
            raise InvalidTrajectoryError(
                f"edge {other_axis.value} ({box[other_axis.column]:.3f} nm) is longer than {length_axis.value} "
                f"({length:.3f} nm), but L is the long side of the base; set length_axis={other_axis.value!r}"
            )
        aspect_ratio = min(aspect_ratio, 1.0)  # a nearly square base (NPT noise) is treated as square
        width = end - start - 2.0 * wall_offset_nm
        if width <= 0.0:
            raise InvalidTrajectoryError(
                f"wall_offset_nm = {wall_offset_nm} nm is not less than half the gap {end - start:.3f} nm; "
                "reduce wall_offset_nm"
            )
        return cls(
            width_nm=width,
            wall_offset_nm=wall_offset_nm,
            length_nm=length,
            base_aspect_ratio=aspect_ratio,
            normal_axis=normal_axis,
            length_axis=length_axis,
            wall_selection=walls,
        )
