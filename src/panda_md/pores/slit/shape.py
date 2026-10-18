"""Common base of the slit shapes: pore frame of reference, accessible width 1 − 2δ, existence by a φ interval,
three-dimensional shape from the profile.

The slit profile is the fraction of the cell cross-section area occupied by the liquid at height z. Hence the
cross-section is recovered from the profile: a disk (droplet, doughnut) of area ρ · S, a circular hole (perforation)
of area (1 − ρ) · S, a band along the short side (roll, worm) of width ρ · l, the whole cross-section (layer);
S = k l² is the cell base.
"""

import math
from enum import Enum
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import BoolArray, FloatArray
from panda_md.pores.slit.pore import SlitGeometry

FULL_DENSITY_THRESHOLD = 0.5  # the layer profile is a 0/1 step


class CrossSection(str, Enum):
    """Cross-section of a slit shape by the plane z = const."""

    DISK = "disk"  # disk centred on the vertical axis
    HOLE = "hole"  # the whole accessible cross-section except a disk on the vertical axis
    BAND = "band"  # band along the short side, symmetric about x = 0
    FULL = "full"  # the whole cross-section


class SlitShape(InterfaceShape[SlitGeometry]):
    """Slit shape. Exists if the accessible width is positive and φ lies strictly inside (φ_min, φ_max)."""

    frame: ClassVar[Frame] = Frame.PORE
    cross_section: ClassVar[CrossSection]

    @staticmethod
    def accessible_width(values: ParameterValues) -> float:
        """1 − 2δ — slit width accessible to the analysed liquid."""
        return 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]

    @staticmethod
    def inside_accessible(coordinate: FloatArray, values: ParameterValues) -> BoolArray:
        """|z| < 1/2 − δ — points accessible to the analysed liquid."""
        half = 0.5 * SlitShape.accessible_width(values)
        return np.abs(coordinate) < half

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if self.accessible_width(values) <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return bool(lower < values[ParameterName.VOLUME_FRACTION] < upper)

    def contains(self, points: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> BoolArray:
        """A point is in the liquid if it is inside the cross-section at its height (see the module docstring)."""
        length = geometry.pore_length_ratio
        width = geometry.short_side_ratio()
        along_length = points[:, 0] - length * np.round(points[:, 0] / length)
        along_width = points[:, 1] - width * np.round(points[:, 1] / width)
        density = self.profile(points[:, 2], geometry, values)
        base = geometry.base_area()
        if self.cross_section is CrossSection.BAND:
            inside: BoolArray = np.abs(along_length) < 0.5 * density * length
            return inside
        radius_squared = along_length**2 + along_width**2
        if self.cross_section is CrossSection.DISK:
            disk: BoolArray = radius_squared < density * base / math.pi
            return disk
        if self.cross_section is CrossSection.HOLE:
            hole_radius_squared = (1.0 - density) * base / math.pi
            liquid: BoolArray = self.inside_accessible(points[:, 2], values) & (radius_squared >= hole_radius_squared)
            return liquid
        full: BoolArray = density >= FULL_DENSITY_THRESHOLD
        return full
