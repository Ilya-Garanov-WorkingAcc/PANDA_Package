"""Common code for channel pores with a polygonal cross-section (square, triangle).

PANDA_Polygon conventions (`docs/model/*.py`, commit 38ff491): cross-section side a = 1, channel length l = L / a,
the profile runs across one wall, the density is normalized to a box layer of width ly (in units of a), φ is
relative to the original cross-section (without the δ layer).
"""

from abc import ABC
from typing import ClassVar

from pydantic import Field, model_validator

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry
from panda_md.core.types import FloatArray


class PolygonGeometry(PoreGeometry, ABC):
    """Dimensionless channel geometry: length l and normalization layer width ly (both in units of the side a)."""

    section_area: ClassVar[float]  # area of the original cross-section at a = 1

    pore_length_ratio: float = Field(gt=0.0)  # l = L / a
    layer_width_ratio: float = Field(gt=0.0)  # ly — width of the profile normalization layer in units of a

    def profile_integral(self) -> float:
        return self.section_area / self.layer_width_ratio

    def layer_area(self) -> float:
        """l · ly — area of the box layer by which the droplet or capsule cross-section is divided."""
        return self.pore_length_ratio * self.layer_width_ratio


class PolygonPore(Pore, ABC):
    """Channel with a polygonal cross-section of side a and length L; profile normalized to a box layer of width ly."""

    geometry_class: ClassVar[type[PolygonGeometry]]

    side_nm: float = Field(gt=0.0)  # a — cross-section side
    length_nm: float = Field(gt=0.0)  # L — channel length (period along the axis)
    layer_width_nm: float = Field(gt=0.0)  # width of the layer the profile is normalized to (box width)

    @model_validator(mode="after")
    def _check_layer_covers_pore(self) -> "PolygonPore":
        if self.layer_width_nm < self.side_nm:
            raise ValueError(
                f"layer width {self.layer_width_nm} nm is smaller than the pore side {self.side_nm} nm; "
                "set layer_width_nm to at least side_nm"
            )
        return self

    def geometry(self) -> PolygonGeometry:
        return self.geometry_class(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Box layer L × ly the profile is normalized to (does not depend on the box)."""
        return self.length_nm * self.layer_width_nm

    def characteristic_size_nm(self) -> float:
        return self.side_nm

    def volume_nm3(self) -> float:
        return self.geometry_class.section_area * self.side_nm**2 * self.length_nm
