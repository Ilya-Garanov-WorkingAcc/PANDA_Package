"""Общее для пор-каналов с многоугольным сечением (квадрат, треугольник).

Соглашения PANDA_Polygon (`docs/model/*.py`, коммит 38ff491): сторона сечения a = 1, длина канала l = L / a,
профиль — поперёк одной стенки, плотность нормирована на слой ящика шириной ly (в долях a), φ — относительно
исходного сечения (без слоя δ).
"""

from abc import ABC
from typing import ClassVar

from pydantic import Field, model_validator

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry
from panda_md.core.types import FloatArray


class PolygonGeometry(PoreGeometry, ABC):
    """Безразмерная геометрия канала: длина l и ширина слоя нормировки ly (обе — в долях стороны a)."""

    section_area: ClassVar[float]  # площадь исходного сечения при a = 1

    pore_length_ratio: float = Field(gt=0.0)  # l = L / a
    layer_width_ratio: float = Field(gt=0.0)  # ly — ширина слоя нормировки профиля в долях a

    def profile_integral(self) -> float:
        return self.section_area / self.layer_width_ratio

    def layer_area(self) -> float:
        """l · ly — площадь слоя ящика, на которую делится сечение капли или капсулы."""
        return self.pore_length_ratio * self.layer_width_ratio


class PolygonPore(Pore, ABC):
    """Канал многоугольного сечения со стороной a и длиной L; профиль нормирован на слой ящика шириной ly."""

    geometry_class: ClassVar[type[PolygonGeometry]]

    side_nm: float = Field(gt=0.0)  # a — сторона сечения
    length_nm: float = Field(gt=0.0)  # L — длина канала (период вдоль оси)
    layer_width_nm: float = Field(gt=0.0)  # ширина слоя, на которую нормирован профиль (ширина ящика)

    @model_validator(mode="after")
    def _check_layer_covers_pore(self) -> "PolygonPore":
        if self.layer_width_nm < self.side_nm:
            raise ValueError(f"ширина слоя {self.layer_width_nm} нм меньше стороны поры {self.side_nm} нм")
        return self

    def geometry(self) -> PolygonGeometry:
        return self.geometry_class(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Слой ящика L × ly, на который нормирован профиль (от ящика не зависит)."""
        return self.length_nm * self.layer_width_nm

    def characteristic_size_nm(self) -> float:
        return self.side_nm

    def volume_nm3(self) -> float:
        return self.geometry_class.section_area * self.side_nm**2 * self.length_nm
