"""Квадратная пора: канал квадратного сечения со стороной a и длиной L вдоль оси.

Соглашения — как в PANDA_Polygon (`docs/model/square.py`, коммит 38ff491): a = 1, z = 0 в центре сечения,
профиль поперёк одной стенки, нормировка на слой шириной ly (в долях a), φ — относительно исходного сечения.
"""

from typing import ClassVar

from pydantic import Field, model_validator

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry

SQUARE_HALF_SIDE = 0.5  # стенки квадрата в безразмерной координате: z = ±0.5
SQUARE_SECTION_AREA = 1.0  # площадь исходного сечения при a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2δ — сторона квадрата, доступная жидкости."""
    return 1.0 - 2.0 * wall_layer


class SquareGeometry(PoreGeometry):
    pore_length_ratio: float = Field(gt=0.0)  # l = L / a
    layer_width_ratio: float = Field(gt=0.0)  # ly — ширина слоя нормировки профиля в долях a

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SQUARE_HALF_SIDE, SQUARE_HALF_SIDE)

    def profile_integral(self) -> float:
        return SQUARE_SECTION_AREA / self.layer_width_ratio


class SquarePore(Pore):
    name: ClassVar[str] = "square"

    side_nm: float = Field(gt=0.0)  # a — сторона квадратного сечения
    length_nm: float = Field(gt=0.0)  # L — длина канала (период вдоль оси)
    layer_width_nm: float = Field(gt=0.0)  # ширина слоя, на которую нормирован профиль (ширина ящика)

    @model_validator(mode="after")
    def _check_layer_covers_pore(self) -> "SquarePore":
        if self.layer_width_nm < self.side_nm:
            raise ValueError(f"ширина слоя {self.layer_width_nm} нм меньше стороны поры {self.side_nm} нм")
        return self

    def geometry(self) -> SquareGeometry:
        return SquareGeometry(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )

    def characteristic_size_nm(self) -> float:
        return self.side_nm

    def volume_nm3(self) -> float:
        return self.side_nm**2 * self.length_nm
