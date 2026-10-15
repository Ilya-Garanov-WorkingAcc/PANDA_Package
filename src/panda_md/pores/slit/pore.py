"""Щелевая пора: две плоские стенки на расстоянии H, квадратное основание L × L."""

from typing import ClassVar

from pydantic import Field

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry

SLIT_HALF_WIDTH = 0.5  # стенки щели в безразмерной координате: z = ±0.5


class SlitGeometry(PoreGeometry):
    """Безразмерная геометрия щели: l = L / H (Semenchuk et al., 2025)."""

    pore_length_ratio: float = Field(gt=0.0)  # l = L / H

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SLIT_HALF_WIDTH, SLIT_HALF_WIDTH)

    def profile_integral(self) -> float:
        return 1.0  # ∫ρ dz = φ: профиль нормирован на всю площадь основания


class SlitPore(Pore):
    name: ClassVar[str] = "slit"

    width_nm: float = Field(gt=0.0)  # H — расстояние между стенками
    length_nm: float = Field(gt=0.0)  # L — период ячейки вдоль стенки

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=self.length_nm / self.width_nm)

    def characteristic_size_nm(self) -> float:
        return self.width_nm

    def volume_nm3(self) -> float:
        return self.length_nm**2 * self.width_nm
