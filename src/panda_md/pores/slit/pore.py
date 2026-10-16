"""Щелевая пора: две плоские стенки на расстоянии H, прямоугольное основание L × kL (0 < k ≤ 1).

Параметр k (squeeze parameter) — Shistko et al., Mol. Simul. (2025), DOI 10.1080/08927022.2025.2574643.
"""

import math
import warnings
from typing import ClassVar

import numpy as np
from pydantic import Field

from panda_md.core.enums import AXIS_ORDER, Axis
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.pbc import empty_gaps, wrap_centered
from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.core.types import FloatArray

SECOND_GAP_WARNING_NM = 1.0  # внутри подложек промежутки < 0.5 нм (графит, SOLD, кальцит); второй шире 1 нм — вакуум?
SLIT_HALF_WIDTH = 0.5  # стенки щели в безразмерной координате: z = ±0.5


class SlitGeometry(PoreGeometry):
    """Безразмерная геометрия щели: l = L / H — длинная сторона основания, k — короткая сторона / длинная."""

    pore_length_ratio: float = Field(gt=0.0)  # l = L / H
    base_aspect_ratio: float = Field(default=1.0, gt=0.0, le=1.0)  # k: основание L × kL

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SLIT_HALF_WIDTH, SLIT_HALF_WIDTH)

    def profile_integral(self) -> float:
        return 1.0  # ∫ρ dz = φ: профиль нормирован на всю площадь основания

    def base_area(self) -> float:
        """k l² — площадь основания ячейки в единицах H²."""
        return self.base_aspect_ratio * self.pore_length_ratio**2

    def short_side_ratio(self) -> float:
        """k l — короткая сторона основания в единицах H."""
        return self.base_aspect_ratio * self.pore_length_ratio

    def equivalent_square(self) -> "SlitGeometry":
        """Квадратная ячейка со стороной kL.

        Осесимметричная форма, не касающаяся боковых границ ячейки, «видит» только её короткую сторону: в ячейке
        L × kL она та же, что в квадратной ячейке kL × kL с объёмной долей, пересчитанной на площадь основания.
        """
        return SlitGeometry(pore_length_ratio=self.short_side_ratio())


class SlitPore(Pore):
    name: ClassVar[str] = "slit"

    width_nm: float = Field(gt=0.0)  # H — расстояние между стенками
    length_nm: float = Field(gt=0.0)  # L — длинная сторона основания ячейки
    base_aspect_ratio: float = Field(default=1.0, gt=0.0, le=1.0)  # k — короткая сторона / длинная
    length_axis: Axis = Axis.X  # ось ящика вдоль длинной стороны основания (для 2D-карт и осевого центрирования)
    wall_offset_nm: float = Field(default=0.0, ge=0.0)  # поверхность стенки — дальше центров крайних атомов

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=self.length_nm / self.width_nm, base_aspect_ratio=self.base_aspect_ratio)

    def characteristic_size_nm(self) -> float:
        return self.width_nm

    def volume_nm3(self) -> float:
        return self.base_aspect_ratio * self.length_nm**2 * self.width_nm

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Площадь основания ящика — произведение двух рёбер, перпендикулярных оси профиля."""
        return math.prod(float(box_nm[axis.column]) for axis in AXIS_ORDER if axis is not self.normal_axis)

    def base_area_nm2(self) -> float:
        """k L² — основание ячейки; должно совпадать с основанием ящика."""
        return self.base_aspect_ratio * self.length_nm**2

    def expected_wall_gap_nm(self) -> float:
        """H + 2 · wall_offset: поверхности стенок сдвинуты внутрь от центров крайних атомов."""
        return self.width_nm + 2.0 * self.wall_offset_nm

    def locate_center_nm(self, wall_coordinates_nm: FloatArray, period_nm: float) -> float:
        """Середина пустого промежутка между атомами стенок, ширина которого ближе всего к H, в [0, L).

        Промежуток может проходить через границу ящика; вакуумный слой другой ширины не выбирается.
        """
        start, end = min(
            empty_gaps(wall_coordinates_nm, period_nm),
            key=lambda gap: abs(gap[1] - gap[0] - self.expected_wall_gap_nm()),
        )
        return (0.5 * (start + end)) % period_nm

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
        """Щель по кадру траектории: H — промежуток между атомами стенок (поверхности — по центрам крайних атомов),
        L — ребро ящика вдоль length_axis, k — второе ребро основания / L.

        Промежуток — наибольший, а если задана выборка жидкости `liquid` — тот, в котором её больше всего (нужно,
        когда в ящике есть вакуумный слой). wall_offset_nm — на сколько поверхность каждой стенки отстоит от центров
        её крайних атомов (например, ван-дер-ваальсов радиус): H = промежуток − 2 · wall_offset_nm.
        """
        if length_axis is normal_axis:
            raise InvalidTrajectoryError(f"ось длины {length_axis.value} совпадает с осью профиля")
        groups = [trajectory.select(walls)] + ([] if liquid is None else [trajectory.select(liquid)])
        frame = trajectory.frame_at(time, groups)
        box = frame.box_nm
        period = float(box[normal_axis.column])
        gaps = empty_gaps(frame.positions_nm[0][:, normal_axis.column], period)
        if liquid is None:
            start, end = gaps[0]
            if len(gaps) > 1 and gaps[1][1] - gaps[1][0] >= SECOND_GAP_WARNING_NM:
                warnings.warn(
                    f"между стенками два широких промежутка ({end - start:.3f} и {gaps[1][1] - gaps[1][0]:.3f} нм) — "
                    f"возможно, в ящике вакуум; выбран наибольший, задай liquid=<выборка>, чтобы выбрать промежуток "
                    f"с жидкостью",
                    PandaWarning,
                    stacklevel=2,
                )
        else:
            heights = frame.positions_nm[1][:, normal_axis.column]

            def liquid_inside(gap: tuple[float, float]) -> int:
                offsets = wrap_centered(heights - 0.5 * (gap[0] + gap[1]), period)
                return int(np.count_nonzero(np.abs(offsets) < 0.5 * (gap[1] - gap[0])))

            start, end = max(gaps, key=liquid_inside)
        (other_axis,) = (axis for axis in AXIS_ORDER if axis not in (length_axis, normal_axis))
        length = float(box[length_axis.column])
        aspect_ratio = float(box[other_axis.column]) / length
        if aspect_ratio > 1.0:
            raise InvalidTrajectoryError(
                f"ребро {other_axis.value} ({box[other_axis.column]:.3f} нм) длиннее {length_axis.value} "
                f"({length:.3f} нм): L — длинная сторона основания, задай length_axis={other_axis.value!r}"
            )
        width = end - start - 2.0 * wall_offset_nm
        if width <= 0.0:
            raise InvalidTrajectoryError(
                f"wall_offset_nm = {wall_offset_nm} нм не меньше половины промежутка {end - start:.3f} нм"
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
