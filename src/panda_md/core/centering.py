"""Осевое центрирование: ось симметрии формы в плоскости стенок и как найти её положение в кадре.

Центрирование вдоль оси профиля (по поре или по жидкости) — `trajectory/density.py`. Здесь — вторая, независимая
часть: сдвиг в плоскости стенок, чтобы ось формы стояла в начале координат. На одномерный профиль щели он не влияет,
нужен для 2D-карт и облаков точек. Ось объявляется в классе формы (`InterfaceShape.symmetry_axis`).
"""

import numpy as np

from panda_md.core.enums import AxisLocator, AxisOrientation
from panda_md.core.pbc import circular_mean
from panda_md.core.types import FloatArray, PandaModel


class SymmetryAxis(PandaModel):
    orientation: AxisOrientation
    locate: AxisLocator


def axis_position(lateral_positions: FloatArray, lateral_periods: FloatArray, axis: SymmetryAxis) -> FloatArray:
    """Положение оси в плоскости стенок: (по длинной стороне, по короткой стороне), в [0, L) по каждой.

    lateral_positions — координаты частиц (n × 2): вдоль длинной стороны ячейки и вдоль короткой. У горизонтальной
    оси (вдоль короткой стороны) положение по короткой стороне не определено и возвращается как 0.
    """
    along_length = circular_mean(lateral_positions[:, 0], float(lateral_periods[0]))
    along_width = 0.0
    if axis.orientation is AxisOrientation.VERTICAL:
        along_width = circular_mean(lateral_positions[:, 1], float(lateral_periods[1]))
    position = np.array([along_length, along_width])
    if axis.locate is AxisLocator.VOID_CENTER:
        shift = 0.5 * lateral_periods
        if axis.orientation is AxisOrientation.HORIZONTAL:
            shift = np.array([shift[0], 0.0])
        position = np.mod(position + shift, lateral_periods)
    centered: FloatArray = position
    return centered
