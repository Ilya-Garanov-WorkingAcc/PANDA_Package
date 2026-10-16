"""Периодические граничные условия вдоль одной оси ящика: заворот, круговое среднее, наибольший пустой промежуток."""

import math

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.types import FloatArray


def wrap_centered(values: FloatArray, period: float) -> FloatArray:
    """Завернуть координаты в [−L/2, L/2) — ближайший образ относительно нуля."""
    wrapped: FloatArray = values - period * np.floor(values / period + 0.5)
    return wrapped


def circular_mean(values: FloatArray, period: float) -> float:
    """Центр облака точек на окружности длины L: среднее направление, результат в [0, L).

    Так центр жидкости, пересекающей границу ящика, не попадает в середину пустоты (как в старой PANDA,
    `utils.get_center_pbc`).
    """
    if values.size == 0:
        raise InvalidTrajectoryError("центр по пустой выборке не определён")
    angles = 2.0 * math.pi * values / period
    mean_angle = math.atan2(-float(np.mean(np.sin(angles))), -float(np.mean(np.cos(angles)))) + math.pi
    return period * mean_angle / (2.0 * math.pi)


def largest_gap(values: FloatArray, period: float) -> tuple[float, float]:
    """Наибольший промежуток без точек на окружности длины L: (начало, конец), конец > начала, конец < начало + L.

    Стенки щели — атомы подложки: пора — самый широкий пустой промежуток между ними, в том числе через границу
    ящика (одна подложка и её образ).
    """
    return empty_gaps(values, period)[0]


def empty_gaps(values: FloatArray, period: float) -> list[tuple[float, float]]:
    """Все промежутки между соседними точками на окружности длины L, от широкого к узкому: (начало, конец)."""
    if values.size == 0:
        raise InvalidTrajectoryError("в выборке стенок нет атомов")
    ordered = np.sort(np.mod(values, period))
    widths = np.diff(np.append(ordered, ordered[0] + period))
    order = np.argsort(-widths, kind="stable")
    return [(float(ordered[index]), float(ordered[index] + widths[index])) for index in order]
