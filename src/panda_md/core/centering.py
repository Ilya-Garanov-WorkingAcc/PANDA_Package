"""Осевое центрирование: ось симметрии формы в плоскости стенок и как найти её положение в кадре.

Центрирование вдоль оси профиля (по поре или по жидкости) — `trajectory/density.py`. Здесь — вторая, независимая
часть: сдвиг в плоскости стенок, чтобы ось формы стояла в начале координат. На одномерный профиль щели он не влияет,
нужен для 2D-карт и облаков точек. Ось объявляется в классе формы (`InterfaceShape.symmetry_axis`).

Положение ищется по гистограмме частиц в плоскости стенок (у горизонтальной оси — вдоль длинной стороны), периодически
сглаженной гауссом шириной L / 16: берётся самый выраженный экстремум — максимум жидкости или минимум пустоты (тогда
центр жидкости — на L/2 дальше) — и уточняется средним с весами вокруг него. Простое круговое среднее всех частиц
теряет ось, когда жидкость почти заполняет ячейку (маленькое отверстие перфорации, широкий валик): вектор среднего
тогда порядка шума выборки.
"""

import warnings

import numpy as np

from panda_md.core.enums import AxisLocator, AxisOrientation
from panda_md.core.errors import PandaWarning
from panda_md.core.types import FloatArray, PandaModel

LATERAL_BINS = 64  # бинов гистограммы вдоль стороны ячейки
SMOOTHING_FRACTION = 1.0 / 16.0  # σ сглаживания в долях стороны ячейки
REFINEMENT_SIGMAS = 2.0  # уточнение — по окну ± 2σ вокруг экстремума
WEAK_CONTRAST = 0.3  # перепад сглаженной гистограммы меньше 30 % среднего — форма почти однородна


class SymmetryAxis(PandaModel):
    orientation: AxisOrientation
    locate: AxisLocator


def _smooth_periodic(counts: FloatArray, sigma_bins: float) -> FloatArray:
    """Периодическое сглаживание гауссом по каждой оси массива (свёртка через БПФ)."""
    smoothed = counts.astype(np.float64)
    for axis_index, size in enumerate(smoothed.shape):
        offsets = np.fft.fftfreq(size) * size  # 0, 1, …, −1 в бинах
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel /= kernel.sum()
        shape = [1] * smoothed.ndim
        shape[axis_index] = size
        transformed = np.fft.fft(smoothed, axis=axis_index) * np.fft.fft(kernel).reshape(shape)
        smoothed = np.real(np.fft.ifft(transformed, axis=axis_index))
    result: FloatArray = smoothed
    return result


def _refined_center(deviation: FloatArray, index: int, period: float) -> float:
    """Центр вдоль одной оси: среднее с весами (положительное отклонение) в окне ± 2σ вокруг экстремума."""
    size = deviation.size
    half_window = max(1, round(REFINEMENT_SIGMAS * SMOOTHING_FRACTION * size))
    offsets = np.arange(-half_window, half_window + 1)
    weights = np.clip(deviation[(index + offsets) % size], 0.0, None)
    total = float(np.sum(weights))
    shift = float(np.sum(weights * offsets)) / total if total > 0.0 else 0.0
    return ((index + 0.5 + shift) * period / size) % period


def structure_center(values: FloatArray, periods: FloatArray) -> tuple[FloatArray, float]:
    """Центр жидкости (по каждой оси, в [0, L)) и перепад сглаженной гистограммы, делённый на среднее.

    values — координаты частиц (n × d, d = 1 или 2), periods — стороны ячейки (d).
    """
    dimensions = values.shape[1]
    counts, _ = np.histogramdd(
        np.mod(values, periods), bins=LATERAL_BINS, range=[(0.0, float(period)) for period in periods]
    )
    smoothed = _smooth_periodic(counts, SMOOTHING_FRACTION * LATERAL_BINS)
    mean = float(np.mean(smoothed))
    if mean == 0.0:
        return np.zeros(dimensions), 0.0
    excess, deficit = float(np.max(smoothed)) - mean, mean - float(np.min(smoothed))
    liquid_peak = excess >= deficit
    deviation = smoothed - mean if liquid_peak else mean - smoothed
    peak = np.unravel_index(int(np.argmax(deviation)), deviation.shape)
    center = np.empty(dimensions)
    for axis_index in range(dimensions):
        index: list[int | slice] = [int(position) for position in peak]
        index[axis_index] = slice(None)
        center[axis_index] = _refined_center(deviation[tuple(index)], int(peak[axis_index]), float(periods[axis_index]))
    if not liquid_peak:
        center = np.mod(center + 0.5 * periods, periods)  # центр пустоты → центр жидкости
    return center, max(excess, deficit) / mean


def axis_position(lateral_positions: FloatArray, lateral_periods: FloatArray, axis: SymmetryAxis) -> FloatArray:
    """Положение оси в плоскости стенок: (по длинной стороне, по короткой стороне), в [0, L) по каждой.

    lateral_positions — координаты частиц (n × 2): вдоль длинной стороны ячейки и вдоль короткой. У горизонтальной
    оси (вдоль короткой стороны) положение по короткой стороне не определено и возвращается как 0.
    """
    if axis.orientation is AxisOrientation.HORIZONTAL:
        along, contrast = structure_center(lateral_positions[:, :1], lateral_periods[:1])
        _, across_contrast = structure_center(lateral_positions[:, 1:], lateral_periods[1:])
        if across_contrast > contrast:
            warnings.warn(
                "форма неоднородна вдоль короткой стороны сильнее, чем вдоль длинной: похоже, ось лежит вдоль длинной "
                "стороны — карта по длинной стороне её не покажет (профиль от этого не зависит)",
                PandaWarning,
                stacklevel=3,
            )
        position = np.array([along[0], 0.0])
    else:
        position, contrast = structure_center(lateral_positions, lateral_periods)
    if contrast < WEAK_CONTRAST:
        warnings.warn(
            "форма почти однородна в плоскости стенок — положение оси определено плохо", PandaWarning, stacklevel=3
        )
    if axis.locate is AxisLocator.VOID_CENTER:
        shift = 0.5 * lateral_periods
        if axis.orientation is AxisOrientation.HORIZONTAL:
            shift = np.array([shift[0], 0.0])
        position = np.mod(position + shift, lateral_periods)
    centered: FloatArray = position
    return centered
