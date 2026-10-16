"""Периодические границы вдоль оси ящика и время в пользовательских единицах."""

import numpy as np
import pytest

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import circular_mean, largest_gap, wrap_centered
from panda_md.core.timing import to_picoseconds

PERIOD = 10.0


def test_wrap_centered_maps_to_nearest_image_around_zero() -> None:
    values = np.array([-12.0, -5.0, -4.9, 0.0, 4.9, 5.0, 17.5])
    np.testing.assert_allclose(wrap_centered(values, PERIOD), [-2.0, -5.0, -4.9, 0.0, 4.9, -5.0, -2.5])


def test_circular_mean_of_cloud_across_boundary_is_not_box_middle() -> None:
    cloud = np.array([9.0, 9.5, 0.5, 1.0])  # облако вокруг 0 ≡ 10, а не вокруг 5
    center = circular_mean(cloud, PERIOD)
    assert min(center, PERIOD - center) == pytest.approx(0.0, abs=1e-12)
    assert circular_mean(np.array([2.0, 3.0, 4.0]), PERIOD) == pytest.approx(3.0)


def test_largest_gap_finds_pore_through_box_boundary() -> None:
    walls = np.array([0.2, 1.0, 2.0, 2.9])  # подложка внизу, её образ сверху: пора от 2.9 до 10.2
    start, end = largest_gap(walls, PERIOD)
    assert (start, end) == pytest.approx((2.9, 10.2))
    inner = np.array([1.0, 1.5, 8.0, 8.5])  # две стенки внутри ящика: пора от 1.5 до 8.0
    assert largest_gap(inner, PERIOD) == pytest.approx((1.5, 8.0))


def test_empty_selections_are_errors() -> None:
    with pytest.raises(InvalidTrajectoryError):
        circular_mean(np.array([]), PERIOD)
    with pytest.raises(InvalidTrajectoryError):
        largest_gap(np.array([]), PERIOD)


@pytest.mark.parametrize(
    ("value", "picoseconds"),
    [
        (2000.0, 2000.0),
        (5, 5.0),
        ("2 ns", 2000.0),
        ("1.5ns", 1500.0),
        ("500 ps", 500.0),
        ("1 us", 1e6),
        ("10 fs", 0.01),
    ],
)
def test_time_values_are_converted_to_picoseconds(value: float | str, picoseconds: float) -> None:
    assert to_picoseconds(value) == pytest.approx(picoseconds)


@pytest.mark.parametrize("value", ["2", "two ns", "2 hours"])
def test_unparsable_time_is_an_error(value: str) -> None:
    with pytest.raises(ValueError, match=r"время|единица"):
        to_picoseconds(value)
