"""Capsule menisci covering an entire polygonal cross-section."""

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy.integrate import quad

from panda_md.core.types import FloatArray


def hemisphere_slice(radius: float, z: FloatArray, half_width: FloatArray) -> FloatArray:
    """Area of a semicircular slice with the transverse coordinate clipped to ±half_width."""
    disk_radius = np.sqrt(np.maximum(radius**2 - z**2, 0.0))
    ratio = np.clip(np.divide(half_width, disk_radius, out=np.zeros_like(z), where=disk_radius > 0), 0.0, 1.0)
    return np.asarray(disk_radius**2 * (ratio * np.sqrt(1.0 - ratio**2) + np.arcsin(ratio)), dtype=float)


@lru_cache(maxsize=256)
def capsule_end_integrals(
    radius: float,
    bottom: float,
    top: float,
    half_width: Callable[[float], float],
    inradius: float,
) -> tuple[float, float, float]:
    """Volume and free area of one end, plus the two ends' contact area per unit-side wall."""

    def cross_section(z: float) -> tuple[float, float]:
        disk_radius = math.sqrt(max(0.0, radius**2 - z**2))
        ratio = min(1.0, half_width(z) / disk_radius) if disk_radius > 0.0 else 1.0
        angle = math.asin(ratio)
        return disk_radius**2 * (angle + ratio * math.sqrt(1.0 - ratio**2)), 2.0 * radius * angle

    volume = quad(lambda z: cross_section(z)[0], bottom, top, epsabs=1e-10)[0]
    free_area = quad(lambda z: cross_section(z)[1], bottom, top, epsabs=1e-10)[0]
    wall_radius = math.sqrt(radius**2 - inradius**2)
    ratio = min(1.0, 0.5 / wall_radius)
    contact_area = wall_radius**2 * (2.0 * math.asin(ratio) + 2.0 * ratio * math.sqrt(1.0 - ratio**2))
    return volume, free_area, contact_area
