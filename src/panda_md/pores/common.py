"""Geometry shared by the shapes of different pores: spherical and circular segments."""

import math

import numpy as np

from panda_md.core.types import FloatArray


def spherical_cap_volume_factor(contact_angle: float) -> float:
    """2 − 3 cos θ + cos³ θ: spherical cap volume = π R³ / 3 · (this factor)."""
    cosine = math.cos(contact_angle)
    return 2.0 - 3.0 * cosine + cosine**3


def spherical_cap_centroid_offset(radius: float, contact_angle: float) -> float:
    """Distance from the sphere centre to the cap centroid: 3 R cos⁴(θ/2) / (2 + cos θ)."""
    return 3.0 * radius * math.cos(0.5 * contact_angle) ** 4 / (2.0 + math.cos(contact_angle))


def circular_segment_factor(contact_angle: float) -> float:
    """θ − sin θ cos θ: area of a circular segment with angle θ = R² · (this factor)."""
    return contact_angle - math.sin(contact_angle) * math.cos(contact_angle)


def circular_segment_centroid_offset(radius: float, contact_angle: float) -> float:
    """Distance from the circle centre to the segment centroid: 4 R sin³θ / (3 (2θ − sin 2θ))."""
    return 4.0 * radius * math.sin(contact_angle) ** 3 / (3.0 * (2.0 * contact_angle - math.sin(2.0 * contact_angle)))


def circular_segment_area(radius: FloatArray, distance: FloatArray | float) -> FloatArray:
    """Area of the segment of a circle of radius r cut off by a chord at distance d from the centre (zero for r ≤ d)."""
    ratio = np.clip(np.divide(distance, radius, out=np.ones_like(radius), where=radius > 0), -1.0, 1.0)
    return radius**2 * np.arccos(ratio) - distance * np.sqrt(np.maximum(0.0, radius**2 - distance**2))
