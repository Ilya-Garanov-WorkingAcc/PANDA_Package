"""Геометрия, общая для форм разных пор: сферический сегмент."""

import math


def spherical_cap_volume_factor(contact_angle: float) -> float:
    """2 − 3 cos θ + cos³ θ: объём сферического сегмента = π R³ / 3 · (этот множитель)."""
    cosine = math.cos(contact_angle)
    return 2.0 - 3.0 * cosine + cosine**3


def spherical_cap_centroid_offset(radius: float, contact_angle: float) -> float:
    """Расстояние от центра сферы до центра масс сегмента: 3 R cos⁴(θ/2) / (2 + cos θ)."""
    return 3.0 * radius * math.cos(0.5 * contact_angle) ** 4 / (2.0 + math.cos(contact_angle))


def circular_segment_factor(contact_angle: float) -> float:
    """θ − sin θ cos θ: площадь кругового сегмента с углом θ = R² · (этот множитель)."""
    return contact_angle - math.sin(contact_angle) * math.cos(contact_angle)


def circular_segment_centroid_offset(radius: float, contact_angle: float) -> float:
    """Расстояние от центра окружности до центра масс сегмента: 4 R sin³θ / (3 (2θ − sin 2θ))."""
    return 4.0 * radius * math.sin(contact_angle) ** 3 / (3.0 * (2.0 * contact_angle - math.sin(2.0 * contact_angle)))
