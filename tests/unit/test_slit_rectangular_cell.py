"""Щель с основанием L × kL против независимых формул Shistko et al., Mol. Simul. (2025) при θ = 180°, δ = 0.

Формулы статьи (уравнения (6)–(27)) записаны здесь заново, в лоб. Пределы φ перфорации в тексте статьи
неоднозначны (уравнение (23)), поэтому они выведены здесь интегрированием объёма отверстия: при d = 1 отверстие
закрывается (φ_max), при d = kl касается своего образа (φ_min).
"""

import math

import numpy as np
import pytest

from panda_md import SlitPore

ASPECT_RATIOS = (0.2, 0.6, 1.0)
LENGTH_RATIOS = (2.0, 4.0)
COORDINATE = np.linspace(-0.4975, 0.4975, 200)
TOLERANCE = 1e-12
FULL_NON_WETTING = 180.0


def pore(aspect_ratio: float, length_ratio: float) -> SlitPore:
    return SlitPore(width_nm=1.0, length_nm=length_ratio, base_aspect_ratio=aspect_ratio)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_droplet_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    droplet = pore(aspect_ratio, length_ratio).interface("droplet")
    upper = min(math.pi / (6 * aspect_ratio * length_ratio**2), math.pi * aspect_ratio**2 * length_ratio / 6)  # (7)
    assert droplet.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = (3 * phi * aspect_ratio * length_ratio**2 / (4 * math.pi)) ** (1 / 3)  # (6)
    center = -0.5 + radius  # при θ = 180° сфера касается нижней стенки
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        math.pi / (aspect_ratio * length_ratio**2) * (radius**2 - (COORDINATE - center) ** 2),  # (8)
        0.0,
    )
    np.testing.assert_allclose(droplet.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = droplet.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(4 * math.pi * radius**2)  # (9)


@pytest.mark.parametrize("shape_name", ["droplet"])
def test_axisymmetric_shape_in_rectangle_equals_square_of_short_side(shape_name: str) -> None:
    """ρ(z; l, k, φ) = k · ρ(z; kl, 1, φ/k): ячейка L × kL эквивалентна квадратной kL × kL."""
    rectangle = SlitPore(width_nm=1.0, length_nm=5.0, base_aspect_ratio=0.4).interface(shape_name)
    square = SlitPore(width_nm=1.0, length_nm=2.0).interface(shape_name)
    values = {"theta": 140.0, "delta": 0.03}
    lower, upper = rectangle.phi_limits(**values, phi=0.1)
    phi = 0.5 * (lower + upper)
    np.testing.assert_allclose(
        rectangle.profile(COORDINATE, **values, phi=phi),
        0.4 * square.profile(COORDINATE, **values, phi=phi / 0.4),
        atol=TOLERANCE,
    )


def test_pore_volume_uses_rectangular_base() -> None:
    assert SlitPore(width_nm=2.0, length_nm=10.0, base_aspect_ratio=0.3).volume_nm3() == pytest.approx(60.0)
