"""Slit with an L × kL base against independent formulas of Shistko et al., Mol. Simul. (2025) at θ = 180°, δ = 0.

The formulas of the paper (equations (6)–(27)) are rewritten here from scratch, directly. The φ limits of the
perforation in the paper text are ambiguous (equation (23)), so they are derived here by integrating the hole
volume: at d = 1 the hole closes (φ_max), at d = kl it touches its own image (φ_min).
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
    center = -0.5 + radius  # at θ = 180° the sphere touches the lower wall
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        math.pi / (aspect_ratio * length_ratio**2) * (radius**2 - (COORDINATE - center) ** 2),  # (8)
        0.0,
    )
    np.testing.assert_allclose(droplet.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = droplet.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(4 * math.pi * radius**2)  # (9)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_doughnut_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    doughnut = pore(aspect_ratio, length_ratio).interface("doughnut")
    side = aspect_ratio * length_ratio
    lower = math.pi / (6 * aspect_ratio * length_ratio**2)  # (11)
    upper = math.pi / (4 * aspect_ratio * length_ratio**2) * ((side - 1) ** 2 + math.pi * (side - 1) / 2 + 2 / 3)
    assert doughnut.phi_limits(theta=FULL_NON_WETTING, phi=0.1) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    spot = -math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * phi * aspect_ratio * length_ratio**2 / math.pi)
    expected = math.pi / (4 * aspect_ratio * length_ratio**2) * (np.sqrt(1 - 4 * COORDINATE**2) + spot) ** 2  # (12)
    np.testing.assert_allclose(doughnut.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_worm_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    worm = pore(aspect_ratio, length_ratio).interface("worm")
    upper = min(math.pi / (4 * length_ratio), math.pi * length_ratio / 4)  # (15)
    assert worm.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = math.sqrt(phi * length_ratio / math.pi)  # (14)
    center = -0.5 + radius
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        2 / length_ratio * np.sqrt(np.maximum(radius**2 - (COORDINATE - center) ** 2, 0.0)),  # (16)
        0.0,
    )
    np.testing.assert_allclose(worm.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = worm.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
        2 * math.pi * aspect_ratio * length_ratio * radius  # (17)
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_roll_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    roll = pore(aspect_ratio, length_ratio).interface("roll")
    lower, upper = math.pi / (4 * length_ratio), 1 - (1 - math.pi / 4) / length_ratio  # (19)
    assert roll.phi_limits(theta=FULL_NON_WETTING, phi=0.5) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    spot = phi * length_ratio - math.pi / 4  # (18)
    expected = (np.sqrt(1 - 4 * COORDINATE**2) + spot) / length_ratio  # (20)
    np.testing.assert_allclose(roll.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = roll.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
        aspect_ratio * length_ratio * (2 * spot + math.pi)  # (21)
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_perforation_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    perforation = pore(aspect_ratio, length_ratio).interface("perforation")
    side = aspect_ratio * length_ratio
    base = aspect_ratio * length_ratio**2
    lower = 1 - math.pi / (4 * base) * (side**2 - math.pi * side / 2 + 2 / 3)  # the hole touches its image
    upper = 1 - math.pi / (4 * base) * (5 / 3 - math.pi / 2)  # the hole closes
    if side < math.pi / 4:
        assert perforation.phi_limits(theta=FULL_NON_WETTING, phi=0.9) == pytest.approx((upper, upper), abs=TOLERANCE)
        assert not perforation.is_feasible(theta=FULL_NON_WETTING, phi=0.5 * (lower + upper))
        return
    assert perforation.phi_limits(theta=FULL_NON_WETTING, phi=0.9) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    hole = math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * (1 - phi) * base / math.pi)  # (22)
    expected = 1 - math.pi / (4 * base) * (np.sqrt(1 - 4 * COORDINATE**2) - hole) ** 2  # (24)
    np.testing.assert_allclose(
        perforation.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
def test_layer_matches_rectcell(aspect_ratio: float) -> None:
    layer = pore(aspect_ratio, 3.0).interface("layer")
    expected = np.where(np.abs(COORDINATE) < 0.3, 1.0, 0.0)  # (26)
    np.testing.assert_array_equal(layer.profile(COORDINATE, phi=0.6), expected)
    assert layer.surface_terms(phi=0.6).liquid_liquid_area == pytest.approx(2 * aspect_ratio * 9.0)  # (27)


@pytest.mark.parametrize("shape_name", ["droplet", "doughnut"])
def test_axisymmetric_shape_in_rectangle_equals_square_of_short_side(shape_name: str) -> None:
    """ρ(z; l, k, φ) = k · ρ(z; kl, 1, φ/k): an L × kL cell is equivalent to a square kL × kL one."""
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
