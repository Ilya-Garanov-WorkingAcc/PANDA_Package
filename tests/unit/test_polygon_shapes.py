"""Boundary and surface regressions for revised polygon morphologies."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from panda_md import SquarePore, TrianglePore
from panda_md.core.errors import UnknownShapeError

TRIANGLE = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0)
SQUARE = SquarePore(side_nm=1.0, length_nm=4.0, layer_width_nm=1.0)


@pytest.mark.parametrize("theta", [135.0, 134.999])
def test_square_corner_droplet_boundary_is_strict(theta: float) -> None:
    droplet = SQUARE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=theta, delta=0.0, phi=0.02) == (theta < 135.0)


def test_triangle_corner_droplet_matches_dry_edge_limit() -> None:
    wet = TRIANGLE.interface("droplet_w2c1")
    dry = TRIANGLE.interface("droplet_w2c0")
    assert not wet.is_feasible(theta=120.0, delta=0.025, phi=0.03)
    assert not dry.is_feasible(theta=120.0, delta=0.025, phi=0.03)
    before = wet.surface_terms(theta=119.999, delta=0.025, phi=0.03)
    after = dry.surface_terms(theta=120.001, delta=0.025, phi=0.03)
    assert before.liquid_liquid_area == pytest.approx(after.liquid_liquid_area, rel=1e-4)
    assert before.liquid_solid_area == pytest.approx(after.liquid_solid_area, rel=1e-4)


def test_triangle_capsule_free_area_against_uniform_sphere_samples() -> None:
    theta = 110.0
    side = 1.0 - 2.0 * math.sqrt(3.0) * 0.03
    radius = -side / (2.0 * math.sqrt(3.0) * math.cos(math.radians(theta)))
    z = np.linspace(-radius, radius, 400_000)
    azimuth = np.arange(z.size) * (math.pi * (3.0 - math.sqrt(5.0)))
    y = np.sqrt(np.maximum(0.0, radius**2 - z**2)) * np.cos(azimuth)
    within = (z >= -side / (2.0 * math.sqrt(3.0))) & (z + math.sqrt(3.0) * np.abs(y) <= side / math.sqrt(3.0))
    expected = 4.0 * math.pi * radius**2 * np.mean(within)
    terms = TRIANGLE.interface("capsule_w3c3").surface_terms(theta=theta, delta=0.03, phi=0.55)
    assert terms.liquid_liquid_area == pytest.approx(expected, rel=3e-4)


def test_square_perforation_uses_gas_volume_and_area_of_revolution() -> None:
    model = SQUARE.interface("perforation")
    theta, delta, phi = 115.0, 0.045, 0.75
    half_width = 0.5 - delta
    length = SQUARE.length_nm
    radius = -half_width / math.cos(math.radians(theta))

    def hole_radius(z: float) -> float:
        density = model.profile(np.array([z]), theta=theta, delta=delta, phi=phi)[0]
        return math.sqrt((2.0 * half_width - density) * length / math.pi)

    gas_volume = quad(lambda z: math.pi * hole_radius(z) ** 2, -half_width, half_width)[0]
    area = quad(
        lambda z: 2.0 * math.pi * hole_radius(z) * radius / math.sqrt(radius**2 - z**2),
        -half_width,
        half_width,
    )[0]
    terms = model.surface_terms(theta=theta, delta=delta, phi=phi)
    assert gas_volume == pytest.approx(((2.0 * half_width) ** 2 - phi) * length)
    assert terms.liquid_liquid_area == pytest.approx(area)
    wetted = 8.0 * half_width * length - 2.0 * math.pi * hole_radius(half_width) ** 2
    assert terms.liquid_solid_area == pytest.approx(wetted)


@pytest.mark.parametrize("old_name", ["filament_w0c0_round", "filament_w0c0_straight"])
def test_square_rejects_retired_filament_names(old_name: str) -> None:
    with pytest.raises(UnknownShapeError, match=old_name):
        SQUARE.interface(old_name)


@pytest.mark.parametrize(("name", "corners"), [("filament_w3c1", 2), ("filament_w3c2", 1)])
def test_triangle_filament_area_counts_empty_corners(name: str, corners: int) -> None:
    theta, phi, side = 150.0, 0.75, 0.94
    delta = (1.0 - side) / (2.0 * math.sqrt(3.0))
    theta_prime = math.pi - math.radians(theta)
    arc = 2.0 * (math.pi / 3.0 - theta_prime)
    missing_area = (side**2 - phi) * math.sqrt(3.0) / 4.0
    # Wedge triangle with an opening of 60°, minus a circular segment.
    contact_per_radius = 2.0 * math.sin(arc / 2.0)
    corner_factor = math.sqrt(3.0) * contact_per_radius**2 / 4.0 - (arc - math.sin(arc)) / 2.0
    radius = math.sqrt(missing_area / (corners * corner_factor))
    terms = TRIANGLE.interface(name).surface_terms(theta=theta, delta=delta, phi=phi)
    assert terms.liquid_liquid_area == pytest.approx(corners * radius * arc * TRIANGLE.length_nm)
    wetted = 3.0 * side - 2.0 * corners * radius * contact_per_radius
    assert terms.liquid_solid_area == pytest.approx(wetted * TRIANGLE.length_nm)
