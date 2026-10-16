"""Контракт формы интерфейса: только формулы, без состояния, данных и оптимизатора."""

import math
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import numpy as np
from pydantic import Field
from scipy.integrate import trapezoid

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.errors import NotDerivedError, UnsupportedRegimeError
from panda_md.core.parameters import Parameter, ParameterValues
from panda_md.core.types import BoolArray, FloatArray, PandaModel

REGIME_BOUNDARY_DEG = 90.0  # граница смачивания и несмачивания
REGIME_GAP_DEG = 0.5  # θ ∈ (90° − зазор, 90° + зазор) не считается: в формулах tan θ и 1/cos θ
ANGLE_TOLERANCE_DEG = 1e-9  # градусы → радианы → градусы: degrees(radians(120)) = 119.99999999999999
CENTER_OF_MASS_GRID_POINTS = 20001  # сетка для численного центра масс, если форма не дала формулу


class PoreGeometry(PandaModel, ABC):
    """Безразмерная геометрия поры, которую получают формулы формы."""

    @abstractmethod
    def coordinate_bounds(self) -> tuple[float, float]:
        """Безразмерные координаты стенок вдоль оси профиля в системе поры (при δ = 0)."""

    @abstractmethod
    def profile_integral(self) -> float:
        """∫ρ dz / φ для любой формы этой поры: площадь исходного сечения / ширина слоя нормировки."""


class SurfaceTerms(PandaModel):
    """Площади межфазных поверхностей формы (безразмерные)."""

    liquid_liquid_area: float = Field(ge=0.0)  # S_LV
    liquid_solid_area: float = Field(ge=0.0)  # S_SL

    def capillary_energy(self, contact_angle_rad: float) -> float:
        """Капиллярный функционал S_LV − cos θ · S_SL: по его минимуму выбирается устойчивая форма."""
        return self.liquid_liquid_area - math.cos(contact_angle_rad) * self.liquid_solid_area


def regime_of(contact_angle_deg: float) -> Regime:
    return Regime.WETTING if contact_angle_deg < REGIME_BOUNDARY_DEG else Regime.NON_WETTING


GeometryT = TypeVar("GeometryT", bound=PoreGeometry)


class InterfaceShape(ABC, Generic[GeometryT]):
    """Одна форма интерфейса в одном типе поры. Значения параметров — во внутренних единицах."""

    name: ClassVar[str]
    parameters: ClassVar[tuple[Parameter, ...]]
    regimes: ClassVar[dict[Regime, tuple[float, float]]]  # диапазон θ в градусах для каждого выведенного режима
    frame: ClassVar[Frame]  # система отсчёта, в которой записаны формулы
    symmetry_axis: ClassVar[SymmetryAxis | None] = None  # ось для осевого центрирования; None — нет
    default_optimize: ClassVar[tuple[str, ...]] = (
        ParameterName.CONTACT_ANGLE.value,
        ParameterName.WALL_LAYER_THICKNESS.value,
    )

    @abstractmethod
    def feasible(self, geometry: GeometryT, values: ParameterValues) -> bool:
        """Существует ли форма при этих параметрах."""

    def profile(self, coordinate: FloatArray, geometry: GeometryT, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk на безразмерной координате. Вызывается только для допустимых параметров."""
        raise NotDerivedError(f"{self.name}: профиль плотности не выведен")

    def contains(self, points: FloatArray, geometry: GeometryT, values: ParameterValues) -> BoolArray:
        """Какие точки (n × 3, безразмерные: вдоль длинной стороны, вдоль короткой, вдоль оси профиля) — в жидкости.

        Ось симметрии формы (если есть) — в начале координат плоскости стенок; координаты в плоскости стенок
        периодичны с периодом ячейки. Вызывается только для допустимых параметров.
        """
        raise NotDerivedError(f"{self.name}: трёхмерная форма (contains) не выведена")

    def phi_limits(self, geometry: GeometryT, values: ParameterValues) -> tuple[float, float]:
        """Допустимый интервал φ при остальных параметрах."""
        raise NotDerivedError(f"{self.name}: интервал φ не выведен")

    def surface_terms(self, geometry: GeometryT, values: ParameterValues) -> SurfaceTerms:
        raise NotDerivedError(f"{self.name}: площади поверхностей не выведены")

    def center_of_mass(self, geometry: GeometryT, values: ParameterValues) -> float:
        """Центр масс жидкости вдоль координаты профиля, в системе формул формы. По умолчанию — численно."""
        lower, upper = geometry.coordinate_bounds()
        grid = np.linspace(lower, upper, CENTER_OF_MASS_GRID_POINTS)
        density = self.profile(grid, geometry, values)
        return float(trapezoid(grid * density, grid) / trapezoid(density, grid))

    def has_phi_limits(self) -> bool:
        return type(self).phi_limits is not InterfaceShape.phi_limits

    def has_contains(self) -> bool:
        return type(self).contains is not InterfaceShape.contains

    def has_profile(self) -> bool:
        return type(self).profile is not InterfaceShape.profile

    def has_surface_terms(self) -> bool:
        return type(self).surface_terms is not InterfaceShape.surface_terms

    def require_supported_contact_angle(self, contact_angle_deg: float) -> Regime:
        """Режим угла, если для него у формы выведены формулы; иначе UnsupportedRegimeError."""
        if abs(contact_angle_deg - REGIME_BOUNDARY_DEG) < REGIME_GAP_DEG - ANGLE_TOLERANCE_DEG:
            raise UnsupportedRegimeError(
                f"θ = {contact_angle_deg:.2f}° внутри зазора ±{REGIME_GAP_DEG}° у 90°: формулы там не определены"
            )
        regime = regime_of(contact_angle_deg)
        if regime not in self.regimes:
            raise UnsupportedRegimeError(
                f"форма {self.name}: режим {regime.value} (θ = {contact_angle_deg:.4g}°) не объявлен — "
                f"формулы не выведены"
            )
        lower, upper = self.regimes[regime]
        if not lower - ANGLE_TOLERANCE_DEG <= contact_angle_deg <= upper + ANGLE_TOLERANCE_DEG:
            raise UnsupportedRegimeError(
                f"форма {self.name}: θ = {contact_angle_deg:.4g}° вне диапазона режима {regime.value} "
                f"[{lower}, {upper}]"
            )
        return regime
