"""Слой (layer) в щели: плоский слой жидкости посередине, не касается стенок.

Профиль — ступенька ширины φ (Semenchuk et al., 2025; alpha.py); граница существования φ ≤ 1 − 2δ — alpha.py;
площадь — две плоские поверхности основания, S = 2 k l² (Shistko et al., 2025, уравнение (27)).
Контактного угла у слоя нет: формула от θ не зависит, поэтому θ не параметр этой формы, а свободна по умолчанию
только φ. От k профиль не зависит.
"""

from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection, SlitShape


@SlitPore.register
class SlitLayer(SlitShape):
    name: ClassVar[str] = "layer"
    cross_section: ClassVar[CrossSection] = CrossSection.FULL
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}  # контактного угла нет
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        return 0.0, self.accessible_width(values)

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # слой посередине щели

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        return SurfaceTerms(liquid_liquid_area=2.0 * geometry.base_area(), liquid_solid_area=0.0)

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 1 при |z| < φ/2, иначе 0."""
        half_thickness = 0.5 * values[ParameterName.VOLUME_FRACTION]
        return np.where(np.abs(coordinate) < half_thickness, 1.0, 0.0)
