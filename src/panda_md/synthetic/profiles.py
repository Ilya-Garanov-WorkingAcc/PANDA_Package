"""Синтетические профили плотности из аналитической модели — для тестов и валидации."""

from collections.abc import Mapping

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.errors import InfeasibleParametersError
from panda_md.core.model import InterfaceModel
from panda_md.core.profile import DensityProfile


def synthetic_profile(
    model: InterfaceModel,
    values: Mapping[str, float],
    *,
    density_bulk_per_nm3: float,
    bin_count: int = 200,
    noise_std: float = 0.0,
    frame: Frame = Frame.PORE,
    offset: float = 0.0,
    seed: int | None = None,
) -> DensityProfile:
    """Профиль, который дала бы МД при идеальной форме с параметрами values (единицы пользователя).

    noise_std — шум в долях ρ_bulk на бин; offset — сдвиг интерфейса вдоль оси профиля в долях размера поры
    (моделирует ошибку положения стенок); frame — в какой системе отсчёта вернуть координату.
    """
    pore = model.pore
    geometry = pore.geometry()
    lower, upper = geometry.coordinate_bounds()
    edges = np.linspace(lower, upper, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    internal = model.supported_values(values)
    if not model.shape.feasible(geometry, internal):
        raise InfeasibleParametersError(f"{model}: форма не существует при {dict(values)}")
    density = density_bulk_per_nm3 * model.shape.profile(centers - offset, geometry, internal)
    if noise_std > 0.0:
        generator = np.random.default_rng(seed)
        density = density + generator.normal(0.0, noise_std * density_bulk_per_nm3, size=density.shape)
    density = np.clip(density, 0.0, None)

    coordinate = centers
    if frame is Frame.LIQUID:
        coordinate = centers - (model.shape.center_of_mass(geometry, internal) + offset)
    volume_fraction = internal[ParameterName.VOLUME_FRACTION]
    return DensityProfile(
        coordinate_nm=coordinate * pore.characteristic_size_nm(),
        density_per_nm3=density,
        frame=frame,
        particle_count=density_bulk_per_nm3 * volume_fraction * pore.volume_nm3(),
        pore_volume_nm3=pore.volume_nm3(),
    )
