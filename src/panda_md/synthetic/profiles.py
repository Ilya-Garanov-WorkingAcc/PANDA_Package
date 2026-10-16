"""Синтетические профили плотности из аналитической модели — для тестов и валидации."""

from collections.abc import Mapping

import numpy as np

from panda_md.core.enums import Frame, NoiseModel, ParameterName, ProfileSampling
from panda_md.core.errors import InfeasibleParametersError
from panda_md.core.model import InterfaceModel
from panda_md.core.profile import DensityProfile

BIN_SUBSAMPLES = 16  # точек на бин при усреднении модели по бину


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
    noise_model: NoiseModel = NoiseModel.CONSTANT,
    sampling: ProfileSampling = ProfileSampling.CENTER,
) -> DensityProfile:
    """Профиль, который дала бы МД при идеальной форме с параметрами values (единицы пользователя).

    noise_std — шум в долях ρ_bulk на бин (у POISSON — при плотности ρ_bulk; σ ∝ √ρ); offset — сдвиг интерфейса
    вдоль оси профиля в долях размера поры (ошибка положения стенок); frame — система отсчёта координаты;
    sampling=BIN_AVERAGE — плотность в бине усреднена по бину, как в гистограмме МД (CENTER — значение в центре).
    """
    pore = model.pore
    geometry = pore.geometry()
    lower, upper = geometry.coordinate_bounds()
    edges = np.linspace(lower, upper, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    internal = model.supported_values(values)
    if not model.shape.feasible(geometry, internal):
        raise InfeasibleParametersError(f"{model}: форма не существует при {dict(values)}")
    if sampling is ProfileSampling.BIN_AVERAGE:
        fractions = (np.arange(BIN_SUBSAMPLES) + 0.5) / BIN_SUBSAMPLES
        points = edges[:-1, np.newaxis] + np.diff(edges)[:, np.newaxis] * fractions
        shape_density = model.shape.profile(points.ravel() - offset, geometry, internal).reshape(points.shape)
        clean = density_bulk_per_nm3 * shape_density.mean(axis=1)
    else:
        clean = density_bulk_per_nm3 * model.shape.profile(centers - offset, geometry, internal)
    density = clean
    if noise_std > 0.0:
        generator = np.random.default_rng(seed)
        sigma = np.full_like(clean, noise_std * density_bulk_per_nm3)
        if noise_model is NoiseModel.POISSON:
            sigma = sigma * np.sqrt(np.maximum(clean, 0.0) / density_bulk_per_nm3)
        density = clean + generator.normal(0.0, 1.0, size=clean.shape) * sigma
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
