"""Synthetic density profiles from an analytical model, for tests and validation."""

from collections.abc import Mapping

import numpy as np

from panda_md.core.enums import Frame, NoiseModel, ParameterName, ProfileSampling
from panda_md.core.errors import InfeasibleParametersError
from panda_md.core.model import InterfaceModel
from panda_md.core.profile import DensityProfile

BIN_SUBSAMPLES = 16  # points per bin when averaging the model over a bin


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
    """Return the profile MD would give for an ideal shape with parameters values (user units).

    noise_std — noise per bin in units of ρ_bulk (for POISSON — at density ρ_bulk; σ ∝ √ρ); offset — interface shift
    along the profile axis in units of the pore size (wall position error); frame — frame of reference of the
    coordinate; sampling=BIN_AVERAGE — the bin density is averaged over the bin, as in an MD histogram (CENTER — the
    value at the bin centre).
    """
    pore = model.pore
    geometry = pore.geometry()
    lower, upper = geometry.coordinate_bounds()
    edges = np.linspace(lower, upper, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    internal = model.supported_values(values)
    if not model.shape.feasible(geometry, internal):
        raise InfeasibleParametersError(f"{model}: the shape does not exist at {dict(values)}")
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
