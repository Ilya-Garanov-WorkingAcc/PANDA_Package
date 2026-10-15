"""Профиль плотности из данных и его безразмерный вид."""

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, Normalization
from panda_md.core.errors import InvalidProfileError
from panda_md.core.types import Array1D, PandaModel

MIN_PROFILE_POINTS = 2


class DensityProfile(PandaModel):
    """Профиль плотности анализируемой жидкости вдоль оси профиля поры."""

    coordinate_nm: Array1D  # центры бинов, нм, в системе frame
    density_per_nm3: Array1D  # число частиц на нм³
    frame: Frame = Frame.PORE
    particle_count: float | None = Field(default=None, gt=0.0)  # N — частиц в поре
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)  # V — объём поры, к которому отнесена φ

    @model_validator(mode="after")
    def _check_arrays(self) -> "DensityProfile":
        if self.coordinate_nm.shape != self.density_per_nm3.shape:
            raise ValueError("длины координаты и плотности различаются")
        if self.coordinate_nm.size < MIN_PROFILE_POINTS:
            raise ValueError(f"в профиле меньше {MIN_PROFILE_POINTS} точек")
        if not np.all(np.diff(self.coordinate_nm) > 0):
            raise ValueError("координата профиля не возрастает")
        if np.any(self.density_per_nm3 < 0):
            raise ValueError("плотность отрицательна")
        if not np.any(self.density_per_nm3 > 0):
            raise ValueError("профиль пуст: плотность везде нулевая — проверь выборку атомов")
        return self

    def mean_density_per_nm3(self) -> float:
        """n̄ = N / V — средняя плотность жидкости в поре."""
        if self.particle_count is None or self.pore_volume_nm3 is None:
            raise InvalidProfileError("для нормировки mean нужны particle_count и pore_volume_nm3")
        return self.particle_count / self.pore_volume_nm3


class ReducedProfile(PandaModel):
    """Профиль в безразмерном виде: координата в долях размера поры, плотность после нормировки."""

    coordinate: Array1D
    density: Array1D
    frame: Frame
    normalization: Normalization
    density_scale_per_nm3: float = Field(gt=0.0)  # на что поделена плотность: ρ_bulk или n̄


def reduce_profile(
    profile: DensityProfile,
    characteristic_size_nm: float,
    normalization: Normalization,
    density_bulk_per_nm3: float | None,
) -> ReducedProfile:
    if normalization is Normalization.BULK:
        if density_bulk_per_nm3 is None:
            raise InvalidProfileError("для нормировки bulk нужна объёмная плотность density_bulk")
        scale = density_bulk_per_nm3
    elif normalization is Normalization.MEAN:
        scale = profile.mean_density_per_nm3()
    else:
        raise ValueError(f"нормировка {normalization.value} должна быть разрешена до приведения профиля")
    return ReducedProfile(
        coordinate=profile.coordinate_nm / characteristic_size_nm,
        density=profile.density_per_nm3 / scale,
        frame=profile.frame,
        normalization=normalization,
        density_scale_per_nm3=scale,
    )
