"""Профиль плотности из данных, набор профилей по блокам времени и безразмерный вид профиля."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, Normalization, ProfileCentering
from panda_md.core.errors import InvalidProfileError
from panda_md.core.types import Array1D, Array2D, FloatArray, PandaModel

MIN_PROFILE_POINTS = 2
CSV_COMMENT = "# "
CSV_METADATA_FIELDS = ("frame", "particle_count", "pore_volume_nm3", "flipped", "frame_count")


class DensityProfile(PandaModel):
    """Профиль плотности анализируемой жидкости вдоль оси профиля поры."""

    coordinate_nm: Array1D  # центры бинов, нм, в системе frame
    density_per_nm3: Array1D  # число частиц на нм³
    density_sem_per_nm3: Array1D | None = None  # ошибка среднего по блокам
    frame: Frame = Frame.PORE
    particle_count: float | None = Field(default=None, gt=0.0)  # N — частиц в поре
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)  # V — объём поры, к которому отнесена φ
    flipped: bool = False  # профиль отражён z → −z (жидкость была у верхней стенки)
    frame_count: int | None = Field(default=None, ge=1)  # сколько кадров усреднено

    @model_validator(mode="after")
    def _check_arrays(self) -> "DensityProfile":
        if self.coordinate_nm.shape != self.density_per_nm3.shape:
            raise ValueError("длины координаты и плотности различаются")
        if self.density_sem_per_nm3 is not None and self.density_sem_per_nm3.shape != self.density_per_nm3.shape:
            raise ValueError("длины плотности и её ошибки различаются")
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

    def save(self, path: str | Path) -> None:
        """Сохранить: .npz — полностью; .csv — столбцы и метаданные в строках-комментариях."""
        target = Path(path)
        if target.suffix == ".npz":
            _save_npz(target, _profile_arrays(self), self._metadata())
        elif target.suffix == ".csv":
            header = "\n".join(f"{key}: {json.dumps(value)}" for key, value in self._metadata().items())
            arrays = _profile_arrays(self)
            columns = list(arrays)
            table = np.column_stack([arrays[name] for name in columns])
            np.savetxt(target, table, delimiter=",", header=header + "\n" + ",".join(columns), comments=CSV_COMMENT)
        else:
            raise ValueError(f"формат {target.suffix!r} не поддерживается: .npz или .csv")

    @classmethod
    def load(cls, path: str | Path) -> "DensityProfile":
        source = Path(path)
        if source.suffix == ".npz":
            with np.load(source) as archive:
                metadata = json.loads(str(archive["metadata"]))
                arrays = {name: archive[name] for name in archive.files if name != "metadata"}
        elif source.suffix == ".csv":
            metadata, arrays = _read_csv(source)
        else:
            raise ValueError(f"формат {source.suffix!r} не поддерживается: .npz или .csv")
        return cls(**arrays, **metadata)

    def _metadata(self) -> dict[str, object]:
        dumped = self.model_dump(mode="json")
        return {name: dumped[name] for name in CSV_METADATA_FIELDS}


def _save_npz(path: Path, arrays: Mapping[str, FloatArray], metadata: Mapping[str, object]) -> None:
    """Массивы — по именам, метаданные — JSON-строкой под ключом metadata."""
    # Any: заглушки numpy типизируют **kwds у savez как bool (allow_pickle); значения — массивы и строка
    payload: dict[str, Any] = {**arrays, "metadata": json.dumps(dict(metadata))}
    np.savez(path, **payload)


def _require_npz(path: Path) -> None:
    """ProfileSet хранится только в .npz: иначе numpy молча дописывает суффикс, и load по тому же имени падает."""
    if path.suffix != ".npz":
        raise ValueError(f"набор профилей хранится в .npz, а не в {path.name!r}")


def _profile_arrays(profile: DensityProfile) -> dict[str, FloatArray]:
    arrays = {"coordinate_nm": profile.coordinate_nm, "density_per_nm3": profile.density_per_nm3}
    if profile.density_sem_per_nm3 is not None:
        arrays["density_sem_per_nm3"] = profile.density_sem_per_nm3
    return arrays


def _read_csv(source: Path) -> tuple[dict[str, object], dict[str, FloatArray]]:
    metadata: dict[str, object] = {}
    columns: list[str] = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith(CSV_COMMENT):
                break
            content = line[len(CSV_COMMENT) :].strip()
            key, separator, value = content.partition(": ")
            if separator and key in CSV_METADATA_FIELDS:
                metadata[key] = json.loads(value)
            else:
                columns = content.split(",")
    table = np.loadtxt(source, delimiter=",", comments=CSV_COMMENT.strip(), ndmin=2)
    return metadata, {name: table[:, index] for index, name in enumerate(columns)}


class ProfileSet(PandaModel):
    """Профили по блокам времени: блоки × бины. Среднее по блокам — профиль для фита, разброс — его ошибка."""

    bin_edges_nm: Array1D  # края бинов, нм, в системе frame
    block_densities_per_nm3: Array2D  # блоки × бины
    block_start_ps: Array1D
    block_stop_ps: Array1D
    block_frame_counts: tuple[int, ...]
    block_particle_counts: Array1D  # среднее по кадрам блока число частиц в поре (N)
    frame: Frame
    centering: ProfileCentering
    selection: str
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)
    flipped: bool = False

    @model_validator(mode="after")
    def _check_shapes(self) -> "ProfileSet":
        block_count, bin_count = self.block_densities_per_nm3.shape
        if self.bin_edges_nm.shape != (bin_count + 1,) or not np.all(np.diff(self.bin_edges_nm) > 0):
            raise ValueError("края бинов: по одному больше, чем бинов, строго по возрастанию")
        for name in ("block_start_ps", "block_stop_ps", "block_particle_counts"):
            if getattr(self, name).shape != (block_count,):
                raise ValueError(f"{name}: по одному значению на блок")
        if len(self.block_frame_counts) != block_count or min(self.block_frame_counts, default=0) < 1:
            raise ValueError("в каждом блоке должен быть хотя бы один кадр")
        return self

    @property
    def coordinate_nm(self) -> FloatArray:
        centers: FloatArray = 0.5 * (self.bin_edges_nm[:-1] + self.bin_edges_nm[1:])
        return centers

    @property
    def block_count(self) -> int:
        return int(self.block_densities_per_nm3.shape[0])

    def std(self) -> FloatArray:
        """Стандартное отклонение плотности по блокам (0 при одном блоке)."""
        if self.block_count < 2:
            return np.zeros(self.block_densities_per_nm3.shape[1])
        spread: FloatArray = np.std(self.block_densities_per_nm3, axis=0, ddof=1)
        return spread

    def sem(self) -> FloatArray:
        """Ошибка среднего по блокам: std / √(число блоков)."""
        error: FloatArray = self.std() / np.sqrt(self.block_count)
        return error

    def mean(self) -> DensityProfile:
        """Средний профиль по блокам с ошибкой среднего."""
        return self._profile(
            np.mean(self.block_densities_per_nm3, axis=0),
            float(np.mean(self.block_particle_counts)),
            sum(self.block_frame_counts),
            self.sem() if self.block_count > 1 else None,
        )

    def block(self, index: int) -> DensityProfile:
        return self._profile(
            self.block_densities_per_nm3[index],
            float(self.block_particle_counts[index]),
            self.block_frame_counts[index],
            None,
        )

    def _profile(
        self, density: FloatArray, particle_count: float, frame_count: int, sem: FloatArray | None
    ) -> DensityProfile:
        return DensityProfile(
            coordinate_nm=self.coordinate_nm,
            density_per_nm3=density,
            density_sem_per_nm3=sem,
            frame=self.frame,
            particle_count=particle_count if particle_count > 0 else None,
            pore_volume_nm3=self.pore_volume_nm3,
            flipped=self.flipped,
            frame_count=frame_count,
        )

    def save(self, path: str | Path) -> None:
        """Сохранить в .npz (массивы и метаданные)."""
        _require_npz(Path(path))
        arrays = {
            name: getattr(self, name)
            for name in (
                "bin_edges_nm",
                "block_densities_per_nm3",
                "block_start_ps",
                "block_stop_ps",
                "block_particle_counts",
            )
        }
        metadata = self.model_dump(mode="json", exclude=set(arrays))
        _save_npz(Path(path), arrays, metadata)

    @classmethod
    def load(cls, path: str | Path) -> "ProfileSet":
        _require_npz(Path(path))
        with np.load(Path(path)) as archive:
            metadata = json.loads(str(archive["metadata"]))
            arrays = {name: archive[name] for name in archive.files if name != "metadata"}
        return cls(**arrays, **metadata)


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
