"""Density profile from the data, a set of profiles over time blocks, and the dimensionless form of a profile."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, Normalization, ProfileCentering
from panda_md.core.errors import InvalidProfileError
from panda_md.core.types import Array1D, Array2D, FloatArray, PandaModel
from panda_md.core.version import library_version

MIN_PROFILE_POINTS = 2
CSV_COMMENT = "# "
CSV_COMMENT_MARK = "#"  # on reading, the space after # is optional
VERSION_KEY = "panda_version"  # PANDA version that wrote the file
CSV_METADATA_FIELDS = ("frame", "particle_count", "pore_volume_nm3", "flipped", "frame_count")
CSV_COLUMNS = ("coordinate_nm", "density_per_nm3", "density_sem_per_nm3")  # column order when there is no header


class DensityProfile(PandaModel):
    """Density profile of the analysed liquid along the profile axis of the pore."""

    coordinate_nm: Array1D  # bin centres, nm, in the frame of reference `frame`
    density_per_nm3: Array1D  # number of particles per nm³
    density_sem_per_nm3: Array1D | None = None  # standard error of the mean over blocks
    frame: Frame = Frame.PORE
    particle_count: float | None = Field(default=None, gt=0.0)  # N — particles in the pore
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)  # V — pore volume to which φ refers
    flipped: bool = False  # profile reflected z → −z (the liquid was at the upper wall)
    frame_count: int | None = Field(default=None, ge=1)  # number of frames averaged

    @model_validator(mode="after")
    def _check_arrays(self) -> "DensityProfile":
        if self.coordinate_nm.shape != self.density_per_nm3.shape:
            raise ValueError("coordinate and density have different lengths")
        if self.density_sem_per_nm3 is not None and self.density_sem_per_nm3.shape != self.density_per_nm3.shape:
            raise ValueError("density and its error have different lengths")
        if self.coordinate_nm.size < MIN_PROFILE_POINTS:
            raise ValueError(f"profile has fewer than {MIN_PROFILE_POINTS} points")
        if not np.all(np.diff(self.coordinate_nm) > 0):
            raise ValueError("profile coordinate is not increasing")
        if np.any(self.density_per_nm3 < 0):
            raise ValueError("density is negative")
        if not np.any(self.density_per_nm3 > 0):
            raise ValueError("profile is empty: the density is zero everywhere; check the atom selection")
        return self

    def mean_density_per_nm3(self) -> float:
        """n̄ = N / V — mean density of the liquid in the pore."""
        if self.particle_count is None or self.pore_volume_nm3 is None:
            raise InvalidProfileError(
                "mean normalization needs particle_count and pore_volume_nm3; set both in the profile"
            )
        return self.particle_count / self.pore_volume_nm3

    def save(self, path: str | Path) -> None:
        """Save: .npz — everything; .csv — columns and metadata in comment lines."""
        target = Path(path)
        if target.suffix == ".npz":
            save_npz_arrays(target, _profile_arrays(self), self._metadata())
        elif target.suffix == ".csv":
            metadata = {**self._metadata(), VERSION_KEY: library_version()}
            header = "\n".join(f"{key}: {json.dumps(value)}" for key, value in metadata.items())
            arrays = _profile_arrays(self)
            columns = list(arrays)
            table = np.column_stack([arrays[name] for name in columns])
            np.savetxt(target, table, delimiter=",", header=header + "\n" + ",".join(columns), comments=CSV_COMMENT)
        else:
            raise ValueError(f"unsupported format {target.suffix!r}; available: .npz, .csv")

    @classmethod
    def load(cls, path: str | Path) -> "DensityProfile":
        source = Path(path)
        if source.suffix == ".npz":
            arrays, metadata = load_npz_arrays(source)
        elif source.suffix == ".csv":
            metadata, arrays = _read_csv(source)
        else:
            raise ValueError(f"unsupported format {source.suffix!r}; available: .npz, .csv")
        return cls(**arrays, **metadata)

    def _metadata(self) -> dict[str, object]:
        dumped = self.model_dump(mode="json")
        return {name: dumped[name] for name in CSV_METADATA_FIELDS}


def load_npz_arrays(path: Path) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Arrays and metadata saved by save_npz_arrays."""
    with np.load(path) as archive:
        metadata: dict[str, Any] = json.loads(str(archive["metadata"]))
        arrays = {name: archive[name] for name in archive.files if name != "metadata"}
    metadata.pop(
        VERSION_KEY, None
    )  # the version that wrote the file is for reference only; it does not go into the model
    return arrays, metadata


def save_npz_arrays(path: Path, arrays: Mapping[str, FloatArray], metadata: Mapping[str, object]) -> None:
    """Arrays by name, metadata as a JSON string under the key metadata."""
    # Any: numpy stubs type **kwds of savez as bool (allow_pickle); the values are arrays and a string
    payload: dict[str, Any] = {**arrays, "metadata": json.dumps({**metadata, VERSION_KEY: library_version()})}
    np.savez(path, **payload)


def _require_npz(path: Path) -> None:
    """A ProfileSet is stored only in .npz: otherwise numpy silently appends the suffix and load by that name fails."""
    if path.suffix != ".npz":
        raise ValueError(f"profile set cannot be stored in {path.name!r}; use a .npz file")


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
            if not line.startswith(CSV_COMMENT_MARK):
                break
            content = line[len(CSV_COMMENT_MARK) :].strip()  # '# frame: …' and '#frame: …'
            key, separator, value = content.partition(": ")
            if separator and key == VERSION_KEY:
                continue
            if separator and key in CSV_METADATA_FIELDS:
                metadata[key] = json.loads(value)
            else:
                columns = [name.strip() for name in content.split(",")]
    table = np.loadtxt(source, delimiter=",", comments=CSV_COMMENT_MARK, ndmin=2)
    if columns and set(columns) <= set(CSV_COLUMNS) and len(columns) != table.shape[1]:
        raise InvalidProfileError(
            f"{source.name}: header has {len(columns)} column(s) ({', '.join(columns)}), the data have {table.shape[1]}"
        )
    if not columns or not set(columns) <= set(CSV_COLUMNS):  # no header or a foreign one — columns in order
        if table.shape[1] not in (len(CSV_COLUMNS) - 1, len(CSV_COLUMNS)):
            raise InvalidProfileError(
                f"{source.name}: {table.shape[1]} column(s), expected 2 or 3 (coordinate, nm; density, nm⁻³; "
                "density error); or add the header '# " + ",".join(CSV_COLUMNS) + "'"
            )
        columns = list(CSV_COLUMNS[: table.shape[1]])
    return metadata, {name: table[:, index] for index, name in enumerate(columns)}


class ProfileSet(PandaModel):
    """Profiles over time blocks: blocks × bins. The block mean is the profile for the fit, the spread is its error."""

    bin_edges_nm: Array1D  # bin edges, nm, in the frame of reference `frame`
    block_densities_per_nm3: Array2D  # blocks × bins
    block_start_ps: Array1D
    block_stop_ps: Array1D
    block_frame_counts: tuple[int, ...]
    block_particle_counts: Array1D  # number of particles in the pore (N) averaged over the frames of the block
    frame: Frame
    centering: ProfileCentering
    selection: str
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)
    flipped: bool = False

    @model_validator(mode="after")
    def _check_shapes(self) -> "ProfileSet":
        block_count, bin_count = self.block_densities_per_nm3.shape
        if self.bin_edges_nm.shape != (bin_count + 1,) or not np.all(np.diff(self.bin_edges_nm) > 0):
            raise ValueError("invalid bin edges; there must be one more edge than bins, strictly increasing")
        for name in ("block_start_ps", "block_stop_ps", "block_particle_counts"):
            if getattr(self, name).shape != (block_count,):
                raise ValueError(f"{name} has a wrong length; give one value per block")
        if len(self.block_frame_counts) != block_count or min(self.block_frame_counts, default=0) < 1:
            raise ValueError("block without frames; every block needs at least one frame")
        return self

    @property
    def coordinate_nm(self) -> FloatArray:
        centers: FloatArray = 0.5 * (self.bin_edges_nm[:-1] + self.bin_edges_nm[1:])
        return centers

    @property
    def block_count(self) -> int:
        return int(self.block_densities_per_nm3.shape[0])

    def std(self) -> FloatArray:
        """Standard deviation of the density over blocks (0 for a single block)."""
        if self.block_count < 2:
            return np.zeros(self.block_densities_per_nm3.shape[1])
        spread: FloatArray = np.std(self.block_densities_per_nm3, axis=0, ddof=1)
        return spread

    def sem(self) -> FloatArray:
        """Standard error of the mean over blocks: std / √(number of blocks)."""
        error: FloatArray = self.std() / np.sqrt(self.block_count)
        return error

    def mean(self) -> DensityProfile:
        """Mean profile over blocks with the standard error of the mean."""
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
        """Save to .npz (arrays and metadata)."""
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
        save_npz_arrays(Path(path), arrays, metadata)

    @classmethod
    def load(cls, path: str | Path) -> "ProfileSet":
        _require_npz(Path(path))
        arrays, metadata = load_npz_arrays(Path(path))
        return cls(**arrays, **metadata)


class ReducedProfile(PandaModel):
    """Profile in dimensionless form: coordinate as a fraction of the pore size, density after normalization."""

    coordinate: Array1D
    density: Array1D
    frame: Frame
    normalization: Normalization
    density_scale_per_nm3: float = Field(gt=0.0)  # what the density was divided by: ρ_bulk or n̄


def reduce_profile(
    profile: DensityProfile,
    characteristic_size_nm: float,
    normalization: Normalization,
    density_bulk_per_nm3: float | None,
) -> ReducedProfile:
    if normalization is Normalization.BULK:
        if density_bulk_per_nm3 is None:
            raise InvalidProfileError("bulk normalization needs the bulk density; set density_bulk_per_nm3")
        scale = density_bulk_per_nm3
    elif normalization is Normalization.MEAN:
        scale = profile.mean_density_per_nm3()
    else:
        raise ValueError(f"normalization {normalization.value} must be resolved before the profile is reduced")
    return ReducedProfile(
        coordinate=profile.coordinate_nm / characteristic_size_nm,
        density=profile.density_per_nm3 / scale,
        frame=profile.frame,
        normalization=normalization,
        density_scale_per_nm3=scale,
    )
