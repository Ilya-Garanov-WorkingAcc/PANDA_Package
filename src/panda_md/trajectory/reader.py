"""Reading MD files through MDAnalysis (optional dependency: `pip install panda-md[md]`).

MDAnalysis stores lengths in ångström; nm are returned. The only module of the package that imports MDAnalysis.
"""

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError, PandaWarning
from panda_md.core.trajectory import Trajectory, TrajectoryFrame, check_box, renumber
from panda_md.core.types import FloatArray, IndexArray

ANGSTROM_PER_NM = 10.0
RECTANGULAR_ANGLE_DEG = 90.0
ANGLE_TOLERANCE_DEG = 1e-3
LAMMPS_DUMP_SUFFIX = ".lammpstrj"
LAMMPS_DUMP_FORMAT = "LAMMPSDUMP"
LAMMPS_DATA_SUFFIX = ".data"
LAMMPS_ATOMIC_STYLE = "id type x y z"  # columns of the Atoms section for atom_style atomic
TIME_MISMATCH_FRACTION = 0.01  # the last frame time differs from t0 + dt·(n − 1) by more than 1 % of the step


def _import_mdanalysis() -> Any:  # noqa: ANN401 — MDAnalysis has no type annotations
    try:
        import MDAnalysis
    except ImportError as error:
        raise MissingDependencyError("reading MD files needs MDAnalysis; run pip install 'panda-md[md]'") from error
    return MDAnalysis


class MDATrajectory(Trajectory):
    """Trajectory from an MDAnalysis.Universe object."""

    universe: Any  # Any: MDAnalysis.Universe has no type annotations

    def __init__(self, universe: Any) -> None:  # noqa: ANN401 — MDAnalysis.Universe has no type annotations
        self.universe = universe
        self._times: FloatArray | None = None

    def __repr__(self) -> str:
        return f"MDATrajectory({self.universe.atoms.n_atoms} atoms, {len(self.universe.trajectory)} frames)"

    def select(self, selection: str | Sequence[int]) -> IndexArray:
        if isinstance(selection, str):
            try:
                indices = np.asarray(self.universe.select_atoms(selection).indices, dtype=np.int64)
            except Exception as error:  # MDAnalysis raises various exceptions on selection syntax
                raise InvalidTrajectoryError(f"MDAnalysis cannot parse selection {selection!r}: {error}") from error
        else:
            indices = np.asarray(selection, dtype=np.int64)
            atom_count = int(self.universe.atoms.n_atoms)
            if indices.size and (int(np.min(indices)) < 0 or int(np.max(indices)) >= atom_count):
                raise InvalidTrajectoryError(f"selection indices outside [0, {atom_count})")
        if indices.size == 0:
            raise InvalidTrajectoryError(f"selection {selection!r} is empty; check the residue and atom names")
        return indices

    def molecules(self, indices: IndexArray) -> tuple[IndexArray, FloatArray]:
        """A molecule is the GROMACS molecule number (present in .tpr), otherwise the residue. Bonded fragments
        do not work: the bond-less virtual site of TIP4P water would become a separate “molecule”."""
        atoms = self.universe.atoms[indices]
        masses = np.asarray(atoms.masses, dtype=np.float64)
        labels = atoms.molnums if hasattr(self.universe.atoms, "molnums") else atoms.resindices
        return renumber(np.asarray(labels, dtype=np.int64)), masses

    def times_ps(self) -> FloatArray:
        """Frame times: t0 + Δt·i (Δt between the first two frames) if the last frame time agrees with it;
        otherwise (concatenated trajectories with gaps or overlaps) the actual time of each frame, read once."""
        if self._times is not None:
            return self._times
        reader = self.universe.trajectory
        first = float(reader[0].time)
        spacing = float(reader[1].time) - first if len(reader) > 1 else float(reader.dt)
        times: FloatArray = first + spacing * np.arange(len(reader), dtype=np.float64)
        if abs(float(reader[len(reader) - 1].time) - float(times[-1])) > TIME_MISMATCH_FRACTION * abs(spacing):
            times = np.array([float(timestep.time) for timestep in reader], dtype=np.float64)
            warnings.warn(
                f"frame times do not match t0 + step · i (step {spacing:g} ps); the times are read from the "
                f"trajectory",
                PandaWarning,
                stacklevel=2,
            )
        self._times = times
        return times

    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        for timestep in self.universe.trajectory[list(frame_indices)]:
            dimensions = timestep.dimensions
            if dimensions is None or np.any(np.abs(dimensions[3:] - RECTANGULAR_ANGLE_DEG) > ANGLE_TOLERANCE_DEG):
                raise InvalidTrajectoryError(f"box of frame {timestep.frame} is not rectangular: {dimensions}")
            box_nm = np.asarray(dimensions[:3], dtype=np.float64) / ANGSTROM_PER_NM
            check_box(box_nm)
            yield TrajectoryFrame(
                index=int(timestep.frame),
                time_ps=float(timestep.time),
                box_nm=box_nm,
                positions_nm=tuple(
                    np.asarray(timestep.positions[group], dtype=np.float64) / ANGSTROM_PER_NM for group in groups
                ),
            )


def load_trajectory(topology: str | Path, *trajectory_files: str | Path, **options: object) -> MDATrajectory:
    """Open an MD trajectory: topology (.tpr, .gro, .pdb, LAMMPS .data, …) and trajectory files (.xtc, .trr, …).
    `options` go to MDAnalysis.Universe (e.g. `topology_format`, `format`, `atom_style`, `dt`). A LAMMPS dump
    `.lammpstrj` is read as LAMMPSDUMP; a LAMMPS data file of `atom_style atomic` opens without `atom_style`. The time
    of a LAMMPS frame is its step number unless `dt` (time per step) is given.
    """
    mdanalysis = _import_mdanalysis()
    paths = [str(path) for path in (topology, *trajectory_files)]
    for path in paths:
        if not Path(path).exists():
            raise InvalidTrajectoryError(f"file not found: {path}")
    lammps_dumps = all(Path(path).suffix.lower() == LAMMPS_DUMP_SUFFIX for path in trajectory_files)
    if trajectory_files and lammps_dumps and "format" not in options:
        options["format"] = LAMMPS_DUMP_FORMAT
    try:
        return MDATrajectory(mdanalysis.Universe(*paths, **options))
    except Exception as error:  # MDAnalysis raises various exceptions for files it cannot parse
        first_error = error
    if Path(paths[0]).suffix.lower() == LAMMPS_DATA_SUFFIX and "atom_style" not in options:
        try:
            return MDATrajectory(mdanalysis.Universe(*paths, atom_style=LAMMPS_ATOMIC_STYLE, **options))
        except Exception:  # report the first error: it names the original problem
            pass
    raise InvalidTrajectoryError(_open_error_message(paths, first_error, mdanalysis.__version__)) from first_error


def _open_error_message(paths: Sequence[str], error: Exception, version: str) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    reason = (lines[-1] if lines else type(error).__name__).removeprefix("Error: ")
    suffix = Path(paths[0]).suffix.lower()
    if suffix == ".tpr":
        hint = (
            f"the .tpr may come from a GROMACS version newer than MDAnalysis {version} reads; "
            "use the .gro file of the run as the topology (molecules are then the residues)"
        )
    elif suffix == LAMMPS_DATA_SUFFIX:
        hint = "for a LAMMPS data file pass atom_style, e.g. atom_style='id type x y z'"
    else:
        hint = "check the file formats; MDAnalysis options can be passed, e.g. topology_format=…, format=…"
    return f"MDAnalysis cannot open {', '.join(paths)}: {reason}; {hint}"
