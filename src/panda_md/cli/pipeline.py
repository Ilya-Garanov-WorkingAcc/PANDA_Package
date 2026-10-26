"""Pipeline stages of the commands: profile from a trajectory, profile fit, result output. Same calls as the API."""

import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile

from panda_md.analysis.blocks import FitResultSet, fit_blocks
from panda_md.analysis.compare import ShapeComparison, compare_shapes
from panda_md.cli.options import OPTIONS_HINT, CommandLineError, missing_fields, pore_json, require_dimensions
from panda_md.core.enums import Orientation, ParticleGrouping, ProfileCentering
from panda_md.core.errors import UnknownShapeError
from panda_md.core.fitting import fit
from panda_md.core.pore import Pore
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.core.types import PandaModel
from panda_md.trajectory.density import density_profile
from panda_md.trajectory.reader import load_trajectory

AUTO_SHAPE = "auto"  # --shape auto: try all shapes of the pore, take the best
PROFILE_SET_KEY = "bin_edges_nm"  # the array that tells a .npz of a profile set from a single profile
METADATA_KEY = "metadata"  # metadata JSON in a .npz saved by PANDA
WALL_SELECTION_FIELD = "wall_selection"
EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_FIT_PROBLEMS = 2
EXIT_INTERRUPTED = 130  # 128 + SIGINT, as is customary in the shell

Echo = Callable[[str], None]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise CommandLineError(f"file not found: {path}")
    return path


class FitReport(PandaModel):
    """What panda fit -o saves: the profile fit (with --blocks — of the block mean), block fits, shape comparison."""

    fit: FitResult
    blocks: FitResultSet | None = None  # --blocks
    comparison: ShapeComparison | None = None  # --shape auto


def load_profile(path: Path) -> tuple[DensityProfile, ProfileSet | None]:
    """Profile for the fit: for a block set (.npz from panda profile) the block mean, otherwise the profile itself."""
    require_file(path)
    if path.suffix == ".npz" and PROFILE_SET_KEY in _panda_npz_keys(path):
        profiles = ProfileSet.load(path)
        return profiles.mean(), profiles
    return DensityProfile.load(path), None


def _panda_npz_keys(path: Path) -> list[str]:
    """Array names of a .npz saved by PANDA; a foreign or corrupted file gives a clear error, not a KeyError."""
    advice = "profiles are saved by panda profile and DensityProfile.save; pass other data as .csv"
    try:
        loaded = np.load(path)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise CommandLineError(f"{path.name} is not a PANDA profile ({error}); {advice}") from None
    if not isinstance(loaded, NpzFile):
        raise CommandLineError(f"{path.name} is not a PANDA profile: not an .npz archive; {advice}")
    with loaded:
        keys = list(loaded.files)
    if METADATA_KEY not in keys:
        raise CommandLineError(f"{path.name} is not a PANDA profile: no metadata; {advice}")
    return keys


def check_outputs(*paths: Path | None) -> None:
    """Before long work: the directories of the output files exist."""
    for path in paths:
        if path is not None and not path.parent.is_dir():
            raise CommandLineError(f"directory {path.parent} for {path.name} does not exist")


def check_profiles_output(path: Path | None) -> None:
    if path is not None and path.suffix != ".npz":
        raise CommandLineError(f"block profiles are saved to .npz, not to {path.name}")
    check_outputs(path)


def check_plot(path: Path | None) -> None:
    """Before the fit: without matplotlib or with an unknown format the plot cannot be built — fail at once."""
    if path is None:
        return
    from panda_md.plotting import supported_figure_formats  # MissingDependencyError if matplotlib is absent

    formats = supported_figure_formats()
    if path.suffix.lstrip(".").lower() not in formats:
        raise CommandLineError(f"unsupported plot format: {path.name}; available: {', '.join(formats)}")
    check_outputs(path)


def check_shape_name(pore_type: type[Pore], shape: str) -> None:
    """Before reading the data: a typo in the shape name is an error at once, with the list of pore shapes."""
    registry = pore_type.own_registry()
    if shape != AUTO_SHAPE and shape not in registry:
        raise UnknownShapeError(
            f"unknown shape '{shape}' in pore '{pore_type.name}'; available: {AUTO_SHAPE}, {', '.join(registry)}"
        )


def trajectory_pore(
    pore_type: type[Pore],
    values: Mapping[str, object],
    trajectory: Trajectory,
    *,
    select: str,
    start: TimeValue,
    hint: str = OPTIONS_HINT,
) -> Pore:
    """Pore with the given dimensions; without them measured from the wall atoms in frame start, if the pore type can
    be measured (Pore.trajectory_fields, Pore.measure_from_trajectory)."""
    missing = missing_fields(pore_type, values)
    required = missing_fields(pore_type, {})
    if not missing or not pore_type.trajectory_fields or missing != required:
        return require_dimensions(pore_type, values, hint)
    walls = values.get(WALL_SELECTION_FIELD)
    unused = sorted(set(values) - set(pore_type.trajectory_fields))
    if walls is None:
        raise CommandLineError(
            f"{pore_type.name} pore dimensions are measured from the trajectory by the wall atoms: set their "
            f"selection (--walls “selection” or wall_selection in [pore]) — wall coordinates “z1,z2” do not measure "
            f"the pore; or set the dimensions with {hint}"
        )
    if unused:
        raise CommandLineError(
            f"without dimensions the {pore_type.name} pore is measured from the trajectory, and fields "
            f"{', '.join(unused)} are unused; remove them or set all dimensions ({hint})"
        )
    return pore_type.measure_from_trajectory(trajectory, walls=str(walls), time=start, liquid=select, fields=values)


def build_profiles(
    pore_type: type[Pore],
    pore_values: Mapping[str, object],
    topology: Path,
    trajectory_files: Sequence[Path],
    select: str,
    *,
    centering: ProfileCentering | None,
    pore_center_nm: float | None,
    bins: int,
    bin_width_nm: float | None,
    start: TimeValue,
    stop: TimeValue | None,
    block: TimeValue | None,
    stride: int,
    per: ParticleGrouping,
    orientation: Orientation,
    output: Path | None,
    pore_output: Path | None,
    echo: Echo,
    hint: str = OPTIONS_HINT,
) -> tuple[Pore, ProfileSet]:
    """Block profiles; the pore has the given dimensions or is measured from the trajectory (JSON to pore_output)."""
    for path in (topology, *trajectory_files):
        require_file(path)
    trajectory = load_trajectory(topology, *trajectory_files)
    measured = bool(missing_fields(pore_type, pore_values))
    pore = trajectory_pore(pore_type, pore_values, trajectory, select=select, start=start, hint=hint)
    if measured:
        echo(f"pore from the trajectory: {pore_json(pore, indent=None)}")
    profiles = density_profile(
        trajectory,
        pore,
        select,
        centering=centering,
        bins=bins,
        bin_width_nm=bin_width_nm,
        start=start,
        stop=stop,
        block=block,
        stride=stride,
        pore_center_nm=pore_center_nm,
        orientation=orientation,
        per=per,
    )
    frames = sum(profiles.block_frame_counts)
    echo(f"profiles: {profiles.block_count} block(s), {frames} frames, {profiles.coordinate_nm.size} bins")
    if output is not None:
        profiles.save(output)
        echo(f"profiles → {output}")
    if pore_output is not None:
        pore_output.write_text(pore_json(pore) + "\n", encoding="utf-8")
        echo(f"pore → {pore_output}")
    return pore, profiles


def run_fit(
    pore: Pore,
    profile: DensityProfile,
    profiles: ProfileSet | None,
    shape: str,
    spec: FitSpec,
    *,
    blocks: bool,
    output: Path | None,
    plot: Path | None,
    echo: Echo,
) -> int:
    """Fit with one shape or by search (shape = auto); blocks — also each block. Exit code 0 or 2."""
    if blocks and profiles is None:
        raise CommandLineError("fit over blocks needs a set of block profiles; pass the .npz from panda profile")
    check_plot(plot)
    check_outputs(output)
    comparison = None
    if shape == AUTO_SHAPE:
        comparison = compare_shapes(pore, profile, spec)
        echo(comparison.summary())
        result = comparison.best
        echo(f"best shape: {result.shape_name}")
    else:
        result = fit(pore.interface(shape), profile, spec)
    echo(result.summary())
    problems = list(result.problems())
    result_set = None
    if blocks:
        assert profiles is not None  # checked above
        result_set = fit_blocks(pore.interface(result.shape_name), profiles, spec)
        echo(result_set.summary())
        problems.extend(problem for block_result in result_set.results for problem in block_result.problems())
    if output is not None:
        report = FitReport(fit=result, blocks=result_set, comparison=comparison)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        echo(f"result → {output}")
    if plot is not None:
        _save_plot(result, plot)
        echo(f"plot → {plot}")
    return EXIT_FIT_PROBLEMS if problems else EXIT_SUCCESS


def _save_plot(result: FitResult, path: Path) -> None:
    from panda_md.plotting import save_fit_figure  # matplotlib only in panda_md.plotting

    save_fit_figure(result, path)
