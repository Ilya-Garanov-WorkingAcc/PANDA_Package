"""Этапы конвейера для команд: профиль из траектории, фит профиля, вывод результата. Те же вызовы, что в API."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from panda_md.analysis.blocks import FitResultSet, fit_blocks
from panda_md.analysis.compare import compare_shapes
from panda_md.cli.options import CommandLineError, missing_fields, pore_json, require_dimensions
from panda_md.core.enums import Axis, Orientation, ParticleGrouping, ProfileCentering
from panda_md.core.fitting import fit
from panda_md.core.pore import Pore
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.pores.slit import SlitPore
from panda_md.trajectory.density import density_profile
from panda_md.trajectory.reader import load_trajectory

AUTO_SHAPE = "auto"  # --shape auto: перебор всех форм поры, берётся лучшая
PROFILE_SET_KEY = "bin_edges_nm"  # массив, по которому .npz набора профилей отличается от одного профиля
WALL_SELECTION_FIELD = "wall_selection"
# поля щели, которые задаются и без размеров: остальное SlitPore.from_trajectory находит сам
FROM_TRAJECTORY_FIELDS = (WALL_SELECTION_FIELD, "length_axis", "normal_axis", "wall_offset_nm")
EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_FIT_PROBLEMS = 2

Echo = Callable[[str], None]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise CommandLineError(f"файл {path} не найден")
    return path


def load_profile(path: Path) -> tuple[DensityProfile, ProfileSet | None]:
    """Профиль для фита: из набора по блокам (.npz от panda profile) — среднее по блокам; иначе сам профиль."""
    require_file(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            is_profile_set = PROFILE_SET_KEY in archive.files
        if is_profile_set:
            profiles = ProfileSet.load(path)
            return profiles.mean(), profiles
    return DensityProfile.load(path), None


def trajectory_pore(
    pore_type: type[Pore], values: Mapping[str, object], trajectory: Trajectory, *, select: str, start: TimeValue
) -> Pore:
    """Пора с заданными размерами; без размеров — щель по атомам стенок в кадре start (как SlitPore.from_trajectory)."""
    missing = missing_fields(pore_type, values)
    required = missing_fields(pore_type, {})
    if not missing or not issubclass(pore_type, SlitPore) or missing != required:
        return require_dimensions(pore_type, values)
    walls = values.get(WALL_SELECTION_FIELD)
    unused = sorted(set(values) - set(FROM_TRAJECTORY_FIELDS))
    if walls is None or unused:
        raise CommandLineError(
            "размеры щели по траектории ищутся по атомам стенок: задай --walls «выборка»"
            + (f"; без размеров не используются: {', '.join(unused)}" if unused else "")
            + "; или задай размеры: --pore-file / -g"
        )
    return SlitPore.from_trajectory(
        trajectory,
        walls=str(walls),
        length_axis=Axis(values.get("length_axis", Axis.X)),
        normal_axis=Axis(values.get("normal_axis", Axis.Z)),
        wall_offset_nm=float(str(values.get("wall_offset_nm", 0.0))),
        time=start,
        liquid=select,
    )


def build_profiles(
    pore_type: type[Pore],
    pore_values: Mapping[str, object],
    topology: Path,
    trajectory_files: Sequence[Path],
    select: str,
    *,
    centering: ProfileCentering,
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
) -> tuple[Pore, ProfileSet]:
    """Профили по блокам; пора — с заданными размерами или найденная по траектории (её JSON — в pore_output)."""
    for path in (topology, *trajectory_files):
        require_file(path)
    trajectory = load_trajectory(topology, *trajectory_files)
    measured = bool(missing_fields(pore_type, pore_values))
    pore = trajectory_pore(pore_type, pore_values, trajectory, select=select, start=start)
    if measured:
        echo(f"пора по траектории: {pore_json(pore, indent=None)}")
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
    target = f" → {output}" if output is not None else ""
    echo(f"профили: {profiles.block_count} блок(ов), {frames} кадров, {profiles.coordinate_nm.size} бинов{target}")
    if output is not None:
        profiles.save(output)
    if pore_output is not None:
        pore_output.write_text(pore_json(pore) + "\n", encoding="utf-8")
        echo(f"пора → {pore_output}")
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
    """Фит одной формой или перебором (shape = auto); blocks — ещё и каждого блока. Код возврата 0 или 2."""
    if blocks and profiles is None:
        raise CommandLineError("фит по блокам: нужен набор профилей по блокам (.npz от panda profile)")
    if plot is not None:
        _check_plotting()
    if shape == AUTO_SHAPE:
        comparison = compare_shapes(pore, profile, spec)
        echo(comparison.summary())
        result = comparison.best
        echo(f"лучшая форма: {result.shape_name}")
    else:
        result = fit(pore.interface(shape), profile, spec)
    echo(result.summary())
    problems = list(result.problems())
    saved: FitResult | FitResultSet = result
    if blocks:
        assert profiles is not None  # проверено выше
        result_set = fit_blocks(pore.interface(result.shape_name), profiles, spec)
        echo(result_set.summary())
        problems.extend(problem for block_result in result_set.results for problem in block_result.problems())
        saved = result_set
    if output is not None:
        output.write_text(saved.model_dump_json(indent=2) + "\n", encoding="utf-8")
        echo(f"результат → {output}")
    if plot is not None:
        _save_plot(result, plot)
        echo(f"график → {plot}")
    return EXIT_FIT_PROBLEMS if problems else EXIT_SUCCESS


def _check_plotting() -> None:
    """До фита: без matplotlib график не построить — понятная ошибка сразу, а не после долгого перебора."""
    import panda_md.plotting  # noqa: F401 — MissingDependencyError, если matplotlib нет


def _save_plot(result: FitResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # только файл, без окна
    import matplotlib.pyplot as plt

    from panda_md.plotting import plot_fit

    figure, axes = plt.subplots()
    plot_fit(result, axes)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
