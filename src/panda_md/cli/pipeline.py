"""Этапы конвейера для команд: профиль из траектории, фит профиля, вывод результата. Те же вызовы, что в API."""

import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile

from panda_md.analysis.blocks import FitResultSet, fit_blocks
from panda_md.analysis.compare import ShapeComparison, compare_shapes
from panda_md.cli.options import OPTIONS_HINT, CommandLineError, missing_fields, pore_json, require_dimensions
from panda_md.core.enums import Axis, Orientation, ParticleGrouping, ProfileCentering
from panda_md.core.errors import UnknownShapeError
from panda_md.core.fitting import fit
from panda_md.core.pore import Pore
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.core.types import PandaModel
from panda_md.pores.slit import SlitPore
from panda_md.trajectory.density import density_profile
from panda_md.trajectory.reader import load_trajectory

AUTO_SHAPE = "auto"  # --shape auto: перебор всех форм поры, берётся лучшая
PROFILE_SET_KEY = "bin_edges_nm"  # массив, по которому .npz набора профилей отличается от одного профиля
METADATA_KEY = "metadata"  # JSON метаданных в .npz, сохранённом PANDA
WALL_SELECTION_FIELD = "wall_selection"
# поля щели, которые задаются и без размеров: остальное SlitPore.from_trajectory находит сам
FROM_TRAJECTORY_FIELDS = (WALL_SELECTION_FIELD, "length_axis", "normal_axis", "wall_offset_nm")
EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_FIT_PROBLEMS = 2
EXIT_INTERRUPTED = 130  # 128 + SIGINT, как принято в оболочке

Echo = Callable[[str], None]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise CommandLineError(f"файл {path} не найден")
    return path


class FitReport(PandaModel):
    """Что сохраняет panda fit -o: фит профиля (с --blocks — среднего по блокам), фиты блоков, сравнение форм."""

    fit: FitResult
    blocks: FitResultSet | None = None  # --blocks
    comparison: ShapeComparison | None = None  # --shape auto


def load_profile(path: Path) -> tuple[DensityProfile, ProfileSet | None]:
    """Профиль для фита: из набора по блокам (.npz от panda profile) — среднее по блокам; иначе сам профиль."""
    require_file(path)
    if path.suffix == ".npz" and PROFILE_SET_KEY in _panda_npz_keys(path):
        profiles = ProfileSet.load(path)
        return profiles.mean(), profiles
    return DensityProfile.load(path), None


def _panda_npz_keys(path: Path) -> list[str]:
    """Имена массивов .npz, сохранённого PANDA; чужой или испорченный файл — понятная ошибка, а не KeyError."""
    advice = "профиль сохраняют panda profile и DensityProfile.save; чужие данные передай в .csv"
    try:
        loaded = np.load(path)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        raise CommandLineError(f"{path.name}: не профиль PANDA ({error}); {advice}") from None
    if not isinstance(loaded, NpzFile):
        raise CommandLineError(f"{path.name}: не профиль PANDA — это не архив .npz; {advice}")
    with loaded:
        keys = list(loaded.files)
    if METADATA_KEY not in keys:
        raise CommandLineError(f"{path.name}: не профиль PANDA — нет metadata; {advice}")
    return keys


def check_outputs(*paths: Path | None) -> None:
    """До долгой работы: каталоги выходных файлов существуют."""
    for path in paths:
        if path is not None and not path.parent.is_dir():
            raise CommandLineError(f"каталог {path.parent} для {path.name} не существует")


def check_profiles_output(path: Path | None) -> None:
    if path is not None and path.suffix != ".npz":
        raise CommandLineError(f"профили по блокам сохраняются в .npz, а не в {path.name}")
    check_outputs(path)


def check_plot(path: Path | None) -> None:
    """До фита: без matplotlib или с неизвестным форматом график не построить — ошибка сразу."""
    if path is None:
        return
    from panda_md.plotting import supported_figure_formats  # MissingDependencyError, если matplotlib нет

    formats = supported_figure_formats()
    if path.suffix.lstrip(".").lower() not in formats:
        raise CommandLineError(f"график {path.name}: формат не поддерживается; доступны: {', '.join(formats)}")
    check_outputs(path)


def check_shape_name(pore_type: type[Pore], shape: str) -> None:
    """До чтения данных: опечатка в имени формы — сразу ошибка со списком форм поры."""
    registry = pore_type.own_registry()
    if shape != AUTO_SHAPE and shape not in registry:
        raise UnknownShapeError(
            f"в поре {pore_type.name} нет формы {shape}; доступны: {AUTO_SHAPE}, {', '.join(registry)}"
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
    """Пора с заданными размерами; без размеров — щель по атомам стенок в кадре start (как SlitPore.from_trajectory)."""
    missing = missing_fields(pore_type, values)
    required = missing_fields(pore_type, {})
    if not missing or not issubclass(pore_type, SlitPore) or missing != required:
        return require_dimensions(pore_type, values, hint)
    walls = values.get(WALL_SELECTION_FIELD)
    unused = sorted(set(values) - set(FROM_TRAJECTORY_FIELDS))
    if walls is None:
        raise CommandLineError(
            "размеры щели по траектории ищутся по атомам стенок: задай их выборку (--walls «выборка» или "
            f"wall_selection в [pore]) — координаты стенок «z1,z2» щель не измеряют; или задай размеры: {hint}"
        )
    if unused:
        raise CommandLineError(
            f"без размеров щель измеряется по траектории, и поля {', '.join(unused)} не используются: "
            f"убери их или задай все размеры ({hint})"
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
    """Профили по блокам; пора — с заданными размерами или найденная по траектории (её JSON — в pore_output)."""
    for path in (topology, *trajectory_files):
        require_file(path)
    trajectory = load_trajectory(topology, *trajectory_files)
    measured = bool(missing_fields(pore_type, pore_values))
    pore = trajectory_pore(pore_type, pore_values, trajectory, select=select, start=start, hint=hint)
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
    echo(f"профили: {profiles.block_count} блок(ов), {frames} кадров, {profiles.coordinate_nm.size} бинов")
    if output is not None:
        profiles.save(output)
        echo(f"профили → {output}")
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
    check_plot(plot)
    check_outputs(output)
    comparison = None
    if shape == AUTO_SHAPE:
        comparison = compare_shapes(pore, profile, spec)
        echo(comparison.summary())
        result = comparison.best
        echo(f"лучшая форма: {result.shape_name}")
    else:
        result = fit(pore.interface(shape), profile, spec)
    echo(result.summary())
    problems = list(result.problems())
    result_set = None
    if blocks:
        assert profiles is not None  # проверено выше
        result_set = fit_blocks(pore.interface(result.shape_name), profiles, spec)
        echo(result_set.summary())
        problems.extend(problem for block_result in result_set.results for problem in block_result.problems())
    if output is not None:
        report = FitReport(fit=result, blocks=result_set, comparison=comparison)
        output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        echo(f"результат → {output}")
    if plot is not None:
        _save_plot(result, plot)
        echo(f"график → {plot}")
    return EXIT_FIT_PROBLEMS if problems else EXIT_SUCCESS


def _save_plot(result: FitResult, path: Path) -> None:
    from panda_md.plotting import save_fit_figure  # matplotlib — только в panda_md.plotting

    save_fit_figure(result, path)
