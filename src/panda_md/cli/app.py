"""Команды `panda`: shapes, profile, fit, run. Тонкая обёртка над API; коды возврата: 0 — успех, 1 — ошибка
входа, 2 — фит завершён с флагами проблем."""

import sys
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated, TextIO

import typer
from pydantic import ValidationError

from panda_md.cli.config import load_config
from panda_md.cli.options import (
    CONFIG_HINT,
    DEFAULT_PORE,
    PORE_TYPE_KEY,
    CommandLineError,
    load_pore,
    parse_bounds,
    parse_numbers,
    parse_walls,
    pore_request,
    require_dimensions,
    split_names,
)
from panda_md.cli.pipeline import (
    AUTO_SHAPE,
    EXIT_INPUT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_SUCCESS,
    WALL_SELECTION_FIELD,
    build_profiles,
    check_outputs,
    check_plot,
    check_profiles_output,
    check_shape_name,
    load_profile,
    require_file,
    run_fit,
)
from panda_md.core.enums import (
    LossFunction,
    Normalization,
    Orientation,
    ParticleGrouping,
    ProfileCentering,
    Regime,
)
from panda_md.core.errors import PandaError
from panda_md.core.parameters import PROFILE_SHIFT
from panda_md.core.pore import Pore
from panda_md.core.spec import FitSpec
from panda_md.pores.registry import pore_class, pore_types
from panda_md.trajectory.density import DEFAULT_BIN_COUNT

app = typer.Typer(
    name="panda",
    help="PANDA: краевой угол и форма интерфейса жидкости в поре по профилю плотности МД.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,  # иначе rich съедает «[md]» в «panda-md[md]»
)


class RegimeChoice(str, Enum):
    AUTO = "auto"  # все режимы, объявленные формой
    WETTING = Regime.WETTING.value
    NON_WETTING = Regime.NON_WETTING.value

    def regime(self) -> Regime | None:
        return None if self is RegimeChoice.AUTO else Regime(self.value)


PoreNameOption = Annotated[
    str | None, typer.Option("--pore", help="тип поры (panda shapes); по умолчанию slit или type из --pore-file")
]
PoreFileOption = Annotated[Path | None, typer.Option("--pore-file", help="JSON с полями поры")]
GeometryOption = Annotated[
    list[str] | None, typer.Option("-g", "--geometry", help="поле поры: имя=значение (width_nm=8), можно несколько")
]


def _echo(text: str) -> None:
    typer.echo(text)


@app.command()
def shapes(pore: Annotated[str | None, typer.Option("--pore", help="показать только этот тип поры")] = None) -> int:
    """Типы пор, их поля и формы интерфейса: параметры, границы, режимы, ось симметрии."""
    selected = {pore: pore_class(pore)} if pore is not None else pore_types()
    for name, pore_type in selected.items():
        _echo(_describe_pore(name, pore_type))
    return EXIT_SUCCESS


@app.command()
def profile(
    topology: Annotated[Path, typer.Argument(help="топология (.gro, .tpr, .pdb, …)")],
    trajectory: Annotated[list[Path], typer.Argument(help="файлы траектории (.xtc, .trr, …), по порядку")],
    select: Annotated[str, typer.Option("--select", help="выборка атомов жидкости (синтаксис MDAnalysis)")],
    output: Annotated[Path, typer.Option("-o", "--output", help="профили по блокам (.npz)")],
    pore_name: PoreNameOption = None,
    pore_file: PoreFileOption = None,
    geometry: GeometryOption = None,
    walls: Annotated[
        str | None, typer.Option("--walls", help="выборка атомов стенок или координаты стенок «z1,z2» (нм)")
    ] = None,
    center_by: Annotated[ProfileCentering, typer.Option("--center-by", help="что поставить в 0")] = (
        ProfileCentering.PORE
    ),
    bins: Annotated[
        int | None, typer.Option("--bins", min=1, help=f"число бинов (по умолчанию {DEFAULT_BIN_COUNT})")
    ] = None,
    bin_width: Annotated[float | None, typer.Option("--bin-width", help="ширина бина, нм (вместо --bins)")] = None,
    start: Annotated[str, typer.Option("--start", help="начало: пс или «2 ns»")] = "0",
    stop: Annotated[str | None, typer.Option("--stop", help="конец: пс или «10 ns»")] = None,
    block: Annotated[str | None, typer.Option("--block", help="длина блока: пс или «1 ns»")] = None,
    stride: Annotated[int, typer.Option("--stride", min=1, help="каждый n-й кадр")] = 1,
    per: Annotated[ParticleGrouping, typer.Option("--per", help="частица: атом или центр масс молекулы")] = (
        ParticleGrouping.ATOM
    ),
    orientation: Annotated[Orientation, typer.Option("--orientation", help="отражать ли профиль z → −z")] = (
        Orientation.AUTO
    ),
    pore_output: Annotated[
        Path | None, typer.Option("--pore-output", help="сохранить пору в JSON (для panda fit --pore-file)")
    ] = None,
) -> int:
    """Профили плотности по блокам времени из траектории МД (нужен panda-md[md]).

    Без размеров поры (--pore-file, -g) щель измеряется по атомам стенок (--walls «выборка») в кадре --start;
    найденную пору сохраняет --pore-output — её JSON потом передаётся в panda fit --pore-file.
    """
    if bins is not None and bin_width is not None:
        raise CommandLineError("--bins и --bin-width взаимоисключающие")
    if bin_width is not None and bin_width <= 0.0:
        raise CommandLineError(f"--bin-width {bin_width:g}: ширина бина должна быть положительной")
    check_profiles_output(output)
    check_outputs(pore_output)
    pore_type, pore_values = pore_request(pore_name, pore_file, geometry or [])
    pore_center_nm = None
    if walls is not None:
        wall_selection, pore_center_nm = parse_walls(walls)
        if wall_selection is not None:
            pore_values[WALL_SELECTION_FIELD] = wall_selection
    build_profiles(
        pore_type,
        pore_values,
        topology,
        trajectory,
        select,
        centering=center_by,
        pore_center_nm=pore_center_nm,
        bins=bins or DEFAULT_BIN_COUNT,
        bin_width_nm=bin_width,
        start=start,
        stop=stop,
        block=block,
        stride=stride,
        per=per,
        orientation=orientation,
        output=output,
        pore_output=pore_output,
        echo=_echo,
    )
    return EXIT_SUCCESS


@app.command("fit")
def fit_command(
    profile_path: Annotated[Path, typer.Argument(metavar="PROFILE", help="профиль .npz/.csv или профили по блокам")],
    pore_name: PoreNameOption = None,
    pore_file: PoreFileOption = None,
    geometry: GeometryOption = None,
    shape: Annotated[str, typer.Option("--shape", help="форма (panda shapes) или auto — перебор всех")] = AUTO_SHAPE,
    optimize: Annotated[
        list[str] | None, typer.Option("--optimize", help="свободные параметры: theta,delta или несколько раз")
    ] = None,
    fix: Annotated[list[str] | None, typer.Option("--fix", help="фиксированный параметр: имя=значение")] = None,
    bounds: Annotated[
        list[str] | None, typer.Option("--bounds", help="границы на этот фит: имя=нижняя:верхняя")
    ] = None,
    rho_bulk: Annotated[
        float | None, typer.Option("--rho-bulk", help="объёмная плотность жидкости, частиц/нм³")
    ] = None,
    regime: Annotated[RegimeChoice, typer.Option("--regime", help="режим смачивания")] = RegimeChoice.AUTO,
    normalization: Annotated[Normalization, typer.Option("--normalization", help="нормировка данных")] = (
        Normalization.AUTO
    ),
    loss: Annotated[LossFunction, typer.Option("--loss", help="функция ошибки")] = LossFunction.L1,
    blocks: Annotated[bool, typer.Option("--blocks", help="ещё и фит каждого блока (разброс по блокам)")] = False,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="результат (.json)")] = None,
    plot: Annotated[Path | None, typer.Option("--plot", help="график фита (.png, .pdf; нужен panda-md[plot])")] = None,
) -> int:
    """Фит профиля формой интерфейса; сводка в терминал, результат в JSON."""
    pore = load_pore(pore_name, pore_file, geometry or [])
    free = split_names(optimize) if optimize else None
    if free is not None and not free:
        raise CommandLineError("--optimize: не назван ни один параметр")
    spec = FitSpec(
        optimize=free,
        fixed=parse_numbers(fix or [], "--fix"),
        bounds=parse_bounds(bounds or []),
        normalization=normalization,
        density_bulk_per_nm3=rho_bulk,
        loss=loss,
        regime=regime.regime(),
    )
    check_shape_name(type(pore), shape)
    check_plot(plot)
    check_outputs(output)
    density, profiles = load_profile(profile_path)
    return run_fit(pore, density, profiles, shape, spec, blocks=blocks, output=output, plot=plot, echo=_echo)


@app.command()
def run(config: Annotated[Path, typer.Argument(help="файл настроек .toml: [pore], [profile], [fit]")]) -> int:
    """Профиль и фит одним прогоном по файлу настроек."""
    settings = load_config(config)
    pore_values = dict(settings.pore)
    pore_type = pore_class(str(pore_values.pop(PORE_TYPE_KEY, DEFAULT_PORE)))
    source = settings.profile
    fit_settings = settings.fit
    if fit_settings is None and source.input is not None:
        raise CommandLineError(f"{config.name}: профиль готов (input), а раздела [fit] нет — нечего делать")
    if fit_settings is not None:  # всё, что можно проверить, — до долгого прохода по траектории
        check_shape_name(pore_type, fit_settings.shape)
        check_plot(fit_settings.plot)
        check_outputs(fit_settings.output)
    if source.input is not None:
        pore = require_dimensions(pore_type, pore_values, CONFIG_HINT)
        density, profiles = load_profile(require_file(source.input))
    else:
        assert source.topology is not None and source.select is not None  # проверено в ProfileConfig
        check_profiles_output(source.output)
        check_outputs(source.pore_output)
        pore, profiles = build_profiles(
            pore_type,
            pore_values,
            source.topology,
            source.trajectory,
            source.select,
            centering=source.centering,
            pore_center_nm=source.pore_center_nm,
            bins=source.bins or DEFAULT_BIN_COUNT,
            bin_width_nm=source.bin_width_nm,
            start=source.start,
            stop=source.stop,
            block=source.block,
            stride=source.stride,
            per=source.per,
            orientation=source.orientation,
            output=source.output,
            pore_output=source.pore_output,
            echo=_echo,
            hint=CONFIG_HINT,
        )
        density = profiles.mean()
    if fit_settings is None:
        return EXIT_SUCCESS
    return run_fit(
        pore,
        density,
        profiles,
        fit_settings.shape,
        fit_settings.spec(),
        blocks=fit_settings.blocks,
        output=fit_settings.output,
        plot=fit_settings.plot,
        echo=_echo,
    )


def _describe_pore(name: str, pore_type: type[Pore]) -> str:
    fields = []
    for field_name, field in pore_type.model_fields.items():
        if field.is_required():
            fields.append(field_name)
        else:
            default = field.default.value if isinstance(field.default, Enum) else field.default
            fields.append(f"{field_name}={default}")
    lines = [f"пора {name}", f"  поля (-g): {', '.join(fields)}"]
    for shape_name, shape_class in pore_type.own_registry().items():
        regimes = ", ".join(
            f"{regime.value} θ {lower:g}–{upper:g}°" for regime, (lower, upper) in shape_class.regimes.items()
        )
        axis = shape_class.symmetry_axis
        axis_text = f"  ось: {axis.orientation.value} ({axis.locate.value})" if axis is not None else ""
        lines.append(f"  {shape_name}  [{regimes or 'без θ'}]{axis_text}")
        lines.append(f"    свободны по умолчанию: {', '.join(shape_class.default_optimize)}")
        for parameter in (*shape_class.parameters, PROFILE_SHIFT):  # shift есть у каждой модели
            default = "из данных" if parameter.default is None else f"{parameter.default:g}"
            lower, upper = parameter.bounds
            lines.append(
                f"    {parameter.name:>6}  по умолчанию {default:<9} [{lower:g}, {upper:g}]  {parameter.unit.value}"
            )
    return "\n".join(lines)


@contextmanager
def _readable_warnings() -> Iterator[None]:
    """Предупреждения PANDA — одной строкой в stderr, без пути к файлу библиотеки."""

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        typer.echo(f"предупреждение: {message}", err=True)

    with warnings.catch_warnings():
        warnings.showwarning = show
        yield


def main(arguments: Sequence[str] | None = None) -> int:
    """Запуск команды; возвращает код возврата (0 — успех, 1 — ошибка входа, 2 — фит с флагами проблем)."""
    command_line = list(sys.argv[1:] if arguments is None else arguments)
    try:
        with _readable_warnings():
            code = app(args=command_line or ["--help"], prog_name="panda", standalone_mode=False)
    except typer.TyperException as error:  # ошибки разбора команды: у click код 2, но 2 у нас — флаги фита
        typer.echo(f"ошибка: {error.format_message()}", err=True)
        return EXIT_INPUT_ERROR
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("прервано", err=True)
        return EXIT_INTERRUPTED
    except ValidationError as error:  # до ValueError: коротко, с именем поля, без ссылок на документацию pydantic
        typer.echo(f"ошибка: {_validation_message(error)}", err=True)
        return EXIT_INPUT_ERROR
    except (PandaError, ValueError, OSError) as error:
        typer.echo(f"ошибка: {error}", err=True)
        return EXIT_INPUT_ERROR
    return code if isinstance(code, int) else EXIT_SUCCESS


def _validation_message(error: ValidationError) -> str:
    parts = []
    for issue in error.errors(include_url=False):
        location = ".".join(str(item) for item in issue["loc"])
        message = str(issue["msg"]).removeprefix("Value error, ")
        parts.append(f"{location}: {message}" if location else message)
    return f"{error.title}: " + "; ".join(parts)


def console() -> None:
    """Точка входа консольного скрипта `panda`."""
    raise SystemExit(main())
