"""The `panda` commands: shapes, profile, fit, run. A thin wrapper over the API; exit codes: 0 — success, 1 — input
error, 2 — the fit finished with problem flags."""

import sys
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
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
from panda_md.core.version import library_version
from panda_md.pores.registry import pore_class, pore_types
from panda_md.trajectory.density import DEFAULT_BIN_COUNT

app = typer.Typer(
    name="panda",
    help="PANDA: contact angle and interface shape of a liquid in a pore from an MD density profile.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,  # otherwise rich eats “[md]” in “panda-md[md]”
)


class RegimeChoice(str, Enum):
    AUTO = "auto"  # all regimes declared by the shape
    WETTING = Regime.WETTING.value
    NON_WETTING = Regime.NON_WETTING.value

    def regime(self) -> Regime | None:
        return None if self is RegimeChoice.AUTO else Regime(self.value)


PoreNameOption = Annotated[
    str | None, typer.Option("--pore", help="pore type (panda shapes); default: slit or type from --pore-file")
]
PoreFileOption = Annotated[Path | None, typer.Option("--pore-file", help="JSON with the pore fields")]
GeometryOption = Annotated[
    list[str] | None, typer.Option("-g", "--geometry", help="pore field: name=value (width_nm=8), may be repeated")
]


def _echo(text: str) -> None:
    typer.echo(text)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"panda-md {library_version()}")
        raise typer.Exit()


@app.callback()
def _options(
    version: Annotated[
        bool, typer.Option("--version", callback=_print_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """PANDA: contact angle and interface shape of a liquid in a pore from an MD density profile."""


@app.command()
def shapes(pore: Annotated[str | None, typer.Option("--pore", help="show only this pore type")] = None) -> int:
    """Pore types, their fields and interface shapes: parameters, bounds, regimes, symmetry axis."""
    selected = {pore: pore_class(pore)} if pore is not None else pore_types()
    for name, pore_type in selected.items():
        _echo(_describe_pore(name, pore_type))
    return EXIT_SUCCESS


@app.command()
def profile(
    topology: Annotated[Path, typer.Argument(help="topology (.gro, .tpr, .pdb, …)")],
    trajectory: Annotated[list[Path], typer.Argument(help="trajectory files (.xtc, .trr, …), in order")],
    select: Annotated[str, typer.Option("--select", help="selection of the liquid atoms (MDAnalysis syntax)")],
    output: Annotated[Path, typer.Option("-o", "--output", help="block profiles (.npz)")],
    pore_name: PoreNameOption = None,
    pore_file: PoreFileOption = None,
    geometry: GeometryOption = None,
    walls: Annotated[
        str | None, typer.Option("--walls", help="selection of the wall atoms or wall coordinates “z1,z2” (nm)")
    ] = None,
    center_by: Annotated[
        ProfileCentering | None,
        typer.Option("--center-by", help="what to put at 0; default: pore if the walls are known, otherwise liquid"),
    ] = None,
    bins: Annotated[
        int | None, typer.Option("--bins", min=1, help=f"number of bins (default {DEFAULT_BIN_COUNT})")
    ] = None,
    bin_width: Annotated[float | None, typer.Option("--bin-width", help="bin width, nm (instead of --bins)")] = None,
    start: Annotated[str, typer.Option("--start", help="start: ps or “2 ns”")] = "0",
    stop: Annotated[str | None, typer.Option("--stop", help="stop: ps or “10 ns”")] = None,
    block: Annotated[str | None, typer.Option("--block", help="block length: ps or “1 ns”")] = None,
    stride: Annotated[int, typer.Option("--stride", min=1, help="every n-th frame")] = 1,
    per: Annotated[ParticleGrouping, typer.Option("--per", help="particle: atom or molecular centre of mass")] = (
        ParticleGrouping.ATOM
    ),
    orientation: Annotated[Orientation, typer.Option("--orientation", help="whether to mirror the profile z → −z")] = (
        Orientation.AUTO
    ),
    pore_output: Annotated[
        Path | None, typer.Option("--pore-output", help="save the pore to JSON (for panda fit --pore-file)")
    ] = None,
) -> int:
    """Density profiles over time blocks from an MD trajectory (needs panda-md[md]).

    Without pore dimensions (--pore-file, -g) a slit or cylinder pore is measured from the wall atoms (--walls
    “selection”) in the --start frame; --pore-output saves the measured pore — its JSON is then passed to panda fit
    --pore-file.
    """
    if bins is not None and bin_width is not None:
        raise CommandLineError("--bins and --bin-width are mutually exclusive; set only one")
    if bin_width is not None and bin_width <= 0.0:
        raise CommandLineError(f"--bin-width must be positive: {bin_width:g}")
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
    profile_path: Annotated[Path, typer.Argument(metavar="PROFILE", help="profile .npz/.csv or block profiles")],
    pore_name: PoreNameOption = None,
    pore_file: PoreFileOption = None,
    geometry: GeometryOption = None,
    shape: Annotated[str, typer.Option("--shape", help="shape (panda shapes) or auto — try all")] = AUTO_SHAPE,
    optimize: Annotated[
        list[str] | None, typer.Option("--optimize", help="free parameters: theta,delta or repeated")
    ] = None,
    fix: Annotated[list[str] | None, typer.Option("--fix", help="fixed parameter: name=value")] = None,
    bounds: Annotated[list[str] | None, typer.Option("--bounds", help="bounds for this fit: name=lower:upper")] = None,
    rho_bulk: Annotated[
        float | None, typer.Option("--rho-bulk", help="bulk density of the liquid, particles/nm³")
    ] = None,
    regime: Annotated[RegimeChoice, typer.Option("--regime", help="wetting regime")] = RegimeChoice.AUTO,
    normalization: Annotated[Normalization, typer.Option("--normalization", help="data normalization")] = (
        Normalization.AUTO
    ),
    loss: Annotated[LossFunction, typer.Option("--loss", help="loss function")] = LossFunction.L1,
    blocks: Annotated[bool, typer.Option("--blocks", help="also fit each block (spread over blocks)")] = False,
    output: Annotated[Path | None, typer.Option("-o", "--output", help="result (.json)")] = None,
    plot: Annotated[Path | None, typer.Option("--plot", help="fit plot (.png, .pdf; needs panda-md[plot])")] = None,
) -> int:
    """Fit the profile with an interface shape; summary to the terminal, result to JSON."""
    pore = load_pore(pore_name, pore_file, geometry or [])
    free = split_names(optimize) if optimize else None
    if free is not None and not free:
        raise CommandLineError("--optimize names no parameter; give at least one")
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
def run(config: Annotated[Path, typer.Argument(help="settings file .toml: [pore], [profile], [fit]")]) -> int:
    """Profile and fit in one run from a settings file."""
    settings = load_config(config)
    pore_values = dict(settings.pore)
    pore_type = pore_class(str(pore_values.pop(PORE_TYPE_KEY, DEFAULT_PORE)))
    source = settings.profile
    fit_settings = settings.fit
    if fit_settings is None and source.input is not None:
        raise CommandLineError(
            f"{config.name}: nothing to do: the profile is ready (input) and there is no [fit] section; add [fit]"
        )
    if fit_settings is not None:  # check all that can be checked before the long pass over the trajectory
        check_shape_name(pore_type, fit_settings.shape)
        check_plot(fit_settings.plot)
        check_outputs(fit_settings.output)
    if source.input is not None:
        pore = require_dimensions(pore_type, pore_values, CONFIG_HINT)
        density, profiles = load_profile(require_file(source.input))
    else:
        assert source.topology is not None and source.select is not None  # checked in ProfileConfig
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
    lines = [f"pore {name}", f"  fields (-g): {', '.join(fields)}"]
    for shape_name, shape_class in pore_type.own_registry().items():
        regimes = ", ".join(
            f"{regime.value} θ {lower:g}–{upper:g}°" for regime, (lower, upper) in shape_class.regimes.items()
        )
        axis = shape_class.symmetry_axis
        axis_text = f"  axis: {axis.orientation.value} ({axis.locate.value})" if axis is not None else ""
        lines.append(f"  {shape_name}  [{regimes or 'no θ'}]{axis_text}")
        lines.append(f"    free by default: {', '.join(shape_class.default_optimize)}")
        for parameter in (*shape_class.parameters, PROFILE_SHIFT):  # every model has shift
            default = "from data" if parameter.default is None else f"{parameter.default:g}"
            lower, upper = parameter.bounds
            lines.append(
                f"    {parameter.name:>6}  default {default:<9} [{lower:g}, {upper:g}]  {parameter.unit.value}"
            )
    return "\n".join(lines)


@contextmanager
def _readable_warnings() -> Iterator[None]:
    """PANDA warnings as one line to stderr, without the path to the library file."""

    def show(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        typer.echo(f"warning: {message}", err=True)

    with warnings.catch_warnings():
        warnings.showwarning = show
        yield


def main(arguments: Sequence[str] | None = None) -> int:
    """Run a command; return the exit code (0 — success, 1 — input error, 2 — fit with problem flags)."""
    command_line = list(sys.argv[1:] if arguments is None else arguments)
    try:
        with _readable_warnings():
            code = app(args=command_line or ["--help"], prog_name="panda", standalone_mode=False)
    except typer.TyperException as error:  # command parsing errors: click uses code 2, but 2 here means fit flags
        typer.echo(f"error: {_strict(error.format_message())}", err=True)
        return EXIT_INPUT_ERROR
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("interrupted", err=True)
        return EXIT_INTERRUPTED
    except (
        ValidationError
    ) as error:  # before ValueError: short, with the field name, without links to the pydantic docs
        typer.echo(f"error: {_validation_message(error)}", err=True)
        return EXIT_INPUT_ERROR
    except (PandaError, ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        return EXIT_INPUT_ERROR
    return code if isinstance(code, int) else EXIT_SUCCESS


def _validation_message(error: ValidationError) -> str:
    parts = []
    for issue in error.errors(include_url=False):
        location = ".".join(str(item) for item in issue["loc"])
        message = str(issue["msg"]).removeprefix("Value error, ")
        parts.append(f"{location}: {message}" if location else message)
    return f"{error.title}: " + "; ".join(parts)


def _strict(message: str) -> str:
    """click messages ('Missing argument ...') in the form of the library messages: lower-case, no final period."""
    message = message.strip().rstrip(".")
    return message[:1].lower() + message[1:]


def _tolerate_unencodable_output() -> None:
    """Write characters the output encoding lacks as escapes (θ → \\u03b8) instead of failing the command.
    On Windows redirected output uses the ANSI code page (cp1251 has no θ, ≈, →) unless PYTHONUTF8=1 is set."""
    for stream in (sys.stdout, sys.stderr):
        # a stream that is not an io text stream (replaced, detached) is left as it is
        with suppress(AttributeError, ValueError):
            stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]


def console() -> None:
    """Entry point of the `panda` console script."""
    _tolerate_unencodable_output()
    raise SystemExit(main())
