"""Settings file of `panda run`: pore, profile (from a trajectory or ready-made), fit. Paths are relative to it."""

import sys
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from panda_md.cli.options import CommandLineError
from panda_md.cli.pipeline import AUTO_SHAPE
from panda_md.core.enums import Orientation, ParticleGrouping, ProfileCentering
from panda_md.core.spec import FitSpec
from panda_md.core.timing import TimeValue
from panda_md.core.types import PandaModel

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — Python 3.10
    import tomli as tomllib


class ProfileConfig(PandaModel):
    """Section [profile]: either a ready-made profile (input) or a trajectory (topology, trajectory, select)."""

    input: Path | None = None  # .npz/.csv of a profile or of a set of block profiles
    topology: Path | None = None
    trajectory: tuple[Path, ...] = ()
    select: str | None = None  # selection of the liquid atoms (MDAnalysis syntax)
    centering: ProfileCentering | None = None  # None — pore if the walls are known, otherwise liquid
    pore_center_nm: float | None = None  # pore centre along the profile axis; None — from the wall atoms
    bins: int | None = Field(default=None, ge=1)  # None — 200 bins unless bin_width_nm is set
    bin_width_nm: float | None = Field(default=None, gt=0.0)
    start: TimeValue = 0.0
    stop: TimeValue | None = None
    block: TimeValue | None = None
    stride: int = Field(default=1, ge=1)
    per: ParticleGrouping = ParticleGrouping.ATOM
    orientation: Orientation = Orientation.AUTO
    output: Path | None = None  # where to save the block profiles
    pore_output: Path | None = (
        None  # where to save the pore (JSON) — needed if the sizes are measured from the trajectory
    )

    @model_validator(mode="after")
    def _one_source(self) -> "ProfileConfig":
        from_trajectory = self.topology is not None or bool(self.trajectory) or self.select is not None
        if self.input is not None and from_trajectory:
            raise ValueError(
                "input and topology/trajectory/select are mutually exclusive; the profile is either ready or built"
            )
        if self.input is None and (self.topology is None or not self.trajectory or self.select is None):
            raise ValueError("no profile source; set input (a ready-made profile) or topology, trajectory and select")
        if self.bins is not None and self.bin_width_nm is not None:
            raise ValueError("bins and bin_width_nm are mutually exclusive; set only one")
        return self


class FitConfig(FitSpec):
    """Section [fit]: the FitSpec fields plus shape, output, plot."""

    shape: str = AUTO_SHAPE
    blocks: bool = False  # also fit each block (needs a set of block profiles)
    output: Path | None = None
    plot: Path | None = None

    def spec(self) -> FitSpec:
        return FitSpec.model_validate(self.model_dump(include=set(FitSpec.model_fields)))


class RunConfig(PandaModel):
    pore: dict[str, Any]  # type and fields of the pore, as in the pore JSON
    profile: ProfileConfig
    fit: FitConfig | None = None  # no section — profile only

    def resolved(self, base: Path) -> "RunConfig":
        """Relative paths are taken from the directory of the settings file."""

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else base / path

        def resolve_optional(path: Path | None) -> Path | None:
            return None if path is None else resolve(path)

        profile = self.profile.model_copy(
            update={
                "input": resolve_optional(self.profile.input),
                "topology": resolve_optional(self.profile.topology),
                "trajectory": tuple(resolve(path) for path in self.profile.trajectory),
                "output": resolve_optional(self.profile.output),
                "pore_output": resolve_optional(self.profile.pore_output),
            }
        )
        fit = None
        if self.fit is not None:
            fit = self.fit.model_copy(
                update={"output": resolve_optional(self.fit.output), "plot": resolve_optional(self.fit.plot)}
            )
        return self.model_copy(update={"profile": profile, "fit": fit})


def load_config(path: Path) -> RunConfig:
    if not path.is_file():
        raise CommandLineError(f"settings file not found: {path}")
    try:
        content = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise CommandLineError(f"{path}: TOML error: {error}") from None
    return RunConfig.model_validate(content).resolved(path.parent)
