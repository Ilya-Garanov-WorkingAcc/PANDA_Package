"""Файл настроек `panda run`: пора, профиль (из траектории или готовый), фит. Пути — относительно файла настроек."""

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
    """Раздел [profile]: либо готовый профиль (input), либо траектория (topology, trajectory, select)."""

    input: Path | None = None  # .npz/.csv профиля или набора профилей по блокам
    topology: Path | None = None
    trajectory: tuple[Path, ...] = ()
    select: str | None = None  # выборка атомов жидкости (синтаксис MDAnalysis)
    centering: ProfileCentering = ProfileCentering.PORE
    pore_center_nm: float | None = None  # центр поры вдоль оси профиля; None — по атомам стенок
    bins: int | None = Field(default=None, ge=1)  # None — 200 бинов, если не задана bin_width_nm
    bin_width_nm: float | None = Field(default=None, gt=0.0)
    start: TimeValue = 0.0
    stop: TimeValue | None = None
    block: TimeValue | None = None
    stride: int = Field(default=1, ge=1)
    per: ParticleGrouping = ParticleGrouping.ATOM
    orientation: Orientation = Orientation.AUTO
    output: Path | None = None  # куда сохранить профили по блокам
    pore_output: Path | None = None  # куда сохранить пору (JSON) — нужно, если размеры найдены по траектории

    @model_validator(mode="after")
    def _one_source(self) -> "ProfileConfig":
        from_trajectory = self.topology is not None or bool(self.trajectory) or self.select is not None
        if self.input is not None and from_trajectory:
            raise ValueError("input и topology/trajectory/select взаимоисключающие: профиль готов или строится")
        if self.input is None and (self.topology is None or not self.trajectory or self.select is None):
            raise ValueError("задай input (готовый профиль) или topology, trajectory и select")
        if self.bins is not None and self.bin_width_nm is not None:
            raise ValueError("bins и bin_width_nm взаимоисключающие")
        return self


class FitConfig(FitSpec):
    """Раздел [fit]: поля FitSpec и форма, выход, график."""

    shape: str = AUTO_SHAPE
    blocks: bool = False  # ещё и фит каждого блока (нужен набор профилей по блокам)
    output: Path | None = None
    plot: Path | None = None

    def spec(self) -> FitSpec:
        return FitSpec.model_validate(self.model_dump(include=set(FitSpec.model_fields)))


class RunConfig(PandaModel):
    pore: dict[str, Any]  # type и поля поры, как в JSON поры
    profile: ProfileConfig
    fit: FitConfig | None = None  # нет раздела — только профиль

    def resolved(self, base: Path) -> "RunConfig":
        """Относительные пути — от каталога файла настроек."""

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
        raise CommandLineError(f"файл настроек {path} не найден")
    try:
        content = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise CommandLineError(f"{path}: ошибка TOML: {error}") from None
    return RunConfig.model_validate(content).resolved(path.parent)
