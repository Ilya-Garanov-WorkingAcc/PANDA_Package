"""Проверка упаковки: пакет импортируется, версия берётся из метаданных, типы объявлены."""

from importlib.metadata import version
from importlib.resources import files

import panda_md


def test_version_matches_metadata() -> None:
    assert panda_md.__version__ == version("panda-md")


def test_package_is_typed() -> None:
    assert files("panda_md").joinpath("py.typed").is_file()
