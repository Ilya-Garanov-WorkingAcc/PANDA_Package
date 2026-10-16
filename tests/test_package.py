"""Проверка упаковки: пакет импортируется, версия берётся из метаданных, типы объявлены."""

import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files

import panda_md


def test_version_matches_metadata() -> None:
    assert panda_md.__version__ == version("panda-md")


def test_package_is_typed() -> None:
    assert files("panda_md").joinpath("py.typed").is_file()


def test_importing_package_does_not_import_mdanalysis() -> None:
    """MDAnalysis — необязательная зависимость: нужен только для чтения файлов МД (load_trajectory)."""
    check = "import sys, panda_md; print('MDAnalysis' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


def test_data_path_is_public() -> None:
    expected = {"ArrayTrajectory", "Axis", "Orientation", "ProfileCentering", "ProfileSet", "density_profile"}
    assert expected | {"load_trajectory"} <= set(panda_md.__all__)


def test_analysis_is_public() -> None:
    expected = {"FitResultSet", "ParticleGrouping", "ShapeComparison", "ShapeFit", "StatisticalInefficiency"}
    assert expected | {"compare_shapes", "fit_blocks", "statistical_inefficiency"} <= set(panda_md.__all__)
