"""Примеры из examples/ запускаются и печатают ожидаемое (те, которым не нужны файлы МД)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from panda_md.cli.config import load_config

EXAMPLES = Path(__file__).parents[1] / "examples"


def run_example(name: str, *arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / name), *arguments],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_synthetic_fit_recovers_the_angle() -> None:
    out = run_example("01_synthetic_fit.py")
    fitted = float(out.split("fitted ")[1].split("°")[0])
    assert fitted == pytest.approx(120.0, abs=1.5)


def test_saved_profiles_give_the_graphite_angle(tmp_path: Path) -> None:
    arguments = ("--output", str(tmp_path)) if _has_matplotlib() else ()
    out = run_example("03_saved_profiles.py", *arguments)
    assert "slit/roll: 8 блоков" in out
    if arguments:
        assert (tmp_path / "graphite_water.png").stat().st_size > 0


def test_stability_diagram_example(tmp_path: Path) -> None:
    arguments = ("--output", str(tmp_path)) if _has_matplotlib() else ()
    out = run_example("04_stability_diagram.py", *arguments)
    assert "slit: shapes on the diagram" in out and "square: shapes on the diagram" in out


def test_custom_shape_is_fitted_and_wins_the_comparison() -> None:
    out = run_example("05_custom_shape.py")
    comparison = out.split("сравнение форм")[1].splitlines()
    assert comparison[1].split()[0] == "bottom_film"


def test_example_settings_file_is_valid() -> None:
    config = load_config(EXAMPLES / "panda.toml")
    assert config.fit is not None and config.fit.shape == "roll"
    assert config.profile.topology == EXAMPLES / "system.tpr"


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True
