"""Исследование точности на синтетике и графики (matplotlib — необязательная зависимость)."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md import FitSpec, SlitPore
from panda_md.analysis.accuracy import (
    AccuracyCase,
    AccuracyStudy,
    accuracy_study,
    angle_errors,
    effective_noise_std,
)
from panda_md.analysis.blocks import fit_blocks
from panda_md.analysis.stability import stability_diagram
from panda_md.core.errors import MissingDependencyError
from panda_md.synthetic.points import synthetic_trajectory
from panda_md.synthetic.profiles import synthetic_profile
from panda_md.trajectory.maps import density_map

DENSITY_BULK = 30.0
PORE = SlitPore(width_nm=8.0, length_nm=16.0)
ROLL_TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}


def test_without_noise_the_angle_is_recovered() -> None:
    errors = angle_errors(
        PORE.interface("roll"),
        AccuracyCase(truth=ROLL_TRUTH, noise_std=0.0),
        density_bulk_per_nm3=DENSITY_BULK,
        seeds=[0],
    )
    assert abs(errors.errors_deg[0]) < 0.05  # среднее по бину против модели в центрах — малая систематика


def test_study_summarises_errors_per_condition() -> None:
    cases = {
        "0.5": (0.5, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.005)),
        "2": (2.0, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.02)),
    }
    study = accuracy_study(PORE.interface("roll"), "шум, %", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=range(3))
    assert study.errors_deg.shape == (2, 3)
    assert study.mean_absolute()[1] > study.mean_absolute()[0]  # больше шума — больше ошибка
    np.testing.assert_allclose(study.maximum_absolute(), np.max(np.abs(study.errors_deg), axis=1))
    table = study.table()
    assert table.splitlines()[0].startswith("| шум, %") and "| 2 |" in table
    with pytest.raises(ValidationError, match="подписи"):
        AccuracyStudy(
            shape_name="roll",
            varied="x",
            labels=("a",),
            values=np.array([1.0, 2.0]),
            errors_deg=np.zeros((2, 1)),
            at_bound_counts=(0, 0),
        )


def test_noise_is_rescaled_with_the_bin_count() -> None:
    """Та же траектория, разбитая мельче, — шум на бин растёт как √(бинов / 200)."""
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01)) == pytest.approx(0.01)
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01, bin_count=400)) == pytest.approx(
        0.01 * np.sqrt(2.0)
    )
    with pytest.raises(ValidationError):
        AccuracyCase(truth=ROLL_TRUTH, noise_std=-0.1)


@pytest.fixture
def pyplot() -> object:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_figures_are_drawn_from_panda_objects(pyplot: object) -> None:
    from panda_md.plotting import plot_accuracy, plot_blocks, plot_density_map, plot_fit, plot_stability

    roll = PORE.interface("roll")
    profile = synthetic_profile(roll, ROLL_TRUTH, density_bulk_per_nm3=DENSITY_BULK, noise_std=0.005, seed=0)
    result = roll.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert "θ = 1" in plot_fit(result).get_legend_handles_labels()[1][1]

    trajectory = synthetic_trajectory(
        SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5).interface("roll"),
        {"theta": 120.0, "delta": 0.03, "phi": 0.3},
        density_bulk_per_nm3=33.0,
        frame_count=4,
        seed=0,
    )
    walls = SlitPore.from_trajectory(trajectory, walls="walls")
    assert (
        "ось формы"
        in plot_density_map(density_map(trajectory, walls, "liquid", shape="roll", bins=(20, 20))).get_xlabel()
    )

    from panda_md import density_profile

    profiles = density_profile(trajectory, walls, "liquid", bins=40, block="10 ps")
    blocks = fit_blocks(walls.interface("roll"), profiles, FitSpec(density_bulk_per_nm3=33.0))
    assert plot_blocks(blocks).get_ylabel() == "theta"

    diagram = stability_diagram(
        PORE, contact_angle_deg=150.0, wall_layer=0.0, volume_fractions=(0.1, 0.4, 0.7), length_ratios=(1.0, 2.0, 4.0)
    )
    assert plot_stability(diagram).get_title().startswith("slit")

    cases = {"0.5": (0.5, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.005))}
    study = accuracy_study(roll, "шум, %", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=range(2))
    assert plot_accuracy(study).get_xlabel() == "шум, %"


def test_plotting_without_matplotlib_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [module for module in sys.modules if module.startswith(("matplotlib", "panda_md.plotting"))]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(MissingDependencyError, match=r"panda-md\[plot\]"):
        importlib.import_module("panda_md.plotting")


def test_importing_package_does_not_import_matplotlib() -> None:
    import subprocess

    check = "import sys, panda_md; print('matplotlib' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


def test_fits_at_the_angle_bound_are_counted() -> None:
    cases = {"175": (175.0, AccuracyCase(truth={**ROLL_TRUTH, "theta": 175.0}, noise_std=0.01))}
    study = accuracy_study(PORE.interface("roll"), "θ, °", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=range(10))
    assert len(study.at_bound_counts) == 1 and 0 < study.at_bound_counts[0] <= 10
    assert "у границы θ" in study.table().splitlines()[0]
    with pytest.raises(ValueError, match="зерно"):
        accuracy_study(PORE.interface("roll"), "θ, °", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=[])


def test_density_map_figure_keeps_orientation_and_stability_colors(pyplot: object) -> None:
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    from panda_md.plotting import plot_density_map, plot_stability
    from panda_md.trajectory.maps import DensityMap

    density = DensityMap(
        length_edges_nm=np.linspace(-2.0, 2.0, 6),
        height_edges_nm=np.linspace(-1.0, 1.0, 4),
        density_per_nm3=np.arange(15.0).reshape(5, 3),
        frame="pore",
        axis_centered=True,
        frame_count=1,
        selection="тест",
    )
    mesh = plot_density_map(density).collections[0]
    np.testing.assert_allclose(np.asarray(mesh.get_array()).reshape(3, 5), density.density_per_nm3.T)

    diagram = stability_diagram(
        PORE, contact_angle_deg=150.0, wall_layer=0.0, volume_fractions=(0.001, 0.4, 0.97), length_ratios=(1.0, 2.0)
    )
    axes = plot_stability(diagram)
    axes.figure.canvas.draw()  # цвета клеток вычисляются при отрисовке
    colors = np.asarray(axes.collections[0].get_facecolor(), dtype=float)
    handles, labels = axes.get_legend_handles_labels()
    label_colors = {}
    for handle, label in zip(handles, labels, strict=True):
        assert isinstance(handle, Line2D)
        label_colors[label] = to_rgba(handle.get_color())
    flat = [name for row in diagram.stable_shapes for name in row]
    for color, name in zip(colors, flat, strict=True):
        expected = to_rgba("#f0f0f0") if name is None else label_colors[name]
        np.testing.assert_allclose(color, np.asarray(expected), atol=1e-6)


def test_save_fit_figure_writes_a_file_without_pyplot_state(tmp_path: Path) -> None:
    """Итоговый аудит: CLI сам импортировал matplotlib; сохранение графика — в panda_md.plotting."""
    import matplotlib.pyplot as plt

    from panda_md.plotting import save_fit_figure

    model = SlitPore(width_nm=8.0, length_nm=16.0).interface("roll")
    profile = synthetic_profile(model, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=30.0)
    before = plt.get_fignums()
    save_fit_figure(model.fit(profile, density_bulk_per_nm3=30.0), tmp_path / "fit.svg")
    assert (tmp_path / "fit.svg").read_text().startswith("<?xml")
    assert plt.get_fignums() == before  # открытых фигур pyplot не прибавилось


def test_only_plotting_imports_matplotlib() -> None:
    import re

    package = Path(__file__).parents[2] / "src" / "panda_md"
    offenders = [
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if path.parent.name != "plotting" and re.search(r"^\s*(import|from) matplotlib", path.read_text(), re.M)
    ]
    assert offenders == []
