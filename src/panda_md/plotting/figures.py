"""Графики: профиль и модель, фит по блокам, 2D-карта плотности, диаграмма устойчивости, точность метода."""

import numpy as np

from panda_md.analysis.accuracy import AccuracyStudy
from panda_md.analysis.blocks import FitResultSet
from panda_md.analysis.stability import StabilityDiagram
from panda_md.core.errors import MissingDependencyError
from panda_md.core.result import FitResult
from panda_md.trajectory.maps import DensityMap

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import ListedColormap
except ImportError as error:  # pragma: no cover — проверяется тестом без matplotlib
    raise MissingDependencyError("для графиков нужен matplotlib: pip install 'panda-md[plot]'") from error

EMPTY_CELL_COLOR = "#f0f0f0"  # точки диаграммы, где ни одна форма не существует


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created = plt.subplots()
    return created


def plot_fit(result: FitResult, ax: Axes | None = None) -> Axes:
    """Данные и модель в безразмерных координатах окна фита."""
    axes = _axes(ax)
    axes.plot(result.coordinate, result.data, ".", markersize=3, label="данные")
    axes.plot(
        result.coordinate,
        result.model_curve,
        "-",
        label=(
            f"{result.shape_name}: θ = {result.values['theta']:.1f}°" if "theta" in result.values else result.shape_name
        ),
    )
    axes.set_xlabel("координата / размер поры")
    axes.set_ylabel("плотность (нормированная)")
    axes.legend()
    return axes


def plot_blocks(blocks: FitResultSet, parameter: str = "theta", ax: Axes | None = None) -> Axes:
    """Значение параметра в каждом блоке, среднее и ошибка среднего."""
    axes = _axes(ax)
    times_ns = 0.5 * (blocks.block_start_ps + blocks.block_stop_ps) / 1000.0
    values = blocks.values(parameter)
    axes.plot(times_ns, values, "o-", label="блоки")
    mean = blocks.mean(parameter)
    axes.axhline(mean, color="black", linewidth=1, label=f"среднее {mean:.4g} ± {blocks.sem(parameter):.2g}")
    axes.set_xlabel("время, нс")
    axes.set_ylabel(parameter)
    axes.legend()
    return axes


def plot_density_map(density: DensityMap, ax: Axes | None = None) -> Axes:
    """ρ(x, z) в нм⁻³: x — вдоль длинной стороны (ось формы в 0, если центрировано), z — ось профиля."""
    axes = _axes(ax)
    mesh = axes.pcolormesh(density.length_edges_nm, density.height_edges_nm, density.density_per_nm3.T, shading="flat")
    axes.figure.colorbar(mesh, ax=axes, label="ρ, нм⁻³")
    axes.set_xlabel("x, нм" + (" (ось формы в 0)" if density.axis_centered else ""))
    axes.set_ylabel(f"z, нм (система {density.frame.value})")
    axes.set_aspect("equal")
    return axes


def plot_stability(diagram: StabilityDiagram, ax: Axes | None = None) -> Axes:
    """Устойчивая форма в каждой точке (φ, l); пустые клетки — ни одна форма не существует."""
    axes = _axes(ax)
    names = list(diagram.shape_names)
    codes = np.array(
        [[names.index(name) if name is not None else -1 for name in row] for row in diagram.stable_shapes], dtype=float
    )
    palette = plt.get_cmap("tab20", max(len(names), 1))
    colors = [EMPTY_CELL_COLOR, *[palette(index) for index in range(len(names))]]
    axes.pcolormesh(
        np.asarray(diagram.volume_fractions),
        np.asarray(diagram.length_ratios),
        codes + 1,
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=len(names) + 0.5,
        shading="nearest",
    )
    present = sorted({name for row in diagram.stable_shapes for name in row if name is not None}, key=names.index)
    for name in present:
        axes.plot([], [], "s", color=palette(names.index(name)), label=name)
    axes.set_xlabel("φ")
    axes.set_ylabel("l = L / размер поры")
    axes.set_title(f"{diagram.pore_name}: θ = {diagram.contact_angle_deg:g}°, δ = {diagram.wall_layer:g}")
    if present:
        axes.legend(fontsize="small")
    return axes


def plot_accuracy(study: AccuracyStudy, ax: Axes | None = None) -> Axes:
    """Средняя и максимальная |Δθ| и смещение по значениям условия (значения — равномерно, с подписями)."""
    axes = _axes(ax)
    positions = np.arange(len(study.labels))
    axes.plot(positions, study.mean_absolute(), "o-", label="среднее |Δθ|")
    axes.plot(positions, study.maximum_absolute(), "s--", label="максимум |Δθ|")
    axes.plot(positions, study.bias(), "^:", label="смещение")
    axes.axhline(0.0, color="grey", linewidth=0.5)
    axes.set_xticks(positions, study.labels)
    axes.set_xlabel(study.varied)
    axes.set_ylabel("Δθ, °")
    axes.set_title(study.shape_name)
    axes.legend()
    return axes
