"""Plots: profile and model, fit over blocks, 2D density map, stability diagram, accuracy of the method."""

import math
from pathlib import Path

import numpy as np

from panda_md.analysis.accuracy import AccuracyStudy
from panda_md.analysis.blocks import FitResultSet
from panda_md.analysis.stability import StabilityDiagram
from panda_md.core.enums import ParameterName
from panda_md.core.errors import MissingDependencyError
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.core.result import FitResult
from panda_md.trajectory.maps import DensityMap

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
except ImportError as error:  # pragma: no cover — checked by a test without matplotlib
    raise MissingDependencyError("plots need matplotlib; run pip install 'panda-md[plot]'") from error

EMPTY_CELL_COLOR = "#f0f0f0"  # diagram points where no shape exists


def supported_figure_formats() -> tuple[str, ...]:
    """File extensions matplotlib can save a figure to (png, pdf, svg, …)."""
    return tuple(FigureCanvasBase.get_supported_filetypes())


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created = plt.subplots()
    return created


def plot_profile(profile: DensityProfile | ProfileSet, ax: Axes | None = None) -> Axes:
    """Density profile in nm and nm⁻³; for a set of time blocks — the mean with the standard error as error bars."""
    axes = _axes(ax)
    if isinstance(profile, ProfileSet):
        mean = profile.mean()
        axes.errorbar(mean.coordinate_nm, mean.density_per_nm3, yerr=profile.sem(), fmt=".", markersize=3,
                      label=f"mean of {profile.block_count} blocks ± standard error")  # fmt: skip
        frame = profile.frame
    else:
        axes.plot(profile.coordinate_nm, profile.density_per_nm3, ".-", markersize=3, label="profile")
        frame = profile.frame
    axes.set_xlabel(f"coordinate, nm ({frame.value} frame)")
    axes.set_ylabel("ρ, nm⁻³")
    axes.legend()
    return axes


def plot_fit(result: FitResult, ax: Axes | None = None) -> Axes:
    """Data and model in the dimensionless coordinates of the fit window."""
    axes = _axes(ax)
    axes.plot(result.coordinate, result.data, ".", markersize=3, label="data")
    axes.plot(
        result.coordinate,
        result.model_curve,
        "-",
        label=(
            f"{result.shape_name}: θ = {result.values[ParameterName.CONTACT_ANGLE.value]:.1f}°"
            if ParameterName.CONTACT_ANGLE.value in result.values
            else result.shape_name
        ),
    )
    axes.set_xlabel("coordinate / pore size")
    axes.set_ylabel("density (normalized)")
    axes.legend()
    return axes


def save_fit_figure(result: FitResult, path: str | Path, dpi: int = 150) -> None:
    """Save the fit plot to a file (format by extension) without pyplot: no window, no global figures."""
    figure = Figure()
    plot_fit(result, figure.subplots())
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_blocks(
    blocks: FitResultSet, parameter: str = ParameterName.CONTACT_ANGLE.value, ax: Axes | None = None
) -> Axes:
    """Value of the parameter in each block, the mean and the standard error of the mean."""
    axes = _axes(ax)
    times_ns = 0.5 * (blocks.block_start_ps + blocks.block_stop_ps) / 1000.0
    values = blocks.values(parameter)
    axes.plot(times_ns, values, "o-", label="blocks")
    mean = blocks.mean(parameter)
    error = blocks.sem(parameter)
    spread = f" ± {error:.2g}" if math.isfinite(error) else " (single block: error not estimated)"
    axes.axhline(mean, color="black", linewidth=1, label=f"mean {mean:.4g}{spread}")
    axes.set_xlabel("time, ns")
    axes.set_ylabel(parameter)
    axes.legend()
    return axes


def plot_density_map(density: DensityMap, ax: Axes | None = None) -> Axes:
    """ρ(x, z) in nm⁻³: x along the long side (shape axis at 0 if centred), z the profile axis."""
    axes = _axes(ax)
    mesh = axes.pcolormesh(density.length_edges_nm, density.height_edges_nm, density.density_per_nm3.T, shading="flat")
    axes.figure.colorbar(mesh, ax=axes, label="ρ, nm⁻³")
    axes.set_xlabel("x, nm" + (" (shape axis at 0)" if density.axis_centered else ""))
    flipped = ", mirrored z → −z" if density.flipped else ""
    axes.set_ylabel(f"z, nm ({density.frame.value} frame{flipped})")
    axes.set_aspect("equal")
    return axes


def plot_stability(diagram: StabilityDiagram, ax: Axes | None = None) -> Axes:
    """Stable shape at each point (φ, l); empty cells mean that no shape exists."""
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
    axes.set_ylabel("l = L / pore size")
    axes.set_title(f"{diagram.pore_name}: θ = {diagram.contact_angle_deg:g}°, δ = {diagram.wall_layer:g}")
    if present:
        axes.legend(fontsize="small")
    return axes


def plot_accuracy(study: AccuracyStudy, ax: Axes | None = None) -> Axes:
    """Mean and maximum |Δθ| and the bias over the condition values (values evenly spaced, with labels)."""
    axes = _axes(ax)
    positions = np.arange(len(study.labels))
    axes.plot(positions, study.mean_absolute(), "o-", label="mean |Δθ|")
    axes.plot(positions, study.maximum_absolute(), "s--", label="max |Δθ|")
    axes.plot(positions, study.bias(), "^:", label="bias")
    axes.axhline(0.0, color="grey", linewidth=0.5)
    axes.set_xticks(positions, study.labels)
    axes.set_xlabel(study.varied)
    axes.set_ylabel("Δθ, °")
    axes.set_title(study.shape_name)
    axes.legend()
    return axes
