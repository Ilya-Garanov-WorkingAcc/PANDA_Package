"""Графики результатов (необязательная зависимость: `pip install 'panda-md[plot]'`).

Единственный пакет, импортирующий matplotlib; функции принимают готовые объекты PANDA и рисуют на переданной оси
(или создают новую) — ничего не считают сами.
"""

from panda_md.plotting.figures import (
    plot_accuracy,
    plot_blocks,
    plot_density_map,
    plot_fit,
    plot_stability,
    supported_figure_formats,
)

__all__ = [
    "plot_accuracy",
    "plot_blocks",
    "plot_density_map",
    "plot_fit",
    "plot_stability",
    "supported_figure_formats",
]
