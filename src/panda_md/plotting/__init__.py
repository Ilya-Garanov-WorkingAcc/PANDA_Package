"""Result plots (optional dependency: `pip install 'panda-md[plot]'`).

The only package that imports matplotlib; the functions take ready PANDA objects and draw on the given axes
(or create new ones) — they compute nothing themselves.
"""

from panda_md.plotting.figures import (
    plot_accuracy,
    plot_blocks,
    plot_density_map,
    plot_fit,
    plot_profile,
    plot_stability,
    save_fit_figure,
    supported_figure_formats,
)

__all__ = [
    "plot_accuracy",
    "plot_blocks",
    "plot_density_map",
    "plot_fit",
    "plot_profile",
    "plot_stability",
    "save_fit_figure",
    "supported_figure_formats",
]
