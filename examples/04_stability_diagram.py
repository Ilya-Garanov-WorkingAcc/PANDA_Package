"""Stability diagram: which interface shape has the lowest capillary energy at given θ and δ.

For every pore length l = L / H and volume fraction φ the feasible shape with the smallest S_LV − cos θ · S_SL wins.

    python examples/04_stability_diagram.py [--output out/]
"""

import argparse
from pathlib import Path

import numpy as np

import panda_md as panda
from panda_md.analysis.stability import stability_diagram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="save the figure here (needs panda-md[plot])")
    arguments = parser.parse_args()

    for pore in (
        panda.SlitPore(width_nm=4.0, length_nm=8.0),
        panda.SquarePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0),
    ):
        diagram = stability_diagram(
            pore,
            contact_angle_deg=130.0,
            wall_layer=0.0,
            volume_fractions=np.linspace(0.05, 0.95, 19),
            length_ratios=np.linspace(0.5, 4.0, 15),
        )
        winners = sorted({name for row in diagram.stable_shapes for name in row if name})
        print(f"{pore.name}: shapes on the diagram — {', '.join(winners)}")
        if arguments.output is not None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from panda_md.plotting import plot_stability

            arguments.output.mkdir(parents=True, exist_ok=True)
            axes = plot_stability(diagram)
            axes.figure.savefig(arguments.output / f"stability_{pore.name}.png", dpi=150, bbox_inches="tight")
            plt.close(axes.figure)


if __name__ == "__main__":
    main()
