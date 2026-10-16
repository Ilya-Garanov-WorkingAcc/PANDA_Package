"""Full pipeline on an MD trajectory of a slit pore (needs `pip install 'panda-md[md,plot]'`).

trajectory → slit dimensions from the wall atoms → density profiles by time blocks → shape comparison →
fit of every block (θ ± its error) → 2D density map with axial centering → figures.

    python examples/02_md_pipeline.py system.tpr system.xtc --walls "resname GRAPHITE" \\
        --select "resname WATER and name OW" --density-bulk 33.1 --start "2 ns" --block "1 ns" --output out/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import panda_md as panda
from panda_md.plotting import plot_blocks, plot_density_map, plot_fit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topology", type=Path)
    parser.add_argument("trajectory", type=Path, nargs="+")
    parser.add_argument("--walls", required=True, help="selection of the wall atoms")
    parser.add_argument("--select", required=True, help="selection of the analysed liquid")
    parser.add_argument("--density-bulk", type=float, required=True, help="bulk density, particles per nm³")
    parser.add_argument("--start", default="0 ps")
    parser.add_argument("--block", default=None, help="block length, e.g. '1 ns'")
    parser.add_argument("--shape", default=None, help="shape to fit per block; default: the best one")
    parser.add_argument("--output", type=Path, default=Path("panda_output"))
    return parser.parse_args()


def save(axes: plt.Axes, path: Path) -> None:
    axes.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(axes.figure)


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    trajectory = panda.load_trajectory(arguments.topology, *arguments.trajectory)
    pore = panda.SlitPore.from_trajectory(
        trajectory, walls=arguments.walls, liquid=arguments.select, time=arguments.start
    )
    print(pore)
    profiles = panda.density_profile(trajectory, pore, arguments.select, start=arguments.start, block=arguments.block)
    profiles.save(arguments.output / "profiles.npz")

    spec = panda.FitSpec(density_bulk_per_nm3=arguments.density_bulk)
    comparison = panda.compare_shapes(pore, profiles.mean(), spec)
    print(comparison)
    if arguments.shape is None and comparison.indistinguishable:
        print(
            f"the profile cannot tell apart {', '.join(comparison.indistinguishable)}: "
            "check the 2D map below and the stability diagram, or choose with --shape"
        )
    shape = arguments.shape or comparison.best.shape_name
    model = pore.interface(shape)
    result = model.fit(profiles.mean(), density_bulk_per_nm3=arguments.density_bulk)
    print(result)
    save(plot_fit(result), arguments.output / "fit.png")
    (arguments.output / "result.json").write_text(result.model_dump_json(indent=2))

    if profiles.block_count > 1:
        blocks = panda.fit_blocks(model, profiles, spec)
        print(blocks)
        save(plot_blocks(blocks), arguments.output / "blocks.png")
    if model.shape.symmetry_axis is not None:
        density = panda.density_map(trajectory, pore, arguments.select, shape=shape, start=arguments.start)
        save(plot_density_map(density), arguments.output / "map.png")
    print(f"results in {arguments.output}")


if __name__ == "__main__":
    main()
