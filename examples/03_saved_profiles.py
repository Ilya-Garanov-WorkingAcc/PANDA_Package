"""Analyse saved density profiles: water in a graphite slit pore (profiles shipped with the tests).

The profiles were computed from a 10 ns trajectory by `panda.density_profile` (8 blocks of 1 ns); the pore was
measured from the wall atoms. No MD files are needed.

    python examples/03_saved_profiles.py [--output out/]
"""

import argparse
from pathlib import Path

import panda_md as panda

DATA = Path(__file__).resolve().parents[1] / "tests" / "md" / "data"
WATER_DENSITY_BULK = 33.13  # water oxygens per nm³ in the core of the water phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="save figures here (needs panda-md[plot])")
    arguments = parser.parse_args()

    pore = panda.SlitPore.model_validate_json((DATA / "graphite_pore.json").read_text())
    profiles = panda.ProfileSet.load(DATA / "graphite_water_profiles.npz")
    spec = panda.FitSpec(density_bulk_per_nm3=WATER_DENSITY_BULK)

    print(panda.compare_shapes(pore, profiles.mean(), spec))
    roll = pore.interface("roll")
    result = roll.fit(profiles.mean(), density_bulk_per_nm3=WATER_DENSITY_BULK)
    print(result)
    blocks = panda.fit_blocks(roll, profiles, spec)
    print(blocks)

    if arguments.output is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panda_md.plotting import plot_blocks, plot_fit

        arguments.output.mkdir(parents=True, exist_ok=True)
        figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
        plot_fit(result, left)
        plot_blocks(blocks, "theta", right)
        figure.tight_layout()
        figure.savefig(arguments.output / "graphite_water.png", dpi=150)
        print(f"figure: {arguments.output / 'graphite_water.png'}")


if __name__ == "__main__":
    main()
