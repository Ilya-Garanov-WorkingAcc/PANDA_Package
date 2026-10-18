"""Build examples/test_panda.ipynb from the cells below (edit here; the notebook is a generated file).

uv run --with nbformat python examples/build_test_notebook.py
"""

from pathlib import Path

import nbformat

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown("""
# Full check of PANDA (`panda_md`) on synthetic data and real trajectories

The notebook runs the library the way it is used: from an MD trajectory to the contact angle, the interface shape
and the block error — through the Python API and through the `panda` command line. Every check is recorded in a
table with its expectation and status **PASS / FAIL / SKIP**; the totals are at the end and in
`panda_test_results/report.json`.

| Part | What is checked | Data |
|------|-----------------|------|
| A | synthetic data: parameter recovery for all shapes of the slit, triangle and square; point cloud → profile → fit; stability diagrams; clear errors | none |
| B | graphite slit with water and decane: pore dimensions from the walls, profiles (atoms, molecules, liquid centering), fit, shape comparison, blocks, correlation, 2D map with axial centering, saving, CLI | `graphite_decane_tip4p_1.0` (PANDA_Reborn) |
| C | benzene on the SOLD substrate: a wetting liquid — regime boundary flag | `roughness_2/angles/base` |
| D | calcite, starting frame: wall search for a thick substrate of different atoms | `calcite_decane_tip4p_sc_2.5_offset` |

**How to run.** From the repository root: `uv sync --all-extras`, then `uv run --with jupyter jupyter lab
examples/test_panda.ipynb` (or open it in VS Code with the `.venv` kernel). Data paths are set in the first code
cell or with the `PANDA_REBORN_DIR` and `PANDA_ROUGHNESS_DIR` environment variables; a part without data is marked
SKIP. Time: part A — about a minute, B — 5–10 minutes (depends on `STRIDE`), C — 2–3 minutes, D — seconds.
""")

code("""
# --- Settings: trajectory paths and run parameters ---
import os
from pathlib import Path

REBORN = Path(os.environ.get("PANDA_REBORN_DIR", "/mnt/c/Phystech/Computing_Physics_Center/Projects/PANDA_Reborn"))
ROUGHNESS = Path(
    os.environ.get("PANDA_ROUGHNESS_DIR", "/mnt/c/Phystech/Computing_Physics_Center/Projects/roughness_2")
)

CALCITE = ROUGHNESS / "angles/Dev_PANDA/Dev_PANDA/systems/calcite_decane_tip4p_sc_2.5_offset"

SYSTEMS = {
    "graphite": {
        "topology": REBORN / "graphite_decane_tip4p_1.0" / "graphite_decane_tip4p.tpr",
        "trajectory": REBORN / "graphite_decane_tip4p_1.0" / "graphite_decane_tip4p.xtc",
        "walls": "resname GRAPHITE",
        "water": "resname WATER and name OW",
        "water_molecules": "resname WATER",
        "decane": "resname DECAN*",
        "water_density_bulk": 33.13,  # water oxygens per nm³ — median of the phase core (task 1)
        "start": "2 ns",
        "block": "1 ns",
    },
    "benzene": {
        "topology": ROUGHNESS / "angles" / "base" / "nvt_prod.gro",
        "trajectory": ROUGHNESS / "angles" / "base" / "nvt_prod.xtc",
        "walls": "resname SOLD",
        "liquid": "resname BENZ",
        "start": "4 ns",
        "block": "1.2 ns",
    },
    "calcite": {
        "topology": CALCITE / "cal_dec_tip4p.gro",
        "trajectory": CALCITE / "cal_dec_tip4p.xtc",
        "walls": "resname CAL",
        "water": "resname WATER and name OW",
        "decane": "resname DECAN",
    },
}

STRIDE = 1  # every n-th frame for the main water profile; larger is faster but noisier
EXTRA_STRIDE = 5  # for the additional profiles, maps and short blocks
RESULTS = Path("panda_test_results")
RESULTS.mkdir(exist_ok=True)
""")

code("""
# --- Imports, versions and check bookkeeping ---
import json
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

import panda_md as panda
from panda_md.core.enums import NoiseModel, ProfileSampling
from panda_md.plotting import (
    plot_blocks,
    plot_density_map,
    plot_fit,
    plot_stability,
)

print(f"panda-md {panda.__version__}, Python {sys.version.split()[0]}, numpy {np.__version__}")
try:
    import MDAnalysis

    print(f"MDAnalysis {MDAnalysis.__version__}")
except ImportError:
    print("MDAnalysis is not installed: parts B–D will be skipped (pip install 'panda-md[md]')")

CHECKS: list[dict[str, str]] = []


class Skip(Exception):
    \"\"\"The part is skipped (no data or dependencies).\"\"\"


def check(section: str, name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    CHECKS.append({"part": section, "check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return passed


@contextmanager
def part(section: str) -> Iterator[None]:
    \"\"\"A notebook part: an error inside is a FAIL with the reason, missing data a SKIP; the other parts go on.\"\"\"
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", panda.PandaWarning)
            yield
        for warning in caught:
            print(f"  PANDA warning: {warning.message}")
    except Skip as reason:
        CHECKS.append({"part": section, "check": "whole part", "status": "SKIP", "detail": str(reason)})
        print(f"[SKIP] {section}: {reason}")
    except Exception as error:  # an unexpected error in a part is recorded as a failure
        CHECKS.append({"part": section, "check": "unexpected error", "status": "FAIL", "detail": repr(error)})
        print(f"[FAIL] {section}: {error!r}")
        traceback.print_exc(limit=3)
    finally:
        print(f"--- {section}: {time.perf_counter() - started:.0f} s")


def require_system(name: str) -> dict[str, object]:
    system = SYSTEMS[name]
    missing = [str(system[key]) for key in ("topology", "trajectory") if not Path(system[key]).exists()]
    if missing:
        raise Skip(f"missing files: {', '.join(missing)}")
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        raise Skip("MDAnalysis is needed: pip install 'panda-md[md]'") from None
    return system
""")

markdown("""
## A. Synthetic data: the method recovers what was put into it

The profile is built from the model the way MD would give it (the bin average) and is fitted back with the same
shape. Without noise the parameters must be recovered exactly (θ within 1°; near θ ≈ 180° and for shapes where θ
barely affects the profile more is allowed — this shows in the table).
""")

code("""
SLIT = panda.SlitPore(width_nm=8.0, length_nm=16.0)
DENSITY_BULK = 30.0


def truth_for(model: panda.InterfaceModel, theta: float | None) -> dict[str, float]:
    \"\"\"A point where the shape exists: θ, δ = 0.02 and φ in the middle of the feasible interval.\"\"\"
    values = {"delta": 0.02}
    if "theta" in model.parameters.names:
        values["theta"] = theta
    lower, upper = model.phi_limits(**values, phi=0.5)
    values["phi"] = round(lower + 0.5 * (upper - lower), 4)
    return values


def recover(
    section: str, pore: panda.SlitPore, shape_name: str, theta: float | None, tolerance: float = 1.0
) -> panda.FitResult:
    model = pore.interface(shape_name)
    truth = truth_for(model, theta)
    profile = panda.synthetic_profile(
        model, truth, density_bulk_per_nm3=DENSITY_BULK, sampling=ProfileSampling.BIN_AVERAGE
    )
    result = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    if "theta" in truth:
        error = abs(result.values["theta"] - truth["theta"])
        check(section, f"{pore.name}/{shape_name}: θ recovered", error < tolerance,
              f"truth {truth['theta']}°, fit {result.values['theta']:.2f}°, |Δθ| = {error:.2f}°")
    else:
        error = abs(result.values["phi"] - truth["phi"])
        check(section, f"{pore.name}/{shape_name}: φ recovered", error < 0.01,
              f"truth {truth['phi']}, fit {result.values['phi']:.4f}")
    return result


with part("A1. All slit shapes"):
    for shape, theta in [("droplet", 130.0), ("doughnut", 130.0), ("worm", 130.0), ("roll", 130.0),
                         ("perforation", 130.0), ("layer", None)]:
        recover("A1", SLIT, shape, theta)
""")

code("""
POLYGON_CASES = [  # polygonal pore shapes with a derived profile — points where they exist
    ("square", "droplet_w1c0", {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    ("square", "droplet_w2c0", {"theta": 150.0, "delta": 0.02, "phi": 0.05}),
    ("square", "droplet_w2c1", {"theta": 115.0, "delta": 0.02, "phi": 0.1}),
    ("square", "filament_w1c0", {"theta": 130.0, "delta": 0.02, "phi": 0.1}),
    ("square", "filament_w1c0", {"theta": 60.0, "delta": 0.02, "phi": 0.05}),  # the wetting regime is derived
    ("square", "capsule_w4c0", {"theta": 160.0, "delta": 0.02, "phi": 0.4}),
    ("triangle", "droplet_w1c0", {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    ("triangle", "filament_w3c0", {"theta": 150.0, "delta": 0.02, "phi": 0.5}),
    ("triangle", "capsule_w3c0", {"theta": 150.0, "delta": 0.02, "phi": 0.4}),
]
POLYGONS = {
    "square": panda.SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
    "triangle": panda.TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
}

with part("A2. Triangular and square pores (shapes with a derived profile)"):
    for pore_name, shape_name, truth in POLYGON_CASES:
        model = POLYGONS[pore_name].interface(shape_name)
        profile = panda.synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK,
                                          sampling=ProfileSampling.BIN_AVERAGE)
        result = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
        error = abs(result.values["theta"] - truth["theta"])
        check("A2", f"{pore_name}/{shape_name} at θ = {truth['theta']:g}°: θ recovered", error < 1.5,
              f"fit {result.values['theta']:.2f}°, |Δθ| = {error:.2f}°")
""")
code("""
with part("A3. Noise and shape comparison"):
    roll = SLIT.interface("roll")
    truth = {"theta": 120.0, "delta": 0.03, "phi": 0.4}
    noisy = panda.synthetic_profile(roll, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=0.005, seed=0,
                                    noise_model=NoiseModel.POISSON, sampling=ProfileSampling.BIN_AVERAGE)
    result = roll.fit(noisy, density_bulk_per_nm3=DENSITY_BULK)
    check("A3", "roll with 0.5 % noise: |Δθ| < 1°", abs(result.values["theta"] - 120.0) < 1.0,
          f"θ = {result.values['theta']:.2f}°")
    comparison = panda.compare_shapes(SLIT, noisy, panda.FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    print(comparison)
    check("A3", "roll is the best or in the group of indistinguishable shapes",
          comparison.best.shape_name == "roll" or "roll" in comparison.indistinguishable,
          f"best: {comparison.best.shape_name}; indistinguishable: {comparison.indistinguishable}")
    plot_fit(result)
    plt.show()
""")

code("""
with part("A4. Point cloud → trajectory → wall search → profile → fit"):
    small = panda.SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5)
    for shape, truth in [("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.3}),
                         ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05})]:
        model = small.interface(shape)
        trajectory = panda.synthetic_trajectory(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame_count=40,
                                                seed=1)
        found = panda.SlitPore.from_trajectory(trajectory, walls="walls")
        check("A4", f"{shape}: slit width from the walls", abs(found.width_nm - small.width_nm) < 0.05,
              f"H = {found.width_nm:.3f} nm (truth {small.width_nm})")
        profiles = panda.density_profile(trajectory, found, "liquid", bins=100, block=100.0)
        result = found.interface(shape).fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK)
        check("A4", f"{shape}: θ from the point cloud (|Δθ| < 2°)", abs(result.values["theta"] - truth["theta"]) < 2.0,
              f"θ = {result.values['theta']:.2f}° (truth {truth['theta']}°)")
        if model.shape.symmetry_axis is not None:
            density = panda.density_map(trajectory, found, "liquid", shape=shape, bins=(40, 20))
            plot_density_map(density)
            plt.title(f"{shape}: 2D map, shape axis at x = 0")
            plt.show()
""")

code("""
with part("A5. Stability diagrams"):
    for pore in (panda.SlitPore(width_nm=4.0, length_nm=8.0),
                 panda.SquarePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0),
                 panda.TrianglePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0)):
        diagram = panda.stability_diagram(pore, contact_angle_deg=140.0, wall_layer=0.0,
                                          volume_fractions=np.linspace(0.05, 0.95, 19),
                                          length_ratios=np.linspace(0.5, 4.0, 15))
        winners = sorted({name for row in diagram.stable_shapes for name in row if name})
        check("A5", f"{pore.name}: more than one shape on the diagram", len(winners) > 1, ", ".join(winners))
        plot_stability(diagram)
        plt.title(f"{pore.name}, θ = 140°")
        plt.show()
""")

code("""
def expect_error(
    section: str, name: str, action: Callable[[], object], error_type: type[Exception], fragment: str
) -> bool:
    try:
        action()
    except error_type as error:
        return check(section, name, fragment in str(error), str(error)[:160])
    return check(section, name, False, "no error was raised")


with part("A6. Clear errors instead of a wrong answer"):
    roll = SLIT.interface("roll")
    profile = panda.synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4},
                                      density_bulk_per_nm3=DENSITY_BULK)
    other = panda.SlitPore(width_nm=8.0, length_nm=20.0)
    expect_error("A6", "profile of a pore with another volume", lambda: other.interface("roll").fit(
        profile, density_bulk_per_nm3=DENSITY_BULK), panda.ProfileMismatchError, "pore volume")
    expect_error("A6", "mean normalization without ρ_bulk and φ", lambda: roll.fit(profile, normalization="mean"),
                 panda.InvalidProfileError, "ρ_bulk")
    expect_error("A6", "θ outside the derived regime", lambda: roll.fit(profile, optimize=["delta"],
                 fixed={"theta": 60.0}, density_bulk_per_nm3=DENSITY_BULK), panda.UnsupportedRegimeError, "regime")
    expect_error("A6", "unknown shape", lambda: SLIT.interface("spiral"), panda.PandaError, "available")
    expect_error("A6", "slit diagram at θ < 90°", lambda: panda.stability_diagram(
        SLIT, contact_angle_deg=60.0, wall_layer=0.0, volume_fractions=[0.3], length_ratios=[2.0]),
        panda.UnsupportedRegimeError, "not derived")
""")

markdown("""
## B. Graphite slit: water and decane (TIP4P, 10 ns)

Expectations (manual calculation of task 1 and the `tests/md` tests): H ≈ 8.27 nm; water is a non-wetting bridge
(roll) with θ ≈ 165–180° (171.8° ± 7.1° over blocks in task 1), φ ≈ 0.487; the roll is the best shape. The longest
part comes first — the water profile over all frames after 2 ns (1–2 minutes at `STRIDE = 1`).
""")

code("""
graphite = {}
with part("B1. Trajectory and pore"):
    system = require_system("graphite")
    trajectory = panda.load_trajectory(system["topology"], system["trajectory"])
    times = trajectory.times_ps()
    print(f"{len(times)} frames, {times[0]:g}–{times[-1]:g} ps")
    pore = panda.SlitPore.from_trajectory(trajectory, walls=system["walls"], liquid=system["water"],
                                          time=system["start"])
    print(pore)
    check("B1", "slit width from the graphite atoms", abs(pore.width_nm - 8.269) < 0.01, f"H = {pore.width_nm:.4f} nm")
    graphite.update(system=system, trajectory=trajectory, pore=pore)
""")

code("""
with part("B2. Water block profiles and a roll fit"):
    if not graphite:
        raise Skip("no graphite trajectory (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    water = panda.density_profile(trajectory, pore, system["water"], start=system["start"], block=system["block"],
                                  stride=STRIDE)
    water.save(RESULTS / "graphite_water.npz")
    check("B2", "8 blocks of 1 ns", water.block_count == 8, f"blocks: {water.block_count}")
    mean = water.mean()
    plt.errorbar(mean.coordinate_nm, mean.density_per_nm3, yerr=water.sem(), fmt=".", markersize=3)
    plt.xlabel("z, nm (pore centre at 0)")
    plt.ylabel("ρ of water oxygens, nm⁻³")
    plt.title("Water profile, block mean ± standard error")
    plt.show()
    roll = pore.interface("roll")
    result = roll.fit(mean, density_bulk_per_nm3=system["water_density_bulk"])
    print(result)
    check("B2", "water θ 165–180°", 165.0 < result.values["theta"] <= 180.0, f"θ = {result.values['theta']:.2f}°")
    check("B2", "water φ ≈ 0.487", abs(result.values["phi"] - 0.487) < 0.01, f"φ = {result.values['phi']:.4f}")
    check("B2", "the result records the PANDA version", result.library_version == panda.__version__,
          str(result.library_version))
    plot_fit(result)
    plt.show()
    graphite.update(water=water, result=result)
""")

code("""
with part("B3. Shape comparison and block fits"):
    if "water" not in graphite:
        raise Skip("no water profile (B2)")
    system, pore, water = graphite["system"], graphite["pore"], graphite["water"]
    spec = panda.FitSpec(density_bulk_per_nm3=system["water_density_bulk"])
    comparison = panda.compare_shapes(pore, water.mean(), spec)
    print(comparison)
    check("B3", "the best shape is the roll", comparison.best.shape_name == "roll", comparison.best.shape_name)
    blocks = panda.fit_blocks(pore.interface("roll"), water, spec)
    print(blocks)
    theta_mean, theta_std = blocks.mean("theta"), blocks.std("theta")
    check("B3", "θ over blocks: mean 165–180°, spread < 12°", 165.0 < theta_mean <= 180.0 and theta_std < 12.0,
          f"{theta_mean:.1f}° ± {theta_std:.1f}° (std), ± {blocks.sem('theta'):.1f}° (standard error)")
    plot_blocks(blocks, "theta")
    plt.show()
""")

code("""
with part("B4. Profile variants: per molecule, liquid centering, free shift"):
    if "water" not in graphite:
        raise Skip("no water profile (B2)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    window = dict(start=system["start"], stop="4 ns", stride=EXTRA_STRIDE)
    atoms = panda.density_profile(trajectory, pore, system["water"], **window).mean()
    molecules = panda.density_profile(trajectory, pore, system["water_molecules"], per="molecule", **window).mean()
    difference = np.max(np.abs(atoms.density_per_nm3 - molecules.density_per_nm3)) / atoms.density_per_nm3.max()
    check("B4", "profiles from O atoms and from molecular centres of mass agree (< 5 %)", difference < 0.05,
          f"max |Δρ| / max ρ = {difference:.3f}; N: {atoms.particle_count:g} and {molecules.particle_count:g}")
    liquid = panda.density_profile(trajectory, pore, system["water"], centering="liquid", **window).mean()
    roll = pore.interface("roll")
    by_liquid = roll.fit(liquid, density_bulk_per_nm3=system["water_density_bulk"])
    check("B4", "fit of the liquid-centred profile: θ 160–180°", 160.0 < by_liquid.values["theta"] <= 180.0,
          f"θ = {by_liquid.values['theta']:.2f}°")
    shifted = roll.fit(atoms, optimize=["theta", "delta", "shift"], density_bulk_per_nm3=system["water_density_bulk"])
    check("B4", "the free shift is small (the walls are found correctly)", abs(shifted.values["shift"]) < 0.03,
          f"shift = {shifted.values['shift']:.4f} H, θ = {shifted.values['theta']:.2f}°")
""")

code("""
with part("B5. Correlation time of θ"):
    if not graphite:
        raise Skip("no graphite trajectory (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    short = panda.density_profile(trajectory, pore, system["water"], start=system["start"], block="100 ps",
                                  stride=EXTRA_STRIDE)
    series = panda.fit_blocks(pore.interface("roll"), short,
                              panda.FitSpec(density_bulk_per_nm3=system["water_density_bulk"])).values("theta")
    inefficiency = panda.statistical_inefficiency(series)
    plateau = inefficiency.plateau()
    check("B5", "statistical inefficiency ≥ 1", plateau >= 1.0,
          f"s ≈ {plateau:.1f} blocks of 100 ps → independent θ estimates about every {plateau * 0.1:.1f} ns")
    plt.plot(inefficiency.block_lengths, inefficiency.inefficiency, "o-")
    plt.xlabel("block length, values")
    plt.ylabel("s")
    plt.show()
""")

code("""
with part("B6. 2D water map with axial centering on the roll"):
    if not graphite:
        raise Skip("no graphite trajectory (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    water_map = panda.density_map(trajectory, pore, system["water"], shape="roll", start=system["start"],
                                  stride=EXTRA_STRIDE)
    water_map.save(RESULTS / "graphite_water_map.npz")
    check("B6", "the map is saved and read", panda.DensityMap.load(RESULTS / "graphite_water_map.npz") == water_map)
    core = water_map.density_per_nm3[water_map.density_per_nm3 > 0.5 * water_map.density_per_nm3.max()]
    check("B6", "density in the roll core ≈ ρ_bulk (±15 %)",
          abs(np.median(core) / system["water_density_bulk"] - 1.0) < 0.15,
          f"core median {np.median(core):.1f} nm⁻³, ρ_bulk {system['water_density_bulk']}")
    plot_density_map(water_map)
    plt.title("water: roll axis at x = 0")
    plt.show()
""")

code("""
with part("B7. Decane: wets graphite"):
    if not graphite:
        raise Skip("no graphite trajectory (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    decane = panda.density_profile(trajectory, pore, system["decane"], start=system["start"], block=system["block"],
                                   per="molecule", stride=EXTRA_STRIDE)
    decane_map = panda.density_map(trajectory, pore, system["decane"], start=system["start"], stride=EXTRA_STRIDE,
                                   per="molecule")
    # decane ρ_bulk from the slit middle (|z| < H/4): near the graphite walls decane forms dense adsorption layers
    heights = 0.5 * (decane_map.height_edges_nm[:-1] + decane_map.height_edges_nm[1:])
    middle = decane_map.density_per_nm3[:, np.abs(heights) < 0.25 * pore.width_nm]
    density_bulk = float(np.median(middle[middle > 0.5 * middle.max()]))
    print(f"decane ρ_bulk in the slit middle ≈ {density_bulk:.2f} molecules/nm³ (liquid decane ≈ 3.1)")
    check("B7", "decane ρ_bulk 2.5–3.5 molecules/nm³", 2.5 < density_bulk < 3.5, f"{density_bulk:.2f}")
    result = pore.interface("roll").fit(decane.mean(), density_bulk_per_nm3=density_bulk)
    print(result)
    phi_water = graphite["result"].values["phi"] if "result" in graphite else 0.487
    check("B7", "decane φ + water φ ≈ 1 (the liquids fill the pore)", abs(result.values["phi"] + phi_water - 1.0) < 0.1,
          f"{result.values['phi']:.3f} + {phi_water:.3f} = {result.values['phi'] + phi_water:.3f}")
    check("B7", "decane wets graphite: regime boundary flag (no formulas for θ < 90°)", result.at_regime_boundary,
          f"θ = {result.values['theta']:.2f}°")
    plot_density_map(decane_map)
    plt.title("decane: adsorption layers at the graphite")
    plt.show()
""")
code("""
with part("B8. The command line gives the same answer as the API"):
    if "water" not in graphite:
        raise Skip("no water profile (B2)")
    if shutil.which("panda") is None:
        raise Skip("the panda command is not found in the PATH of this kernel")
    system, pore = graphite["system"], graphite["pore"]
    (RESULTS / "graphite_pore.json").write_text(json.dumps({"type": "slit", **pore.model_dump(mode="json")}))
    command = ["panda", "fit", str(RESULTS / "graphite_water.npz"), "--pore-file", str(RESULTS / "graphite_pore.json"),
               "--shape", "roll", "--rho-bulk", str(system["water_density_bulk"]), "--blocks",
               "-o", str(RESULTS / "graphite_cli.json"), "--plot", str(RESULTS / "graphite_cli_fit.png")]
    completed = subprocess.run(command, capture_output=True, text=True)
    print(completed.stdout[-1500:])
    check("B8", "panda fit: exit code 0 or 2 (2 — flags, e.g. θ near 180° in a block)", completed.returncode in (0, 2),
          f"code {completed.returncode}; {completed.stderr.strip()[:200]}")
    report = json.loads((RESULTS / "graphite_cli.json").read_text())
    cli_theta, api_theta = report["fit"]["values"]["theta"], graphite["result"].values["theta"]
    check("B8", "θ from the CLI = θ from the API", abs(cli_theta - api_theta) < 1e-6,
          f"{cli_theta:.4f}° and {api_theta:.4f}°")
    check("B8", "the CLI report has the fits of all blocks",
          len(report["blocks"]["results"]) == graphite["water"].block_count)
    version = subprocess.run(["panda", "--version"], capture_output=True, text=True).stdout.strip()
    check("B8", "panda --version", panda.__version__ in version, version)
""")

markdown("""
## C. Benzene on the SOLD substrate — a wetting liquid

There are no wetting regime formulas (θ < 90°) for the slit shapes (not derived yet). Expectation: there is more
benzene near the walls than in the middle of the pore; the fit hits the regime boundary and the result carries the
`at_regime_boundary` flag — that is, the library reports “the liquid wets, the angle is not determined” instead of
giving a false angle.
""")

code("""
with part("C. Benzene / SOLD"):
    system = require_system("benzene")
    trajectory = panda.load_trajectory(system["topology"], system["trajectory"])
    pore = panda.SlitPore.from_trajectory(trajectory, walls=system["walls"], liquid=system["liquid"])
    print(pore)
    profiles = panda.density_profile(trajectory, pore, system["liquid"], start=system["start"], block=system["block"],
                                     stride=EXTRA_STRIDE)
    mean = profiles.mean()
    near_wall = mean.density_per_nm3[np.abs(mean.coordinate_nm) > 0.35 * pore.width_nm].max()
    middle = mean.density_per_nm3[np.abs(mean.coordinate_nm) < 0.1 * pore.width_nm].mean()
    check("C", "more benzene near the walls than in the middle", near_wall > middle,
          f"near the walls {near_wall:.2f}, in the middle {middle:.2f} nm⁻³")
    result = pore.interface("roll").fit(mean, density_bulk_per_nm3=15.0)
    print(result)
    check("C", "wetting regime boundary flag", result.at_regime_boundary, f"θ = {result.values['theta']:.2f}°")
    plt.plot(mean.coordinate_nm, mean.density_per_nm3)
    plt.xlabel("z, nm")
    plt.ylabel("ρ of benzene, nm⁻³")
    plt.show()
""")

markdown("""
## D. Calcite, starting frame

The substrate is thick and made of different atoms; the trajectory has one frame. The wall search is checked
(manual calculation: H = 12.684 + 0.009 − 3.554 = 9.139 nm) and that water and decane are entirely inside the pore.
""")

code("""
with part("D. Calcite"):
    system = require_system("calcite")
    trajectory = panda.load_trajectory(system["topology"], system["trajectory"])
    pore = panda.SlitPore.from_trajectory(trajectory, walls=system["walls"])
    check("D", "slit width from the calcite atoms", abs(pore.width_nm - 9.139) < 0.01, f"H = {pore.width_nm:.4f} nm")
    for name in ("water", "decane"):
        profiles = panda.density_profile(trajectory, pore, system[name])
        mean = profiles.mean()
        box = trajectory.frame_at(0.0, [trajectory.select(system[name])]).box_nm
        count = float(np.sum(mean.density_per_nm3 * np.diff(profiles.bin_edges_nm))) * pore.bin_cross_section_nm2(box)
        check("D", f"{name}: all particles in the pore", abs(count / mean.particle_count - 1.0) < 0.02,
              f"∫ρ·A = {count:.0f}, N = {mean.particle_count:g}")
        plt.plot(mean.coordinate_nm, mean.density_per_nm3, label=name)
    plt.legend()
    plt.xlabel("z, nm")
    plt.ylabel("ρ, nm⁻³")
    plt.show()
""")

markdown("## Summary")

code("""
counts = {status: sum(1 for item in CHECKS if item["status"] == status) for status in ("PASS", "FAIL", "SKIP")}
print(f"PASS {counts['PASS']}, FAIL {counts['FAIL']}, SKIP {counts['SKIP']}")
for item in CHECKS:
    if item["status"] != "PASS":
        print(f"  {item['status']} [{item['part']}] {item['check']}: {item['detail']}")
(RESULTS / "report.json").write_text(json.dumps({"panda_version": panda.__version__, "checks": CHECKS},
                                                 ensure_ascii=False, indent=2))
print(f"report: {RESULTS / 'report.json'}")
""")


def build(path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        for kind, source in CELLS
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"cell-{index:02d}"  # stable ids: rebuilding the notebook does not change it
    nbformat.write(notebook, path)


if __name__ == "__main__":
    target = Path(__file__).with_name("test_panda.ipynb")
    build(target)
    print(f"written: {target} ({len(CELLS)} cells)")
