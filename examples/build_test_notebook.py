"""Собирает examples/test_panda.ipynb из ячеек ниже (редактировать здесь, ноутбук — производный файл).

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
# Полная проверка PANDA (`panda_md`) на синтетике и реальных траекториях

Ноутбук прогоняет библиотеку так, как ей пользуются: от траектории МД до угла смачивания, формы интерфейса и
ошибки по блокам — через Python API и через командную строку `panda`. Каждая проверка записывается в таблицу
с ожиданием и статусом **PASS / FAIL / SKIP**; итог — в конце и в `panda_test_results/report.json`.

| Часть | Что проверяется | Данные |
|-------|-----------------|--------|
| A | синтетика: восстановление параметров всех форм щели, треугольника и квадрата; облако точек → профиль → фит; диаграммы устойчивости; понятные ошибки | не нужны |
| B | щель графита с водой и деканом: размеры поры по стенкам, профили (атомы, молекулы, центр по жидкости), фит, сравнение форм, блоки, корреляция, 2D-карта с осевым центрированием, сохранение, CLI | `graphite_decane_tip4p_1.0` (PANDA_Reborn) |
| C | бензол на подложке SOLD: смачивающая жидкость — флаг границы режима | `roughness_2/angles/base` |
| D | кальцит, стартовый кадр: поиск стенок толстой подложки из разных атомов | `calcite_decane_tip4p_sc_2.5_offset` |

**Как запустить.** Из корня репозитория: `uv sync --all-extras`, затем `uv run --with jupyter jupyter lab
examples/test_panda.ipynb` (или открыть в VS Code с ядром `.venv`). Пути к данным — в первой ячейке кода или
через переменные окружения `PANDA_REBORN_DIR` и `PANDA_ROUGHNESS_DIR`; часть без данных помечается SKIP.
Время: часть A — около минуты, B — 5–10 минут (зависит от `STRIDE`), C — 2–3 минуты, D — секунды.
""")

code("""
# --- Настройки: пути к траекториям и параметры прогона ---
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
        "water_density_bulk": 33.13,  # кислородов воды на нм³ — медиана сердцевины фазы (задача 1)
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

STRIDE = 1  # каждый n-й кадр для основного профиля воды; больше — быстрее, но шумнее
EXTRA_STRIDE = 5  # для дополнительных профилей, карт и коротких блоков
RESULTS = Path("panda_test_results")
RESULTS.mkdir(exist_ok=True)
""")

code("""
# --- Импорты, версии и учёт проверок ---
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
    print("MDAnalysis не установлен: части B–D будут пропущены (pip install 'panda-md[md]')")

CHECKS: list[dict[str, str]] = []


class Skip(Exception):
    \"\"\"Часть пропускается (нет данных или зависимостей).\"\"\"


def check(section: str, name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    CHECKS.append({"part": section, "check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return passed


@contextmanager
def part(section: str) -> Iterator[None]:
    \"\"\"Часть ноутбука: ошибка внутри — FAIL с причиной, отсутствие данных — SKIP; остальные части идут дальше.\"\"\"
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", panda.PandaWarning)
            yield
        for warning in caught:
            print(f"  предупреждение PANDA: {warning.message}")
    except Skip as reason:
        CHECKS.append({"part": section, "check": "часть целиком", "status": "SKIP", "detail": str(reason)})
        print(f"[SKIP] {section}: {reason}")
    except Exception as error:  # непредвиденная ошибка части записывается как провал
        CHECKS.append({"part": section, "check": "непредвиденная ошибка", "status": "FAIL", "detail": repr(error)})
        print(f"[FAIL] {section}: {error!r}")
        traceback.print_exc(limit=3)
    finally:
        print(f"--- {section}: {time.perf_counter() - started:.0f} с")


def require_system(name: str) -> dict[str, object]:
    system = SYSTEMS[name]
    missing = [str(system[key]) for key in ("topology", "trajectory") if not Path(system[key]).exists()]
    if missing:
        raise Skip(f"нет файлов: {', '.join(missing)}")
    try:
        import MDAnalysis  # noqa: F401
    except ImportError:
        raise Skip("нужен MDAnalysis: pip install 'panda-md[md]'") from None
    return system
""")

markdown("""
## A. Синтетика: метод восстанавливает то, что в него заложено

Профиль строится по модели так, как его дала бы МД (среднее по бину), и фитируется обратно той же формой.
Без шума параметры должны восстанавливаться точно (θ — до 1°; у θ ≈ 180° и у форм, где θ слабо влияет на
профиль, допускается больше — это видно в таблице).
""")

code("""
SLIT = panda.SlitPore(width_nm=8.0, length_nm=16.0)
DENSITY_BULK = 30.0


def truth_for(model: panda.InterfaceModel, theta: float | None) -> dict[str, float]:
    \"\"\"Точка, в которой форма существует: θ, δ = 0.02 и φ в середине допустимого интервала.\"\"\"
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
        check(section, f"{pore.name}/{shape_name}: θ восстановлен", error < tolerance,
              f"истина {truth['theta']}°, фит {result.values['theta']:.2f}°, |Δθ| = {error:.2f}°")
    else:
        error = abs(result.values["phi"] - truth["phi"])
        check(section, f"{pore.name}/{shape_name}: φ восстановлена", error < 0.01,
              f"истина {truth['phi']}, фит {result.values['phi']:.4f}")
    return result


with part("A1. Все формы щели"):
    for shape, theta in [("droplet", 130.0), ("doughnut", 130.0), ("worm", 130.0), ("roll", 130.0),
                         ("perforation", 130.0), ("layer", None)]:
        recover("A1", SLIT, shape, theta)
""")

code("""
POLYGON_CASES = [  # формы многоугольных пор с выведенным профилем — точки, где они существуют
    ("square", "droplet_w1c0", {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    ("square", "droplet_w2c0", {"theta": 150.0, "delta": 0.02, "phi": 0.05}),
    ("square", "droplet_w2c1", {"theta": 115.0, "delta": 0.02, "phi": 0.1}),
    ("square", "filament_w1c0", {"theta": 130.0, "delta": 0.02, "phi": 0.1}),
    ("square", "filament_w1c0", {"theta": 60.0, "delta": 0.02, "phi": 0.05}),  # смачивающий режим выведен
    ("square", "capsule_w4c0", {"theta": 160.0, "delta": 0.02, "phi": 0.4}),
    ("triangle", "droplet_w1c0", {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    ("triangle", "filament_w3c0", {"theta": 150.0, "delta": 0.02, "phi": 0.5}),
    ("triangle", "capsule_w3c0", {"theta": 150.0, "delta": 0.02, "phi": 0.4}),
]
POLYGONS = {
    "square": panda.SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
    "triangle": panda.TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
}

with part("A2. Треугольная и квадратная поры (формы с выведенным профилем)"):
    for pore_name, shape_name, truth in POLYGON_CASES:
        model = POLYGONS[pore_name].interface(shape_name)
        profile = panda.synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK,
                                          sampling=ProfileSampling.BIN_AVERAGE)
        result = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
        error = abs(result.values["theta"] - truth["theta"])
        check("A2", f"{pore_name}/{shape_name} при θ = {truth['theta']:g}°: θ восстановлен", error < 1.5,
              f"фит {result.values['theta']:.2f}°, |Δθ| = {error:.2f}°")
""")
code("""
with part("A3. Шум и сравнение форм"):
    roll = SLIT.interface("roll")
    truth = {"theta": 120.0, "delta": 0.03, "phi": 0.4}
    noisy = panda.synthetic_profile(roll, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=0.005, seed=0,
                                    noise_model=NoiseModel.POISSON, sampling=ProfileSampling.BIN_AVERAGE)
    result = roll.fit(noisy, density_bulk_per_nm3=DENSITY_BULK)
    check("A3", "валик при шуме 0.5 %: |Δθ| < 1°", abs(result.values["theta"] - 120.0) < 1.0,
          f"θ = {result.values['theta']:.2f}°")
    comparison = panda.compare_shapes(SLIT, noisy, panda.FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    print(comparison)
    check("A3", "валик — лучший или в группе неразличимых форм",
          comparison.best.shape_name == "roll" or "roll" in comparison.indistinguishable,
          f"лучший: {comparison.best.shape_name}; неразличимы: {comparison.indistinguishable}")
    plot_fit(result)
    plt.show()
""")

code("""
with part("A4. Облако точек → траектория → поиск стенок → профиль → фит"):
    small = panda.SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5)
    for shape, truth in [("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.3}),
                         ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05})]:
        model = small.interface(shape)
        trajectory = panda.synthetic_trajectory(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame_count=40,
                                                seed=1)
        found = panda.SlitPore.from_trajectory(trajectory, walls="walls")
        check("A4", f"{shape}: ширина щели по стенкам", abs(found.width_nm - small.width_nm) < 0.05,
              f"H = {found.width_nm:.3f} нм (истина {small.width_nm})")
        profiles = panda.density_profile(trajectory, found, "liquid", bins=100, block=100.0)
        result = found.interface(shape).fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK)
        check("A4", f"{shape}: θ из облака точек (|Δθ| < 2°)", abs(result.values["theta"] - truth["theta"]) < 2.0,
              f"θ = {result.values['theta']:.2f}° (истина {truth['theta']}°)")
        if model.shape.symmetry_axis is not None:
            density = panda.density_map(trajectory, found, "liquid", shape=shape, bins=(40, 20))
            plot_density_map(density)
            plt.title(f"{shape}: 2D-карта, ось формы в x = 0")
            plt.show()
""")

code("""
with part("A5. Диаграммы устойчивости"):
    for pore in (panda.SlitPore(width_nm=4.0, length_nm=8.0),
                 panda.SquarePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0),
                 panda.TrianglePore(side_nm=4.0, length_nm=8.0, layer_width_nm=4.0)):
        diagram = panda.stability_diagram(pore, contact_angle_deg=140.0, wall_layer=0.0,
                                          volume_fractions=np.linspace(0.05, 0.95, 19),
                                          length_ratios=np.linspace(0.5, 4.0, 15))
        winners = sorted({name for row in diagram.stable_shapes for name in row if name})
        check("A5", f"{pore.name}: на диаграмме больше одной формы", len(winners) > 1, ", ".join(winners))
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
    return check(section, name, False, "ошибки не было")


with part("A6. Понятные ошибки вместо неверного ответа"):
    roll = SLIT.interface("roll")
    profile = panda.synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4},
                                      density_bulk_per_nm3=DENSITY_BULK)
    other = panda.SlitPore(width_nm=8.0, length_nm=20.0)
    expect_error("A6", "профиль поры другого объёма", lambda: other.interface("roll").fit(
        profile, density_bulk_per_nm3=DENSITY_BULK), panda.ProfileMismatchError, "объём")
    expect_error("A6", "нормировка mean без ρ_bulk и φ", lambda: roll.fit(profile, normalization="mean"),
                 panda.InvalidProfileError, "ρ_bulk")
    expect_error("A6", "θ вне выведенного режима", lambda: roll.fit(profile, optimize=["delta"],
                 fixed={"theta": 60.0}, density_bulk_per_nm3=DENSITY_BULK), panda.UnsupportedRegimeError, "режим")
    expect_error("A6", "неизвестная форма", lambda: SLIT.interface("spiral"), panda.PandaError, "доступны")
    expect_error("A6", "диаграмма щели при θ < 90°", lambda: panda.stability_diagram(
        SLIT, contact_angle_deg=60.0, wall_layer=0.0, volume_fractions=[0.3], length_ratios=[2.0]),
        panda.UnsupportedRegimeError, "не выведены")
""")

markdown("""
## B. Щель графита: вода и декан (TIP4P, 10 нс)

Ожидания (ручной расчёт задачи 1 и тесты `tests/md`): H ≈ 8.27 нм; вода — несмачивающая перемычка-валик с
θ ≈ 165–180° (по блокам 171.8° ± 7.1° в задаче 1), φ ≈ 0.487; валик — лучшая форма. Сначала самая долгая часть —
профиль воды по всем кадрам после 2 нс (1–2 минуты при `STRIDE = 1`).
""")

code("""
graphite = {}
with part("B1. Траектория и пора"):
    system = require_system("graphite")
    trajectory = panda.load_trajectory(system["topology"], system["trajectory"])
    times = trajectory.times_ps()
    print(f"{len(times)} кадров, {times[0]:g}–{times[-1]:g} пс")
    pore = panda.SlitPore.from_trajectory(trajectory, walls=system["walls"], liquid=system["water"],
                                          time=system["start"])
    print(pore)
    check("B1", "ширина щели по атомам графита", abs(pore.width_nm - 8.269) < 0.01, f"H = {pore.width_nm:.4f} нм")
    graphite.update(system=system, trajectory=trajectory, pore=pore)
""")

code("""
with part("B2. Профиль воды по блокам и фит валиком"):
    if not graphite:
        raise Skip("нет траектории графита (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    water = panda.density_profile(trajectory, pore, system["water"], start=system["start"], block=system["block"],
                                  stride=STRIDE)
    water.save(RESULTS / "graphite_water.npz")
    check("B2", "8 блоков по 1 нс", water.block_count == 8, f"блоков: {water.block_count}")
    mean = water.mean()
    plt.errorbar(mean.coordinate_nm, mean.density_per_nm3, yerr=water.sem(), fmt=".", markersize=3)
    plt.xlabel("z, нм (центр поры в 0)")
    plt.ylabel("ρ кислородов воды, нм⁻³")
    plt.title("Профиль воды, среднее по блокам ± ошибка среднего")
    plt.show()
    roll = pore.interface("roll")
    result = roll.fit(mean, density_bulk_per_nm3=system["water_density_bulk"])
    print(result)
    check("B2", "θ воды 165–180°", 165.0 < result.values["theta"] <= 180.0, f"θ = {result.values['theta']:.2f}°")
    check("B2", "φ воды ≈ 0.487", abs(result.values["phi"] - 0.487) < 0.01, f"φ = {result.values['phi']:.4f}")
    check("B2", "в результате записана версия PANDA", result.library_version == panda.__version__,
          str(result.library_version))
    plot_fit(result)
    plt.show()
    graphite.update(water=water, result=result)
""")

code("""
with part("B3. Сравнение форм и фит по блокам"):
    if "water" not in graphite:
        raise Skip("нет профиля воды (B2)")
    system, pore, water = graphite["system"], graphite["pore"], graphite["water"]
    spec = panda.FitSpec(density_bulk_per_nm3=system["water_density_bulk"])
    comparison = panda.compare_shapes(pore, water.mean(), spec)
    print(comparison)
    check("B3", "лучшая форма — валик", comparison.best.shape_name == "roll", comparison.best.shape_name)
    blocks = panda.fit_blocks(pore.interface("roll"), water, spec)
    print(blocks)
    theta_mean, theta_std = blocks.mean("theta"), blocks.std("theta")
    check("B3", "θ по блокам: среднее 165–180°, разброс < 12°", 165.0 < theta_mean <= 180.0 and theta_std < 12.0,
          f"{theta_mean:.1f}° ± {theta_std:.1f}° (std), ± {blocks.sem('theta'):.1f}° (ошибка среднего)")
    plot_blocks(blocks, "theta")
    plt.show()
""")

code("""
with part("B4. Варианты профиля: по молекулам, центр по жидкости, свободный shift"):
    if "water" not in graphite:
        raise Skip("нет профиля воды (B2)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    window = dict(start=system["start"], stop="4 ns", stride=EXTRA_STRIDE)
    atoms = panda.density_profile(trajectory, pore, system["water"], **window).mean()
    molecules = panda.density_profile(trajectory, pore, system["water_molecules"], per="molecule", **window).mean()
    difference = np.max(np.abs(atoms.density_per_nm3 - molecules.density_per_nm3)) / atoms.density_per_nm3.max()
    check("B4", "профили по атомам O и по центрам масс молекул совпадают (< 5 %)", difference < 0.05,
          f"max |Δρ| / max ρ = {difference:.3f}; N: {atoms.particle_count:g} и {molecules.particle_count:g}")
    liquid = panda.density_profile(trajectory, pore, system["water"], centering="liquid", **window).mean()
    roll = pore.interface("roll")
    by_liquid = roll.fit(liquid, density_bulk_per_nm3=system["water_density_bulk"])
    check("B4", "фит профиля в системе жидкости: θ 160–180°", 160.0 < by_liquid.values["theta"] <= 180.0,
          f"θ = {by_liquid.values['theta']:.2f}°")
    shifted = roll.fit(atoms, optimize=["theta", "delta", "shift"], density_bulk_per_nm3=system["water_density_bulk"])
    check("B4", "свободный shift мал (стенки найдены верно)", abs(shifted.values["shift"]) < 0.03,
          f"shift = {shifted.values['shift']:.4f} H, θ = {shifted.values['theta']:.2f}°")
""")

code("""
with part("B5. Время корреляции θ"):
    if not graphite:
        raise Skip("нет траектории графита (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    short = panda.density_profile(trajectory, pore, system["water"], start=system["start"], block="100 ps",
                                  stride=EXTRA_STRIDE)
    series = panda.fit_blocks(pore.interface("roll"), short,
                              panda.FitSpec(density_bulk_per_nm3=system["water_density_bulk"])).values("theta")
    inefficiency = panda.statistical_inefficiency(series)
    plateau = inefficiency.plateau()
    check("B5", "статистическая неэффективность ≥ 1", plateau >= 1.0,
          f"s ≈ {plateau:.1f} блоков по 100 пс → независимые оценки θ примерно через {plateau * 0.1:.1f} нс")
    plt.plot(inefficiency.block_lengths, inefficiency.inefficiency, "o-")
    plt.xlabel("длина блока, значений")
    plt.ylabel("s")
    plt.show()
""")

code("""
with part("B6. 2D-карта воды с осевым центрированием валика"):
    if not graphite:
        raise Skip("нет траектории графита (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    water_map = panda.density_map(trajectory, pore, system["water"], shape="roll", start=system["start"],
                                  stride=EXTRA_STRIDE)
    water_map.save(RESULTS / "graphite_water_map.npz")
    check("B6", "карта сохраняется и читается", panda.DensityMap.load(RESULTS / "graphite_water_map.npz") == water_map)
    core = water_map.density_per_nm3[water_map.density_per_nm3 > 0.5 * water_map.density_per_nm3.max()]
    check("B6", "плотность в сердцевине валика ≈ ρ_bulk (±15 %)",
          abs(np.median(core) / system["water_density_bulk"] - 1.0) < 0.15,
          f"медиана сердцевины {np.median(core):.1f} нм⁻³, ρ_bulk {system['water_density_bulk']}")
    plot_density_map(water_map)
    plt.title("вода: ось валика в x = 0")
    plt.show()
""")

code("""
with part("B7. Декан: смачивает графит"):
    if not graphite:
        raise Skip("нет траектории графита (B1)")
    system, trajectory, pore = graphite["system"], graphite["trajectory"], graphite["pore"]
    decane = panda.density_profile(trajectory, pore, system["decane"], start=system["start"], block=system["block"],
                                   per="molecule", stride=EXTRA_STRIDE)
    decane_map = panda.density_map(trajectory, pore, system["decane"], start=system["start"], stride=EXTRA_STRIDE,
                                   per="molecule")
    # ρ_bulk декана — по середине щели (|z| < H/4): у стенок графита декан образует плотные адсорбционные слои
    heights = 0.5 * (decane_map.height_edges_nm[:-1] + decane_map.height_edges_nm[1:])
    middle = decane_map.density_per_nm3[:, np.abs(heights) < 0.25 * pore.width_nm]
    density_bulk = float(np.median(middle[middle > 0.5 * middle.max()]))
    print(f"ρ_bulk декана по середине щели ≈ {density_bulk:.2f} молекул/нм³ (жидкий декан ≈ 3.1)")
    check("B7", "ρ_bulk декана 2.5–3.5 молекул/нм³", 2.5 < density_bulk < 3.5, f"{density_bulk:.2f}")
    result = pore.interface("roll").fit(decane.mean(), density_bulk_per_nm3=density_bulk)
    print(result)
    phi_water = graphite["result"].values["phi"] if "result" in graphite else 0.487
    check("B7", "φ декана + φ воды ≈ 1 (жидкости заполняют пору)", abs(result.values["phi"] + phi_water - 1.0) < 0.1,
          f"{result.values['phi']:.3f} + {phi_water:.3f} = {result.values['phi'] + phi_water:.3f}")
    check("B7", "декан смачивает графит: флаг границы режима (формул θ < 90° нет)", result.at_regime_boundary,
          f"θ = {result.values['theta']:.2f}°")
    plot_density_map(decane_map)
    plt.title("декан: адсорбционные слои у графита")
    plt.show()
""")
code("""
with part("B8. Командная строка дает тот же ответ, что API"):
    if "water" not in graphite:
        raise Skip("нет профиля воды (B2)")
    if shutil.which("panda") is None:
        raise Skip("команда panda не найдена в PATH этого ядра")
    system, pore = graphite["system"], graphite["pore"]
    (RESULTS / "graphite_pore.json").write_text(json.dumps({"type": "slit", **pore.model_dump(mode="json")}))
    command = ["panda", "fit", str(RESULTS / "graphite_water.npz"), "--pore-file", str(RESULTS / "graphite_pore.json"),
               "--shape", "roll", "--rho-bulk", str(system["water_density_bulk"]), "--blocks",
               "-o", str(RESULTS / "graphite_cli.json"), "--plot", str(RESULTS / "graphite_cli_fit.png")]
    completed = subprocess.run(command, capture_output=True, text=True)
    print(completed.stdout[-1500:])
    check("B8", "panda fit: код 0 или 2 (2 — флаги, например θ у 180° в блоке)", completed.returncode in (0, 2),
          f"код {completed.returncode}; {completed.stderr.strip()[:200]}")
    report = json.loads((RESULTS / "graphite_cli.json").read_text())
    cli_theta, api_theta = report["fit"]["values"]["theta"], graphite["result"].values["theta"]
    check("B8", "θ из CLI = θ из API", abs(cli_theta - api_theta) < 1e-6, f"{cli_theta:.4f}° и {api_theta:.4f}°")
    check("B8", "в отчёте CLI — фиты всех блоков", len(report["blocks"]["results"]) == graphite["water"].block_count)
    version = subprocess.run(["panda", "--version"], capture_output=True, text=True).stdout.strip()
    check("B8", "panda --version", panda.__version__ in version, version)
""")

markdown("""
## C. Бензол на подложке SOLD — смачивающая жидкость

Формул смачивающего режима (θ < 90°) для форм щели нет (открытый пункт O1). Ожидание: у стенок бензола больше,
чем в середине поры; фит упирается в границу режима, и результат несёт флаг `at_regime_boundary` — то есть
библиотека сообщает «жидкость смачивает, угол не определён», а не выдаёт ложный угол.
""")

code("""
with part("C. Бензол / SOLD"):
    system = require_system("benzene")
    trajectory = panda.load_trajectory(system["topology"], system["trajectory"])
    pore = panda.SlitPore.from_trajectory(trajectory, walls=system["walls"], liquid=system["liquid"])
    print(pore)
    profiles = panda.density_profile(trajectory, pore, system["liquid"], start=system["start"], block=system["block"],
                                     stride=EXTRA_STRIDE)
    mean = profiles.mean()
    near_wall = mean.density_per_nm3[np.abs(mean.coordinate_nm) > 0.35 * pore.width_nm].max()
    middle = mean.density_per_nm3[np.abs(mean.coordinate_nm) < 0.1 * pore.width_nm].mean()
    check("C", "у стенок бензола больше, чем в середине", near_wall > middle,
          f"у стенок {near_wall:.2f}, в середине {middle:.2f} нм⁻³")
    result = pore.interface("roll").fit(mean, density_bulk_per_nm3=15.0)
    print(result)
    check("C", "флаг границы режима смачивания", result.at_regime_boundary, f"θ = {result.values['theta']:.2f}°")
    plt.plot(mean.coordinate_nm, mean.density_per_nm3)
    plt.xlabel("z, нм")
    plt.ylabel("ρ бензола, нм⁻³")
    plt.show()
""")

markdown("""
## D. Кальцит, стартовый кадр

Подложка толстая и состоит из разных атомов; в траектории один кадр. Проверяется поиск стенок (ручной расчёт:
H = 12.684 + 0.009 − 3.554 = 9.139 нм) и то, что вода и декан целиком внутри поры.
""")

code("""
with part("D. Кальцит"):
    system = require_system("calcite")
    trajectory = panda.load_trajectory(system["topology"], system["trajectory"])
    pore = panda.SlitPore.from_trajectory(trajectory, walls=system["walls"])
    check("D", "ширина щели по атомам кальцита", abs(pore.width_nm - 9.139) < 0.01, f"H = {pore.width_nm:.4f} нм")
    for name in ("water", "decane"):
        profiles = panda.density_profile(trajectory, pore, system[name])
        mean = profiles.mean()
        box = trajectory.frame_at(0.0, [trajectory.select(system[name])]).box_nm
        count = float(np.sum(mean.density_per_nm3 * np.diff(profiles.bin_edges_nm))) * pore.bin_cross_section_nm2(box)
        check("D", f"{name}: все частицы в поре", abs(count / mean.particle_count - 1.0) < 0.02,
              f"∫ρ·A = {count:.0f}, N = {mean.particle_count:g}")
        plt.plot(mean.coordinate_nm, mean.density_per_nm3, label=name)
    plt.legend()
    plt.xlabel("z, нм")
    plt.ylabel("ρ, нм⁻³")
    plt.show()
""")

markdown("## Итог")

code("""
counts = {status: sum(1 for item in CHECKS if item["status"] == status) for status in ("PASS", "FAIL", "SKIP")}
print(f"PASS {counts['PASS']}, FAIL {counts['FAIL']}, SKIP {counts['SKIP']}")
for item in CHECKS:
    if item["status"] != "PASS":
        print(f"  {item['status']} [{item['part']}] {item['check']}: {item['detail']}")
(RESULTS / "report.json").write_text(json.dumps({"panda_version": panda.__version__, "checks": CHECKS},
                                                 ensure_ascii=False, indent=2))
print(f"отчёт: {RESULTS / 'report.json'}")
""")


def build(path: Path) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
        for kind, source in CELLS
    ]
    nbformat.write(notebook, path)


if __name__ == "__main__":
    target = Path(__file__).with_name("test_panda.ipynb")
    build(target)
    print(f"записано: {target} ({len(CELLS)} ячеек)")
