"""Точность метода на синтетике — рисунки и таблицы для документации и статьи.

Запуск из корня репозитория (нужен panda-md[plot]):
    uv run python validation/run_accuracy.py
Результат: validation/figures/*.png и validation/REPORT.md (таблицы и выводы пересобираются при каждом запуске).
Время — несколько минут: каждое значение условия — 10 зёрен шума, каждый фит — мультистарт.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: только файлы

import matplotlib.pyplot as plt

from panda_md import Frame, SlitPore
from panda_md.analysis.accuracy import AccuracyCase, AccuracyStudy, accuracy_study
from panda_md.plotting import plot_accuracy

HERE = Path(__file__).parent
FIGURES = HERE / "figures"
REPORT = HERE / "REPORT.md"
DENSITY_BULK = 30.0
SEEDS = range(10)
PORE = SlitPore(width_nm=8.0, length_nm=16.0)
TRUTHS = {
    "roll": {"theta": 120.0, "delta": 0.03, "phi": 0.4},
    "droplet": {"theta": 130.0, "delta": 0.03, "phi": 0.05},
    "worm": {"theta": 130.0, "delta": 0.03, "phi": 0.1},
}
NOISE_LEVELS = (0.0025, 0.005, 0.01, 0.02)
BIN_COUNTS = (50, 100, 200, 400)
ANGLES = (100.0, 120.0, 140.0, 160.0, 170.0, 175.0)
BASE_NOISE = 0.005
WALL_ERROR = 0.05  # ошибка положения стенок — 5 % ширины щели (O3)


def noise_study(shape: str) -> AccuracyStudy:
    cases = {
        f"{100 * noise:g}": (100 * noise, AccuracyCase(truth=TRUTHS[shape], noise_std=noise)) for noise in NOISE_LEVELS
    }
    return accuracy_study(
        PORE.interface(shape), "шум, % ρ_bulk на бин", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS
    )


def bins_study(shape: str) -> AccuracyStudy:
    cases = {
        f"{bins}": (float(bins), AccuracyCase(truth=TRUTHS[shape], noise_std=BASE_NOISE, bin_count=bins))
        for bins in BIN_COUNTS
    }
    return accuracy_study(PORE.interface(shape), "бинов", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS)


def angle_study(shape: str) -> AccuracyStudy:
    cases = {}
    for angle in ANGLES:
        truth = {**TRUTHS[shape], "theta": angle}
        if not PORE.interface(shape).is_feasible(**truth):
            continue
        cases[f"{angle:g}"] = (angle, AccuracyCase(truth=truth, noise_std=BASE_NOISE))
    return accuracy_study(PORE.interface(shape), "θ, °", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS)


def centering_study(shape: str) -> AccuracyStudy:
    truth = TRUTHS[shape]
    cases = {
        "поре": (0.0, AccuracyCase(truth=truth, noise_std=BASE_NOISE, offset=WALL_ERROR)),
        "поре + shift": (
            1.0,
            AccuracyCase(truth=truth, noise_std=BASE_NOISE, offset=WALL_ERROR, optimize=("theta", "delta", "shift")),
        ),
        "жидкости, δ известна": (
            2.0,
            AccuracyCase(
                truth=truth,
                noise_std=BASE_NOISE,
                offset=WALL_ERROR,
                frame=Frame.LIQUID,
                optimize=("theta",),
                fixed={"delta": truth["delta"]},
            ),
        ),
    }
    return accuracy_study(
        PORE.interface(shape), "центрирование по", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS
    )


def save_figure(study: AccuracyStudy, name: str) -> str:
    figure, axes = plt.subplots(figsize=(5.0, 3.6))
    plot_accuracy(study, axes)
    figure.tight_layout()
    path = FIGURES / f"{name}.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path.name


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    started = time.perf_counter()
    sections = [
        "# Точность метода на синтетике\n",
        f"Сгенерировано `validation/run_accuracy.py`. Щель {PORE.width_nm:g} × {PORE.length_nm:g} нм, ρ_bulk = "
        f"{DENSITY_BULK:g} нм⁻³, {len(SEEDS)} зёрен шума на точку, фит — настройки по умолчанию (θ и δ свободны, "
        f"φ из ∫ρ). Шум задан для 200 бинов и пересчитывается как √(бинов / 200). Δθ = восстановленный − истинный.\n",
    ]
    studies = [
        ("Шум", "noise", noise_study, ("roll", "droplet", "worm")),
        ("Число бинов (шум 0.5 % при 200 бинах)", "bins", bins_study, ("roll", "droplet")),
        ("Угол (шум 0.5 %): точность у 180° — O11", "angle", angle_study, ("roll", "droplet", "worm")),
        ("Центрирование при ошибке положения стенок 5 % H — O3", "centering", centering_study, ("roll", "droplet")),
    ]
    for title, key, build, shapes in studies:
        sections.append(f"\n## {title}\n")
        for shape in shapes:
            study = build(shape)
            figure = save_figure(study, f"{key}_{shape}")
            sections.append(
                f"\n### {shape} (истинные {TRUTHS[shape]})\n\n{study.table()}\n\n![{shape}](figures/{figure})\n"
            )
            print(f"{key}/{shape}: {time.perf_counter() - started:.0f} с", flush=True)
    REPORT.write_text("".join(sections))
    print(f"записано: {REPORT}")


if __name__ == "__main__":
    main()
