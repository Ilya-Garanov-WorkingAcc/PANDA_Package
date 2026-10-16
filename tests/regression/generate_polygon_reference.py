"""Эталоны треугольной и квадратной пор из PANDA_Polygon (`docs/model/{triangle,square}.py`, коммит 38ff491).

Запуск из корня репозитория (нужна локальная копия PANDA_Polygon/, см. ROADMAP «Фаза 4а»):
    uv run python tests/regression/generate_polygon_reference.py
Результат: tests/regression/data/{triangle,square}_reference.json — коммитятся, тесты читают только их.

Для каждой формы записываются условие существования, энергия S_LV − cos θ · S_SL и сумма площадей S_LV + S_SL;
профиль — только у форм, для которых он выведен в источнике. Формы с дефектами источника — девять
форм квадрата, droplet_w2c1 и capsule_w3c3 треугольника (docs/architecture/05_polygon_pores.md §4) — не переносятся
и в эталон не входят; диаграммы источника строятся по тому же набору форм.
"""

import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
from pydantic import BaseModel, ConfigDict

SOURCE_DIRECTORY = Path(__file__).parents[2] / "PANDA_Polygon" / "docs" / "model"
OUTPUT_DIRECTORY = Path(__file__).parent / "data"
SOURCE_COMMIT = "github.com/Vandyshge/PANDA_Polygon@38ff491"
BIN_COUNT = 200  # координата — центры бинов: ни одна точка не лежит на стенке
LAYER_WIDTH_RATIO = 1.0
LENGTH_RATIOS = (1.0, 2.0, 4.0, 8.0)
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9)
UNUSED_ANGLE_RAD = math.pi  # форме без θ источник всё равно принимает угол, но не использует его
FIXED_VOLUME_SCALES = (1.0, 1.01, 0.98)  # droplet_w3c0: φ ровно по объёму, внутри и вне допуска 1.5 %
STABILITY_POINTS = 30  # сетка диаграммы устойчивости: φ и l — как в computePhase (docs/model-runtime.js)
STABILITY_MAX_LENGTH_RATIO = 5.0
STABILITY_WALL_LAYERS = (0.0, 0.03)


class PolygonReferenceSetup(BaseModel):
    """Что и на какой сетке снимать с одного модуля источника."""

    model_config = ConfigDict(frozen=True)

    pore_name: str
    coordinate_bounds: tuple[float, float]
    angles_deg: tuple[float, ...]
    stability_angles_deg: tuple[float, ...]
    shapes: tuple[str, ...]
    shapes_without_angle: tuple[str, ...]


TRIANGLE_HEIGHT = math.sqrt(3.0) / 2.0
SETUPS = (
    PolygonReferenceSetup(
        pore_name="triangle",
        coordinate_bounds=(-TRIANGLE_HEIGHT / 3.0, 2.0 * TRIANGLE_HEIGHT / 3.0),
        angles_deg=(60.0, 100.0, 110.0, 125.0, 150.0, 175.0),
        stability_angles_deg=(100.0, 115.0, 125.0, 150.0, 180.0),
        shapes=(
            "filament_w0c0",
            "filament_w0c0_straight",
            "filament_w1c0",
            "filament_w2c0",
            "filament_w2c1",
            "filament_w3c0",
            "filament_w3c1",
            "filament_w3c2",
            "droplet_w1c0",
            "droplet_w2c0",
            "droplet_w3c0",
            "capsule_w3c0",
        ),
        shapes_without_angle=("filament_w0c0", "filament_w0c0_straight"),
    ),
    PolygonReferenceSetup(
        pore_name="square",
        coordinate_bounds=(-0.5, 0.5),
        # 135° не берём: у droplet_w2c1 условие источника допускает θ = 3π/4, а функция площади — нет
        angles_deg=(60.0, 100.0, 120.0, 140.0, 150.0, 175.0),
        stability_angles_deg=(100.0, 120.0, 140.0, 160.0, 180.0),
        shapes=(
            "filament_w0c0",
            "filament_w0c0_straight",
            "filament_w1c0",
            "droplet_w1c0",
            "droplet_w2c0",
            "droplet_w2c1",
            "capsule_w4c0",
        ),
        shapes_without_angle=("filament_w0c0", "filament_w0c0_straight"),
    ),
)


def load_source(pore_name: str) -> ModuleType:
    path = SOURCE_DIRECTORY / f"{pore_name}.py"
    spec = importlib.util.spec_from_file_location(f"panda_polygon_{pore_name}", path)
    assert spec is not None and spec.loader is not None, f"нет файла {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bin_centers(bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    width = (upper - lower) / BIN_COUNT
    return np.linspace(lower + width / 2.0, upper - width / 2.0, BIN_COUNT)


def fixed_volume_fractions(length_ratio: float, angle: float, wall_layer: float) -> list[float]:
    """φ, при которых droplet_w3c0 треугольника существует: объём капли задан сечением (cond_droplet_w3c0)."""
    side = 1.0 - 2.0 * math.sqrt(3.0) * wall_layer
    cosine = math.cos(angle)
    if cosine >= 0.0:
        return []
    radius = -side / (2.0 * math.sqrt(3.0) * cosine)
    volume = math.pi * radius**3 / 3.0 * (3.0 * cosine**3 - 9.0 * cosine - 2.0)
    exact = volume / (math.sqrt(3.0) / 4.0 * length_ratio)
    return [exact * scale for scale in FIXED_VOLUME_SCALES if 0.0 < exact * scale < 1.0]


def shape_cases(source: ModuleType, setup: PolygonReferenceSetup, shape_name: str) -> list[dict[str, object]]:
    suffix = setup.pore_name.upper()
    condition = getattr(source, f"COND_{suffix}")[shape_name]
    area = getattr(source, f"AREA_{suffix}")[shape_name]
    density = getattr(source, f"RHO_{suffix}").get(shape_name)
    coordinate = bin_centers(setup.coordinate_bounds)
    without_angle = shape_name in setup.shapes_without_angle
    angles_deg: tuple[float | None, ...] = (None,) if without_angle else setup.angles_deg
    cases: list[dict[str, object]] = []
    for length_ratio, angle_deg, wall_layer in itertools.product(LENGTH_RATIOS, angles_deg, WALL_LAYERS):
        angle = UNUSED_ANGLE_RAD if angle_deg is None else math.radians(angle_deg)
        fractions = list(VOLUME_FRACTIONS)
        if shape_name == "droplet_w3c0":
            fractions += fixed_volume_fractions(length_ratio, angle, wall_layer)
        for phi in fractions:
            feasible = bool(condition(length_ratio, phi, angle, wall_layer))
            case: dict[str, object] = {"length_ratio": length_ratio, "delta": wall_layer, "phi": phi}
            if angle_deg is not None:
                case["theta"] = angle_deg
            case["feasible"] = feasible
            if feasible:
                case["energy"] = float(area(length_ratio, phi, angle, wall_layer, flag=False))
                case["area_sum"] = float(area(length_ratio, phi, angle, wall_layer, flag=True))
                if density is not None:
                    profile = density(coordinate, length_ratio, LAYER_WIDTH_RATIO, phi, angle, wall_layer)
                    case["profile"] = np.asarray(profile, dtype=float).tolist()
            cases.append(case)
    return cases


def stability_grids() -> tuple[np.ndarray, np.ndarray]:
    """(φ, l) — те же формулы сеток, что в computePhase."""
    return np.linspace(0.001, 1.0, STABILITY_POINTS), np.linspace(0.001, STABILITY_MAX_LENGTH_RATIO, STABILITY_POINTS)


def stability_cases(source: ModuleType, setup: PolygonReferenceSetup) -> list[dict[str, object]]:
    """Диаграммы устойчивости источника: argmin энергии по формам setup.shapes (порядок важен при равенстве)."""
    registry = getattr(source, f"AREA_{setup.pore_name.upper()}")
    volume_fractions, length_ratios = stability_grids()
    phi_mesh, length_mesh = np.meshgrid(volume_fractions, length_ratios)
    cases: list[dict[str, object]] = []
    for angle_deg, wall_layer in itertools.product(setup.stability_angles_deg, STABILITY_WALL_LAYERS):
        angle = math.radians(angle_deg)
        energies = []
        for shape_name in setup.shapes:
            energy = np.asarray(registry[shape_name](length_mesh, phi_mesh, angle, wall_layer, False), dtype=float)
            energy[~np.isfinite(energy)] = np.inf
            energies.append(energy)
        stacked = np.stack(energies)
        valid = np.isfinite(stacked).any(axis=0)
        winner = np.argmin(stacked, axis=0)
        stable = [
            [setup.shapes[winner[i, j]] if valid[i, j] else None for j in range(len(volume_fractions))]
            for i in range(len(length_ratios))
        ]
        cases.append({"theta": angle_deg, "delta": wall_layer, "stable_shapes": stable})
    return cases


def main() -> None:
    for setup in SETUPS:
        source = load_source(setup.pore_name)
        reference: dict[str, object] = {
            "source": f"{SOURCE_COMMIT} docs/model/{setup.pore_name}.py",
            "coordinate": bin_centers(setup.coordinate_bounds).tolist(),
            "layer_width_ratio": LAYER_WIDTH_RATIO,
            "shapes": {name: shape_cases(source, setup, name) for name in setup.shapes},
            "stability": {
                "volume_fractions": stability_grids()[0].tolist(),
                "length_ratios": stability_grids()[1].tolist(),
                "shape_order": list(setup.shapes),
                "cases": stability_cases(source, setup),
            },
        }
        output = OUTPUT_DIRECTORY / f"{setup.pore_name}_reference.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reference))
        print(f"записано в {output}: {len(setup.shapes)} форм")


if __name__ == "__main__":
    main()
