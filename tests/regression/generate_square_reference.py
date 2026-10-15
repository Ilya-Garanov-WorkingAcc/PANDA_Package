"""Эталон квадратной поры из PANDA_Polygon (`docs/model/square.py`, коммит 38ff491).

Запуск из корня репозитория (нужна локальная копия PANDA_Polygon/, см. ROADMAP «Фаза 4а»):
    uv run python tests/regression/generate_square_reference.py
Результат: tests/regression/data/square_reference.json — коммитится, тесты читают только его.
"""

import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np

SOURCE = Path(__file__).parents[2] / "PANDA_Polygon" / "docs" / "model" / "square.py"
OUTPUT = Path(__file__).parent / "data" / "square_reference.json"
COORDINATE = np.linspace(-0.4975, 0.4975, 200)  # центры 200 бинов: ни одна точка не лежит на стенке или краю δ
LENGTH_RATIOS = (1.0, 2.0, 4.0)
LAYER_WIDTH_RATIO = 1.0
ANGLES_DEG = (60.0, 100.0, 120.0, 150.0, 175.0)
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.3)
SHAPES = ("filament_w1c0", "droplet_w1c0")


def load_square_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("panda_polygon_square", SOURCE)
    assert spec is not None and spec.loader is not None, f"нет файла {SOURCE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    square = load_square_module()
    reference: dict[str, object] = {
        "source": "github.com/Vandyshge/PANDA_Polygon@38ff491 docs/model/square.py",
        "coordinate": COORDINATE.tolist(),
        "layer_width_ratio": LAYER_WIDTH_RATIO,
    }
    for shape_name in SHAPES:
        cases = []
        for length_ratio, angle_deg, wall_layer, phi in itertools.product(
            LENGTH_RATIOS, ANGLES_DEG, WALL_LAYERS, VOLUME_FRACTIONS
        ):
            angle = math.radians(angle_deg)
            feasible = bool(square.COND_SQUARE[shape_name](length_ratio, phi, angle, wall_layer))
            case: dict[str, object] = {
                "length_ratio": length_ratio,
                "theta": angle_deg,
                "delta": wall_layer,
                "phi": phi,
                "feasible": feasible,
            }
            if feasible:
                profile = square.RHO_SQUARE[shape_name](
                    COORDINATE, length_ratio, LAYER_WIDTH_RATIO, phi, angle, wall_layer
                )
                case["profile"] = np.asarray(profile, dtype=float).tolist()
                case["energy"] = float(square.AREA_SQUARE[shape_name](length_ratio, phi, angle, wall_layer, flag=False))
                case["area_sum"] = float(
                    square.AREA_SQUARE[shape_name](length_ratio, phi, angle, wall_layer, flag=True)
                )
            cases.append(case)
        reference[shape_name] = cases
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(reference))
    print(f"записано в {OUTPUT}: " + ", ".join(f"{name}: {len(reference[name])}" for name in SHAPES))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
