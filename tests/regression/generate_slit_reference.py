"""Эталон щели из старой PANDA (`interface/alpha.py`, v0.2.0, коммит 93c6dbb), основание квадратное (k = 1).

Запуск (нужно окружение со старой PANDA, см. ROADMAP «Фаза 0»):
    .venv-legacy/bin/python tests/regression/generate_slit_reference.py
Результат: tests/regression/data/slit_reference.json — коммитится, тесты читают только его.

Капля и червь в старом коде записаны от центра сферы / цилиндра (флаг center=False); в новой версии — в системе
поры, форма лежит на слое δ над стенкой z = −1/2. Поэтому их старый профиль считается в точках z − z_c,
где z_c = (−1/2 + δ) − R cos θ.
"""

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

alpha = importlib.import_module("panda.interface.alpha")

OUTPUT = Path(__file__).parent / "data" / "slit_reference.json"
COORDINATE = np.linspace(-0.4975, 0.4975, 200)  # центры 200 бинов: ни одна точка не лежит на стенке или краю δ
LENGTH_RATIOS = (1.0, 2.0, 3.5)
ANGLES_DEG = (100.0, 120.0, 150.0, 179.0)
WALL_LAYERS = (0.0, 0.03, 0.08)
PHI_POSITIONS = (0.2, 0.5, 0.8)  # положение φ внутри интервала существования
LAYER_ANGLE = math.radians(120.0)  # у слоя θ не участвует в формулах; alpha.py требует аргумент


def lying_shape_profile(shape: str, length_ratio: float, phi: float, angle: float, wall_layer: float) -> list[float]:
    """Капля или червь: старый профиль от центра сферы/цилиндра, перенесённый на опору у нижней стенки."""
    radius = float(getattr(alpha, f"r_{shape}_alpha")(length_ratio, phi, angle, wall_layer))
    center = (-0.5 + wall_layer) - radius * math.cos(angle)
    profile = getattr(alpha, f"rho_{shape}_alpha")(
        COORDINATE - center, length_ratio, phi, angle, wall_layer, center=False
    )
    return [float(value) for value in np.asarray(profile, dtype=float)]


def centered_shape_profile(shape: str, length_ratio: float, phi: float, angle: float, wall_layer: float) -> list[float]:
    profile = getattr(alpha, f"rho_{shape}_alpha")(COORDINATE, length_ratio, phi, angle, wall_layer)
    return [float(value) for value in np.asarray(profile, dtype=float)]


PROFILES: dict[str, Callable[[str, float, float, float, float], list[float]]] = {
    "roll": centered_shape_profile,
    "doughnut": centered_shape_profile,
    "perforation": centered_shape_profile,
    "droplet": lying_shape_profile,
    "worm": lying_shape_profile,
}


def shape_cases(shape: str) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for length_ratio in LENGTH_RATIOS:
        for angle_deg in ANGLES_DEG:
            for wall_layer in WALL_LAYERS:
                angle = math.radians(angle_deg)
                lower = float(getattr(alpha, f"phi_min_{shape}_alpha")(length_ratio, angle, wall_layer))
                upper = float(getattr(alpha, f"phi_max_{shape}_alpha")(length_ratio, angle, wall_layer))
                low, high = max(lower, 0.0), min(upper, 1.0)
                if not low < high:
                    continue  # форма не существует при этих l, θ, δ
                for position in PHI_POSITIONS:
                    phi = low + position * (high - low)
                    cases.append(
                        {
                            "length_ratio": length_ratio,
                            "theta": angle_deg,
                            "delta": wall_layer,
                            "phi": phi,
                            "phi_limits": [lower, upper],
                            "profile": PROFILES[shape](shape, length_ratio, phi, angle, wall_layer),
                        }
                    )
    return cases


def layer_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for wall_layer in WALL_LAYERS:
        upper = float(alpha.phi_max_layer_alpha(1.0, LAYER_ANGLE, wall_layer))
        for position in PHI_POSITIONS:
            phi = position * upper
            profile = alpha.rho_layer_alpha(COORDINATE, 1.0, phi, LAYER_ANGLE, wall_layer)
            cases.append(
                {
                    "length_ratio": 1.0,
                    "delta": wall_layer,
                    "phi": phi,
                    "phi_limits": [0.0, upper],
                    "profile": np.asarray(profile, dtype=float).tolist(),
                }
            )
    return cases


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reference: dict[str, object] = {
        "source": "github.com/FlufffyMelon/PANDA@93c6dbb panda/interface/alpha.py",
        "coordinate": COORDINATE.tolist(),
    }
    cases: dict[str, list[dict[str, object]]] = {shape: shape_cases(shape) for shape in PROFILES}
    cases["layer"] = layer_cases()
    reference.update(cases)
    OUTPUT.write_text(json.dumps(reference))
    counts = ", ".join(f"{shape}: {len(shape_list)}" for shape, shape_list in cases.items())
    print(f"записано в {OUTPUT}: {counts}")


if __name__ == "__main__":
    main()
