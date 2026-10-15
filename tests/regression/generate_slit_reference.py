"""Эталон щели из старой PANDA (`interface/alpha.py`, v0.2.0, коммит 93c6dbb).

Запуск (нужно окружение со старой PANDA, см. ROADMAP «Фаза 0»):
    .venv-legacy/bin/python tests/regression/generate_slit_reference.py
Результат: tests/regression/data/slit_reference.json — коммитится, тесты читают только его.

Капля в старом коде записана от центра сферы (флаг center=False); в новой версии — в системе поры,
капля стоит на слое δ над стенкой z = −1/2. Поэтому старый профиль считается в точках z − z_c,
где z_c = (−1/2 + δ) − R cos θ.
"""

import importlib
import json
import math
from pathlib import Path

import numpy as np

alpha = importlib.import_module("panda.interface.alpha")

OUTPUT = Path(__file__).parent / "data" / "slit_reference.json"
COORDINATE = np.linspace(-0.4975, 0.4975, 200)  # центры 200 бинов: ни одна точка не лежит на стенке или краю δ
LENGTH_RATIOS = (1.0, 2.0, 3.5)
ANGLES_DEG = (100.0, 120.0, 150.0, 179.0)
WALL_LAYERS = (0.0, 0.03, 0.08)
PHI_POSITIONS = (0.2, 0.5, 0.8)  # положение φ внутри интервала существования


def roll_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for length_ratio in LENGTH_RATIOS:
        for angle_deg in ANGLES_DEG:
            for wall_layer in WALL_LAYERS:
                angle = math.radians(angle_deg)
                lower = float(alpha.phi_min_roll_alpha(length_ratio, angle, wall_layer))
                upper = float(alpha.phi_max_roll_alpha(length_ratio, angle, wall_layer))
                for position in PHI_POSITIONS:
                    phi = max(lower, 0.0) + position * (min(upper, 1.0) - max(lower, 0.0))
                    profile = alpha.rho_roll_alpha(COORDINATE, length_ratio, phi, angle, wall_layer)
                    cases.append(
                        {
                            "length_ratio": length_ratio,
                            "theta": angle_deg,
                            "delta": wall_layer,
                            "phi": phi,
                            "phi_limits": [lower, upper],
                            "profile": np.asarray(profile, dtype=float).tolist(),
                        }
                    )
    return cases


def droplet_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for length_ratio in LENGTH_RATIOS:
        for angle_deg in ANGLES_DEG:
            for wall_layer in WALL_LAYERS:
                angle = math.radians(angle_deg)
                upper = float(alpha.phi_max_droplet_alpha(length_ratio, angle, wall_layer))
                for position in PHI_POSITIONS:
                    phi = position * upper
                    radius = float(alpha.r_droplet_alpha(length_ratio, phi, angle, wall_layer))
                    sphere_center = (-0.5 + wall_layer) - radius * math.cos(angle)
                    profile = alpha.rho_droplet_alpha(
                        COORDINATE - sphere_center, length_ratio, phi, angle, wall_layer, center=False
                    )
                    cases.append(
                        {
                            "length_ratio": length_ratio,
                            "theta": angle_deg,
                            "delta": wall_layer,
                            "phi": phi,
                            "phi_limits": [0.0, upper],
                            "profile": np.asarray(profile, dtype=float).tolist(),
                        }
                    )
    return cases


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reference = {
        "source": "github.com/FlufffyMelon/PANDA@93c6dbb panda/interface/alpha.py",
        "coordinate": COORDINATE.tolist(),
        "roll": roll_cases(),
        "droplet": droplet_cases(),
    }
    OUTPUT.write_text(json.dumps(reference))
    print(f"записано {len(reference['roll'])} + {len(reference['droplet'])} случаев в {OUTPUT}")


if __name__ == "__main__":
    main()
