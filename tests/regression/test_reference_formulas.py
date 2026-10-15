"""Перенос формул: новая реализация совпадает с исходным кодом на зафиксированной сетке.

Эталоны — JSON в data/, генераторы — generate_*.py рядом (источник и коммит записаны в файле).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from panda_md import SlitPore, SquarePore

DATA = Path(__file__).parent / "data"
TOLERANCE = 1e-12
SLIT_WIDTH_NM = 1.0


def load(name: str) -> dict[str, Any]:  # Any: структура JSON эталона описана в генераторе
    reference: dict[str, Any] = json.loads((DATA / name).read_text())
    return reference


SLIT = load("slit_reference.json")
SQUARE = load("square_reference.json")


def case_id(case: dict[str, Any]) -> str:
    return f"l={case['length_ratio']}-θ={case.get('theta', '—')}-δ={case['delta']}-φ={case['phi']:.4f}"


@pytest.mark.parametrize("shape_name", ["droplet", "doughnut", "worm", "roll", "perforation", "layer"])
def test_slit_matches_old_panda(shape_name: str) -> None:
    coordinate = np.array(SLIT["coordinate"])
    for case in SLIT[shape_name]:
        pore = SlitPore(width_nm=SLIT_WIDTH_NM, length_nm=case["length_ratio"] * SLIT_WIDTH_NM)
        model = pore.interface(shape_name)
        values = {name: case[name] for name in ("theta", "delta", "phi") if name in case}
        np.testing.assert_allclose(
            model.profile(coordinate, **values), case["profile"], rtol=0.0, atol=TOLERANCE, err_msg=case_id(case)
        )
        np.testing.assert_allclose(
            model.phi_limits(**values), case["phi_limits"], rtol=0.0, atol=TOLERANCE, err_msg=case_id(case)
        )


@pytest.mark.parametrize("shape_name", ["filament_w1c0", "droplet_w1c0"])
def test_square_matches_panda_polygon(shape_name: str) -> None:
    coordinate = np.array(SQUARE["coordinate"])
    for case in SQUARE[shape_name]:
        side_nm = 1.0
        pore = SquarePore(
            side_nm=side_nm,
            length_nm=case["length_ratio"] * side_nm,
            layer_width_nm=SQUARE["layer_width_ratio"] * side_nm,
        )
        model = pore.interface(shape_name)
        values = {"theta": case["theta"], "delta": case["delta"], "phi": case["phi"]}
        assert model.is_feasible(**values) == case["feasible"], case_id(case)
        if not case["feasible"]:
            continue
        np.testing.assert_allclose(
            model.profile(coordinate, **values), case["profile"], rtol=0.0, atol=TOLERANCE, err_msg=case_id(case)
        )
        terms = model.surface_terms(**values)
        assert model.capillary_energy(**values) == pytest.approx(case["energy"], abs=TOLERANCE), case_id(case)
        assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
            case["area_sum"], abs=TOLERANCE
        ), case_id(case)
