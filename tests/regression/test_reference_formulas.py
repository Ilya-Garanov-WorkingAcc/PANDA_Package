"""Перенос формул: новая реализация совпадает с исходным кодом на зафиксированной сетке.

Эталоны — JSON в data/, генераторы — generate_*.py рядом (источник и коммит записаны в файле).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from panda_md import SlitPore, SquarePore, TrianglePore
from panda_md.analysis.stability import stability_diagram
from panda_md.pores.polygon import PolygonPore

DATA = Path(__file__).parent / "data"
TOLERANCE = 1e-12
SLIT_WIDTH_NM = 1.0


def load(name: str) -> dict[str, Any]:  # Any: структура JSON эталона описана в генераторе
    reference: dict[str, Any] = json.loads((DATA / name).read_text())
    return reference


SLIT = load("slit_reference.json")
POLYGON = {pore_name: load(f"{pore_name}_reference.json") for pore_name in ("triangle", "square")}


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


POLYGON_PORES: dict[str, type[PolygonPore]] = {"triangle": TrianglePore, "square": SquarePore}
POLYGON_CASES = [(pore_name, shape_name) for pore_name in POLYGON_PORES for shape_name in POLYGON[pore_name]["shapes"]]


@pytest.mark.parametrize(("pore_name", "shape_name"), POLYGON_CASES)
def test_polygon_matches_panda_polygon(pore_name: str, shape_name: str) -> None:
    reference = POLYGON[pore_name]
    coordinate = np.array(reference["coordinate"])
    side_nm = 1.0
    for case in reference["shapes"][shape_name]:
        pore = POLYGON_PORES[pore_name](
            side_nm=side_nm,
            length_nm=case["length_ratio"] * side_nm,
            layer_width_nm=reference["layer_width_ratio"] * side_nm,
        )
        model = pore.interface(shape_name)
        values = {name: case[name] for name in ("theta", "delta", "phi") if name in case}
        assert model.is_feasible(**values) == case["feasible"], case_id(case)
        if not case["feasible"]:
            continue
        terms = model.surface_terms(**values)
        assert model.capillary_energy(**values) == pytest.approx(case["energy"], abs=TOLERANCE), case_id(case)
        assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
            case["area_sum"], abs=TOLERANCE
        ), case_id(case)
        if "profile" in case:
            np.testing.assert_allclose(
                model.profile(coordinate, **values), case["profile"], rtol=0.0, atol=TOLERANCE, err_msg=case_id(case)
            )


@pytest.mark.parametrize("pore_name", list(POLYGON_PORES))
def test_polygon_stability_diagram_matches_panda_polygon(pore_name: str) -> None:
    reference = POLYGON[pore_name]["stability"]
    pore = POLYGON_PORES[pore_name](side_nm=1.0, length_nm=1.0, layer_width_nm=POLYGON[pore_name]["layer_width_ratio"])
    for case in reference["cases"]:
        diagram = stability_diagram(
            pore,
            contact_angle_deg=case["theta"],
            wall_layer=case["delta"],
            volume_fractions=reference["volume_fractions"],
            length_ratios=reference["length_ratios"],
            shape_names=reference["shape_order"],
        )
        expected = tuple(tuple(row) for row in case["stable_shapes"])
        mismatches = [
            (
                reference["length_ratios"][i],
                reference["volume_fractions"][j],
                expected[i][j],
                diagram.stable_shapes[i][j],
            )
            for i in range(len(expected))
            for j in range(len(expected[i]))
            if expected[i][j] != diagram.stable_shapes[i][j]
        ]
        assert not mismatches, f"θ={case['theta']} δ={case['delta']}: {mismatches[:5]} (всего {len(mismatches)})"
