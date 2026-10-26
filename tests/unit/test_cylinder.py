"""Cylindrical pore and its capsule shape (formulas: alpha.py rollx)."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import panda_md as panda
from panda_md import CylinderPore
from panda_md.cli.options import load_pore, pore_json
from panda_md.core.enums import Axis
from panda_md.pores.registry import pore_class

TUBE = {"diameter_nm": 8.0, "length_nm": 16.0, "layer_width_nm": 11.345}


def test_cylinder_pore_geometry() -> None:
    pore = CylinderPore(**TUBE)
    geometry = pore.geometry()
    assert geometry.pore_length_ratio == 2.0
    assert geometry.layer_width_ratio == pytest.approx(1.418125)
    assert geometry.coordinate_bounds() == (-0.5, 0.5)
    assert geometry.profile_integral() == pytest.approx(math.pi / 4.0 / 1.418125)
    assert geometry.layer_area() == pytest.approx(2.0 * 1.418125)
    assert pore.characteristic_size_nm() == 8.0
    assert pore.volume_nm3() == pytest.approx(math.pi * 64.0 * 16.0 / 4.0)
    assert pore.bin_cross_section_nm2(np.array([16.0, 11.345, 11.345])) == pytest.approx(16.0 * 11.345)
    assert pore.layer_axis is Axis.Y


def test_cylinder_pore_rejects_bad_dimensions() -> None:
    with pytest.raises(ValidationError, match="layer width"):
        CylinderPore(diameter_nm=8.0, length_nm=16.0, layer_width_nm=7.0)
    with pytest.raises(ValidationError, match="axis"):
        CylinderPore(**TUBE, axis="z")


def test_cylinder_pore_is_registered(tmp_path: Path) -> None:
    assert pore_class("cylinder") is CylinderPore
    assert panda.CylinderPore is CylinderPore
    pore = CylinderPore(**TUBE, wall_offset_nm=0.1, wall_selection="resname GRAPH*")
    path = tmp_path / "pore.json"
    path.write_text(pore_json(pore), encoding="utf-8")
    assert load_pore(None, path, []) == pore
