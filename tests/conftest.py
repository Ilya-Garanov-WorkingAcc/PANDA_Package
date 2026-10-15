"""Общие поры для тестов: размеры порядка реальных МД-систем."""

import pytest

from panda_md import SlitPore, SquarePore


@pytest.fixture
def slit_pore() -> SlitPore:
    return SlitPore(width_nm=8.0, length_nm=16.0)


@pytest.fixture
def square_pore() -> SquarePore:
    return SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
