"""Мелкие находки итогового аудита перед 0.1.0: время, PBC, размеры щели по траектории, реестр пор, сводки."""

import math

import numpy as np
import pytest

from panda_md import ArrayTrajectory, FitSpec, SlitPore
from panda_md.analysis.blocks import fit_blocks
from panda_md.core.errors import InvalidTrajectoryError, PandaError
from panda_md.core.pbc import circular_mean
from panda_md.core.profile import ProfileSet
from panda_md.core.timing import to_picoseconds


@pytest.mark.parametrize(("text", "picoseconds"), [(".5 ns", 500.0), ("2 μs", 2.0e6), ("2 µs", 2.0e6), ("0", 0.0)])
def test_time_strings_that_people_write(text: str, picoseconds: float) -> None:
    assert to_picoseconds(text) == pytest.approx(picoseconds)


@pytest.mark.parametrize("value", ["nan", "inf", "-1 ns", -5.0, True, math.nan])
def test_time_must_be_a_finite_non_negative_number(value: object) -> None:
    with pytest.raises(ValueError, match="время"):
        to_picoseconds(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("values", [[-1e-17], [5.0], [0.1, -0.1 - 1e-16]])
def test_circular_mean_stays_inside_the_period(values: list[float]) -> None:
    center = circular_mean(np.array(values), 5.0)
    assert 0.0 <= center < 5.0


def test_nearly_square_base_is_not_rejected_by_rounding() -> None:
    """Анизотропный NPT: рёбра 5.0000010 и 5.0000005 нм — основание квадратное, а не «ось перепутана»."""
    walls = np.array([[x, y, z] for x in (0.5, 2.5, 4.5) for y in (0.5, 2.5, 4.5) for z in (0.5, 1.0)])
    trajectory = ArrayTrajectory(
        positions_nm=walls[np.newaxis],
        boxes_nm=np.array([[5.0000005, 5.0000010, 6.0]]),
        times_ps=np.array([0.0]),
        groups={"walls": np.arange(len(walls))},
    )
    pore = SlitPore.from_trajectory(trajectory, walls="walls")
    assert pore.base_aspect_ratio == pytest.approx(1.0)
    with pytest.raises(InvalidTrajectoryError, match="length_axis"):  # заметно длиннее — по-прежнему ошибка
        SlitPore.from_trajectory(trajectory.model_copy(update={"boxes_nm": np.array([[5.0, 5.5, 6.0]])}), walls="walls")


def test_duplicate_pore_type_names_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from panda_md.pores import registry

    class OtherSlit(SlitPore):
        pass  # то же имя «slit»

    monkeypatch.setattr(registry, "PORE_TYPES", (*registry.PORE_TYPES, OtherSlit))
    with pytest.raises(PandaError, match="slit"):
        registry.pore_types()


def test_fixed_parameters_have_exactly_zero_spread_in_block_summary() -> None:
    from pathlib import Path

    data = Path(__file__).parents[1] / "md" / "data"
    pore = SlitPore.model_validate_json((data / "graphite_pore.json").read_text())
    profiles = ProfileSet.load(data / "graphite_water_profiles.npz")
    blocks = fit_blocks(pore.interface("roll"), profiles, FitSpec(density_bulk_per_nm3=33.13))
    assert blocks.std("phi") == 0.0
    assert "phi = 0.48661 ± 0," in blocks.summary()
