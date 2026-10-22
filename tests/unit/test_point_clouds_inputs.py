"""Input checks of the synthetic trajectory."""

import pytest

from panda_md import SlitPore
from panda_md.synthetic.points import synthetic_trajectory

PORE = SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5)


@pytest.mark.parametrize(("argument", "value"), [("substrate_nm", -1.0), ("substrate_nm", 0.0), ("frame_count", 0)])
def test_synthetic_trajectory_checks_its_input(argument: str, value: float) -> None:
    options = {"substrate_nm": 1.0, "frame_count": 1, argument: value}
    with pytest.raises(ValueError, match=argument):
        synthetic_trajectory(
            PORE.interface("roll"),
            {"theta": 120.0, "delta": 0.03, "phi": 0.3},
            density_bulk_per_nm3=33.0,
            substrate_nm=float(options["substrate_nm"]),
            frame_count=int(options["frame_count"]),
        )
