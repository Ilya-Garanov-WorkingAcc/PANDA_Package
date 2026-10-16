"""Анализ: фит по блокам и статистическая неэффективность."""

import numpy as np
import pytest

from panda_md import FitSpec, SlitPore
from panda_md.analysis.blocks import fit_blocks, statistical_inefficiency
from panda_md.core.enums import Frame, ProfileCentering
from panda_md.core.errors import PandaError
from panda_md.core.profile import ProfileSet
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
SLIT = SlitPore(width_nm=8.0, length_nm=16.0)
TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}
BLOCKS = 4


def noisy_blocks(model_name: str = "roll") -> ProfileSet:
    """Блоки — синтетические профили одной формы с разным шумом (как блоки одной траектории)."""
    model = SLIT.interface(model_name)
    profiles = [
        synthetic_profile(model, TRUTH, density_bulk_per_nm3=DENSITY_BULK, noise_std=0.005, seed=seed)
        for seed in range(BLOCKS)
    ]
    centers = profiles[0].coordinate_nm
    step = centers[1] - centers[0]
    edges = np.append(centers - step / 2.0, centers[-1] + step / 2.0)
    return ProfileSet(
        bin_edges_nm=edges,
        block_densities_per_nm3=np.array([profile.density_per_nm3 for profile in profiles]),
        block_start_ps=1000.0 * np.arange(BLOCKS),
        block_stop_ps=1000.0 * np.arange(BLOCKS) + 998.0,
        block_frame_counts=(500,) * BLOCKS,
        block_particle_counts=np.full(BLOCKS, profiles[0].particle_count),
        frame=Frame.PORE,
        centering=ProfileCentering.PORE,
        selection="синтетика",
        pore_volume_nm3=SLIT.volume_nm3(),
    )


def test_blocks_are_fitted_with_one_spec_and_summarised() -> None:
    blocks = fit_blocks(SLIT.interface("roll"), noisy_blocks(), FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    angles = blocks.values("theta")
    assert angles.shape == (BLOCKS,)
    assert blocks.mean("theta") == pytest.approx(TRUTH["theta"], abs=1.0)
    assert blocks.std("theta") > 0.0
    assert blocks.sem("theta") == pytest.approx(blocks.std("theta") / np.sqrt(BLOCKS))
    np.testing.assert_allclose(blocks.block_start_ps, [0.0, 1000.0, 2000.0, 3000.0])
    text = blocks.summary()
    assert "roll" in text and "среднее ± ошибка среднего" in text and "theta" in text
    with pytest.raises(PandaError, match="kappa"):
        blocks.values("kappa")


def test_statistical_inefficiency_matches_block_variance_formula() -> None:
    """s(2) для ряда 1…6: средние блоков 1.5, 3.5, 5.5 → var = 4; var ряда = 3.5; s = 2 · 4 / 3.5."""
    result = statistical_inefficiency([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert result.block_lengths == (1, 2, 3)
    assert result.inefficiency[0] == pytest.approx(1.0)
    assert result.inefficiency[1] == pytest.approx(2.0 * 4.0 / 3.5)


def test_statistical_inefficiency_finds_correlation_time() -> None:
    generator = np.random.default_rng(3)
    white = generator.normal(size=20000)
    assert statistical_inefficiency(white, max_block_length=200).plateau() == pytest.approx(1.0, abs=0.3)
    correlation = 0.8  # AR(1): s → (1 + ρ) / (1 − ρ) = 9
    series = np.empty(20000)
    series[0] = 0.0
    for index in range(1, series.size):
        series[index] = correlation * series[index - 1] + generator.normal()
    assert statistical_inefficiency(series, max_block_length=400).plateau() == pytest.approx(9.0, rel=0.25)


def test_statistical_inefficiency_rejects_useless_series() -> None:
    with pytest.raises(PandaError, match="хотя бы"):
        statistical_inefficiency([1.0, 2.0])
    with pytest.raises(PandaError, match="постоянен"):
        statistical_inefficiency([1.0] * 10)
