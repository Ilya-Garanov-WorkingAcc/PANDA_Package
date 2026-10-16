"""Анализ: фит по блокам, статистическая неэффективность, перебор форм."""

import numpy as np
import pytest

from panda_md import FitSpec, SlitPore, TrianglePore
from panda_md.analysis.blocks import fit_blocks, statistical_inefficiency
from panda_md.analysis.compare import compare_shapes
from panda_md.core.enums import Frame, Normalization, ProfileCentering
from panda_md.core.errors import PandaError, PandaWarning, ProfileMismatchError
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
    result = statistical_inefficiency([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], max_block_length=3)
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


def test_shape_comparison_ranks_the_true_shape_first() -> None:
    profile = noisy_blocks().mean()
    with pytest.warns(PandaWarning, match="неразличимы"):  # перфорация по профилю почти как валик
        comparison = compare_shapes(SLIT, profile, FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    assert comparison.best.shape_name == "roll"
    assert comparison.rows[0].shape_name == "roll"
    losses = [result.loss for result in comparison.fitted if result.feasible]
    assert losses == sorted(losses)
    assert {row.shape_name for row in comparison.rows} == set(SLIT.shape_names())
    assert "roll" in comparison.summary()


def test_shapes_that_cannot_be_fitted_are_listed_with_a_reason() -> None:
    triangle = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    profile = synthetic_profile(
        triangle.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=30.0
    )
    comparison = compare_shapes(
        triangle, profile, FitSpec(density_bulk_per_nm3=30.0), shape_names=["droplet_w1c0", "droplet_w2c0"]
    )
    assert [row.shape_name for row in comparison.rows] == ["droplet_w1c0", "droplet_w2c0"]
    skipped = comparison.rows[1]
    assert skipped.result is None and skipped.skipped_reason is not None and "профиль" in skipped.skipped_reason
    assert "пропущена" in comparison.summary()
    only_skipped = compare_shapes(triangle, profile, FitSpec(density_bulk_per_nm3=30.0), shape_names=["droplet_w2c0"])
    with pytest.raises(PandaError, match="ни одна форма"):
        _ = only_skipped.best


def ar1_series(correlation: float, size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    series = np.empty(size)
    series[0] = 0.0
    for index in range(1, size):
        series[index] = correlation * series[index - 1] + generator.normal()
    return series


def test_default_plateau_is_reliable() -> None:
    """Без max_block_length плато считается только по длинам, при которых блоков достаточно для дисперсии."""
    plateaus = [statistical_inefficiency(ar1_series(0.8, 5000, seed)).plateau() for seed in range(10)]
    assert float(np.mean(plateaus)) == pytest.approx(9.0, rel=0.2)
    assert float(np.std(plateaus)) < 3.0
    result = statistical_inefficiency(ar1_series(0.8, 5000, 0))
    assert max(result.block_lengths) <= 5000 // 20


def test_single_block_has_no_spread_estimate() -> None:
    blocks = noisy_blocks()
    single = ProfileSet(
        **{
            **blocks.model_dump(),
            "block_densities_per_nm3": blocks.block_densities_per_nm3[:1],
            "block_start_ps": blocks.block_start_ps[:1],
            "block_stop_ps": blocks.block_stop_ps[:1],
            "block_frame_counts": blocks.block_frame_counts[:1],
            "block_particle_counts": blocks.block_particle_counts[:1],
        }
    )
    result = fit_blocks(SLIT.interface("roll"), single, FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    assert np.isnan(result.std("theta")) and np.isnan(result.sem("theta"))
    assert "один блок" in result.summary()


def test_comparison_does_not_hide_profile_level_errors() -> None:
    profile = noisy_blocks().mean()
    wrong_pore = SlitPore(width_nm=2.0, length_nm=16.0)  # профиль шире поры
    with pytest.raises(ProfileMismatchError):
        compare_shapes(wrong_pore, profile, FitSpec(density_bulk_per_nm3=DENSITY_BULK))
    with pytest.raises(PandaError, match="повтор"):
        compare_shapes(SLIT, profile, FitSpec(density_bulk_per_nm3=DENSITY_BULK), shape_names=["roll", "roll"])


def test_best_names_the_reasons_when_nothing_fits() -> None:
    triangle = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    profile = synthetic_profile(
        triangle.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=30.0
    )
    comparison = compare_shapes(triangle, profile, FitSpec(density_bulk_per_nm3=30.0), shape_names=["droplet_w2c0"])
    with pytest.raises(PandaError, match=r"droplet_w2c0.*профиль"):
        _ = comparison.best


def test_statistical_inefficiency_of_a_short_series_is_not_reported_as_independent() -> None:
    """Итоговый аудит: 32 значения θ по блокам давали одну длину блока и s = 1 («ряд независим»)."""
    from panda_md.core.errors import PandaError, PandaWarning

    series = np.random.default_rng(0).normal(size=32)
    with pytest.raises(PandaError, match="короток"):
        statistical_inefficiency(series)
    with pytest.warns(PandaWarning, match="плато"):
        statistical_inefficiency(series, max_block_length=4).plateau()


def test_compare_shapes_flags_shapes_that_the_profile_cannot_tell_apart() -> None:
    """Итоговый аудит: на профиле валика перфорация выигрывала у валика на 0.3 % loss — без всякой оговорки."""
    from panda_md import SlitPore, compare_shapes
    from panda_md.core.enums import NoiseModel, ProfileSampling
    from panda_md.core.errors import PandaWarning
    from panda_md.synthetic.profiles import synthetic_profile

    pore = SlitPore(width_nm=8.0, length_nm=16.0)
    profile = synthetic_profile(
        pore.interface("roll"), {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=30.0,
        noise_std=0.005, seed=0, noise_model=NoiseModel.POISSON, sampling=ProfileSampling.BIN_AVERAGE,
    )  # fmt: skip
    with pytest.warns(PandaWarning, match="неразличимы"):
        comparison = compare_shapes(pore, profile, FitSpec(density_bulk_per_nm3=30.0))
    assert {"roll", "perforation"} <= set(comparison.indistinguishable)
    assert "≈" in comparison.summary()


def test_mean_normalization_without_phi_explains_what_to_set() -> None:
    """Итоговый аудит: «задай значение phi…» без причины; compare_shapes тихо оставлял один слой."""
    from panda_md import SlitPore, compare_shapes
    from panda_md.core.errors import InvalidProfileError
    from panda_md.synthetic.profiles import synthetic_profile

    pore = SlitPore(width_nm=8.0, length_nm=16.0)
    profile = synthetic_profile(pore.interface("roll"), {"theta": 120.0, "phi": 0.4}, density_bulk_per_nm3=30.0)
    with pytest.raises(InvalidProfileError, match="ρ_bulk"):
        pore.interface("roll").fit(profile, normalization=Normalization.MEAN)
    with pytest.raises(InvalidProfileError, match="ρ_bulk"):
        compare_shapes(pore, profile)
