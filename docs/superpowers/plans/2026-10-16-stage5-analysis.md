# Стадия 5 — анализ (фаза 6а): план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Анализ поверх фита: фит каждого блока времени с оценкой ошибки, перебор всех форм поры на одном профиле, статистическая неэффективность для выбора длины блока; профиль по центрам масс молекул и поправка положения поверхности стенки.

**Architecture:** Анализ — функции слоя `analysis` (ядро его не импортирует): `fit_blocks` и `compare_shapes` вызывают `core.fitting.fit` с одним `FitSpec`. Молекулы — новый метод интерфейса `Trajectory.molecules` (номер молекулы и масса атома); центр масс с учётом PBC — `core.pbc.molecule_centers`. Поправка стенки — поле `SlitPore.wall_offset_nm` и метод поры `expected_wall_gap_nm`, по которому путь данных сверяет найденный промежуток.

**Tech Stack:** Python ≥ 3.10, numpy, scipy, pydantic v2, MDAnalysis (extra `md`), pytest, uv, black, ruff, mypy (strict + плагин pydantic).

**Spec:** `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (§3.6 `FitResultSet`, `ShapeComparison`; §5 `per`, `wall_offset`); `docs/architecture/03_centering_data.md` §5 (статистическая неэффективность). Фаза 6 разделена на три стадии (ROADMAP): 6а — эта, 6б — трёхмерные формы и осевое центрирование, 6в — графики и валидация.

**Решения** (вносятся в спецификацию в задаче 4, D24–D25): `compare_shapes(pore, …)` вместо `pore.fit_all` — ядро не импортирует `analysis`; настройки фита для блоков и перебора — один `FitSpec`.

**Проверено до написания плана.** Код собран и прогнан по задачам в чистом клоне ветки `stage1-core-prototype` (коммит `2e41271`): тесты каждой задачи падают до реализации, после — `black --check`, `ruff check`, `mypy`, `pytest` проходят (итого 432 теста, 28 пропусков). На реальной МД (вода в щели графита): `fit_blocks` — θ = 170.7° ± 3.0°; `compare_shapes` ставит первым валик; профиль по молекулам воды — 11 807 молекул. Код переносить дословно.

## Global Constraints

- Все ограничения стадий 1–4 в силе: Python ≥ 3.10; pydantic, без датаклассов (D14); `Enum`, типы у всех полей, говорящие имена (D17); black 120, ruff, mypy strict; комментарии и сообщения — по-русски; слои (§2): `core` не импортирует `analysis`, `pores`, `trajectory`.
- Разработка локальная, в ветке `stage1-core-prototype`; в GitHub не отправлять.
- Многострочный текст — только через Write/Edit; коммиты — отдельными `-m`, последний — `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Проверка перед коммитом: `make check`.

## Review Focus

1. Молекула, разрезанная границей ящика: ожидание — центр масс у границы, а не в середине ящика. Тест — `test_molecule_cut_by_box_boundary_has_its_center_at_the_boundary` (задача 3).
2. Поправка стенки: путь данных не должен предупреждать «промежуток ≠ H», когда промежуток = H + 2 · offset. Тест — `test_wall_offset_moves_surfaces_inward` (задача 3).
3. Перебор форм, где часть форм не фитируется (нет профиля, нет допустимого старта): ожидание — форма в таблице с причиной, а не исключение на весь перебор. Тест — `test_shapes_that_cannot_be_fitted_are_listed_with_a_reason` (задача 2).
4. Статистическая неэффективность коррелированного ряда: ожидание — плато ≈ (1 + ρ)/(1 − ρ). Тест — `test_statistical_inefficiency_finds_correlation_time` (задача 1).
5. Фит по блокам: одни настройки для всех блоков, ошибка среднего = std / √n. Тест — `test_blocks_are_fitted_with_one_spec_and_summarised` (задача 1).

---

### Задача 1: Фит по блокам и статистическая неэффективность

**Files:**
- Create: `src/panda_md/analysis/blocks.py`
- Create: `tests/unit/test_analysis.py` (без тестов перебора форм — они в задаче 2)

**Interfaces:**
- Consumes: `core.fitting.fit(model, profile, spec)`, `ProfileSet.block(index)`, `FitResult`, `FitSpec`.
- Produces: `FitResultSet(results, block_start_ps, block_stop_ps)` с `values(name)`, `mean`, `std`, `sem`, `summary()`; `fit_blocks(model, profiles, spec=None) -> FitResultSet`; `StatisticalInefficiency(block_lengths, inefficiency)` с `plateau()`; `statistical_inefficiency(values, max_block_length=None)`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_analysis.py`:

```python
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
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.analysis.blocks'`.

- [ ] **Шаг 3: Реализация**

`src/panda_md/analysis/blocks.py`:

```python
"""Фит по блокам времени и статистическая неэффективность — оценка ошибки угла и выбор длины блока."""

import math
from collections.abc import Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.errors import PandaError
from panda_md.core.fitting import fit
from panda_md.core.model import InterfaceModel
from panda_md.core.profile import ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.types import Array1D, FloatArray, PandaModel

MIN_SERIES_LENGTH = 4  # статистическую неэффективность по меньшему ряду не оценить


class FitResultSet(PandaModel):
    """Результаты фита каждого блока: среднее, разброс и ошибка среднего по блокам для любого параметра."""

    results: tuple[FitResult, ...] = Field(min_length=1)
    block_start_ps: Array1D
    block_stop_ps: Array1D

    @model_validator(mode="after")
    def _check_lengths(self) -> "FitResultSet":
        count = len(self.results)
        if self.block_start_ps.shape != (count,) or self.block_stop_ps.shape != (count,):
            raise ValueError("времена блоков: по одному на результат")
        return self

    def values(self, name: str) -> FloatArray:
        """Значение параметра (единицы пользователя) в каждом блоке."""
        if name not in self.results[0].values:
            raise PandaError(f"параметра {name} нет; есть: {list(self.results[0].values)}")
        return np.array([result.values[name] for result in self.results])

    def mean(self, name: str) -> float:
        return float(np.mean(self.values(name)))

    def std(self, name: str) -> float:
        """Разброс по блокам (0 при одном блоке)."""
        return float(np.std(self.values(name), ddof=1)) if len(self.results) > 1 else 0.0

    def sem(self, name: str) -> float:
        """Ошибка среднего: std / √(число блоков) — верна, если блоки длиннее времени корреляции."""
        return self.std(name) / math.sqrt(len(self.results))

    def summary(self) -> str:
        names = list(self.results[0].values)
        header = "блок  t, пс           " + "".join(f"{name:>11}" for name in names) + "       loss  флаги"
        lines = [f"{self.results[0].pore_name}/{self.results[0].shape_name}: {len(self.results)} блоков", header]
        for index, result in enumerate(self.results):
            flags = ", ".join(_flags(result))
            span = f"{self.block_start_ps[index]:g}–{self.block_stop_ps[index]:g}"
            row = "".join(f"{result.values[name]:>11.5g}" for name in names)
            lines.append(f"{index:>4}  {span:<16}{row}  {result.loss:>9.4g}  {flags}")
        lines.append(
            "среднее ± ошибка среднего: "
            + ", ".join(f"{name} = {self.mean(name):.5g} ± {self.sem(name):.2g}" for name in names)
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _flags(result: FitResult) -> list[str]:
    flags = [f"у границы {name}" for name in result.at_bounds]
    if result.at_regime_boundary:
        flags.append("θ у 90°")
    if not result.feasible:
        flags.append("вне области существования")
    return flags


def fit_blocks(model: InterfaceModel, profiles: ProfileSet, spec: FitSpec | None = None) -> FitResultSet:
    """Фит профиля каждого блока с одними настройками; разброс по блокам — оценка ошибки параметров."""
    settings = spec if spec is not None else FitSpec()
    results = tuple(fit(model, profiles.block(index), settings) for index in range(profiles.block_count))
    return FitResultSet(results=results, block_start_ps=profiles.block_start_ps, block_stop_ps=profiles.block_stop_ps)


class StatisticalInefficiency(PandaModel):
    """s(t) = t · var(средних по блокам из t значений) / var(ряда). Выходит на плато s ≈ 2τ + 1 (τ — время
    корреляции в шагах ряда), когда блоки длиннее τ; блоки длиннее плато дают независимые оценки."""

    block_lengths: tuple[int, ...]
    inefficiency: Array1D

    def plateau(self) -> float:
        """Среднее s по последней трети длин блоков — оценка числа шагов на одно независимое значение."""
        tail = self.inefficiency[2 * len(self.inefficiency) // 3 :]
        return float(np.mean(tail))


def statistical_inefficiency(
    values: Sequence[float] | FloatArray, max_block_length: int | None = None
) -> StatisticalInefficiency:
    """Статистическая неэффективность ряда (θ по коротким блокам, φ по кадрам, …) — как `stat_ineff.py` старой PANDA."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size < MIN_SERIES_LENGTH:
        raise PandaError(f"нужен одномерный ряд хотя бы из {MIN_SERIES_LENGTH} значений")
    variance = float(np.var(series, ddof=1))
    if variance == 0.0:
        raise PandaError("ряд постоянен: неэффективность не определена")
    longest = series.size // 2 if max_block_length is None else min(max_block_length, series.size // 2)
    mean = float(np.mean(series))
    lengths: list[int] = []
    inefficiency: list[float] = []
    for length in range(1, longest + 1):
        block_count = series.size // length
        block_means = series[: block_count * length].reshape(block_count, length).mean(axis=1)
        block_variance = float(np.sum((block_means - mean) ** 2)) / (block_count - 1)
        lengths.append(length)
        inefficiency.append(length * block_variance / variance)
    return StatisticalInefficiency(block_lengths=tuple(lengths), inefficiency=np.array(inefficiency))
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `423 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/analysis/blocks.py tests/unit/test_analysis.py
git commit -m "Анализ: фит по блокам (среднее ± ошибка среднего) и статистическая неэффективность" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 2: Перебор форм

**Files:**
- Create: `src/panda_md/analysis/compare.py`
- Modify: `tests/unit/test_analysis.py`

**Interfaces:**
- Consumes: `core.fitting.fit`, `Pore.own_registry`, `Pore.interface`.
- Produces: `ShapeFit(shape_name, result=None, skipped_reason=None)`; `ShapeComparison(pore_name, rows)` с `fitted`, `best`, `summary()`; `compare_shapes(pore, profile, spec=None, shape_names=None) -> ShapeComparison`.

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_analysis.py` целиком:

`tests/unit/test_analysis.py`:

```python
"""Анализ: фит по блокам, статистическая неэффективность, перебор форм."""

import numpy as np
import pytest

from panda_md import FitSpec, SlitPore, TrianglePore
from panda_md.analysis.blocks import fit_blocks, statistical_inefficiency
from panda_md.analysis.compare import compare_shapes
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


def test_shape_comparison_ranks_the_true_shape_first() -> None:
    profile = noisy_blocks().mean()
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
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.analysis.compare'`.

- [ ] **Шаг 3: Реализация**

`src/panda_md/analysis/compare.py`:

```python
"""Перебор форм: фит одного профиля всеми формами поры и таблица по качеству."""

from collections.abc import Sequence

from pydantic import model_validator

from panda_md.core.errors import PandaError
from panda_md.core.fitting import fit
from panda_md.core.pore import Pore
from panda_md.core.profile import DensityProfile
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.types import PandaModel


class ShapeFit(PandaModel):
    """Одна строка сравнения: результат фита или причина, по которой форма не фитировалась."""

    shape_name: str
    result: FitResult | None = None
    skipped_reason: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "ShapeFit":
        if (self.result is None) == (self.skipped_reason is None):
            raise ValueError("нужен либо результат, либо причина пропуска")
        return self


class ShapeComparison(PandaModel):
    """Формы по возрастанию ошибки фита; пропущенные — в конце с причиной.

    Ошибки сравнимы между формами только при одинаковых данных и настройках; меньшая ошибка — довод, а не
    доказательство: у разных форм разное число свободных параметров.
    """

    pore_name: str
    rows: tuple[ShapeFit, ...]

    @property
    def fitted(self) -> tuple[FitResult, ...]:
        return tuple(row.result for row in self.rows if row.result is not None)

    @property
    def best(self) -> FitResult:
        """Допустимый результат с наименьшей ошибкой."""
        feasible = [result for result in self.fitted if result.feasible]
        if not feasible:
            raise PandaError("ни одна форма не дала допустимого результата")
        return feasible[0]

    def summary(self) -> str:
        lines = [f"{self.pore_name}: сравнение форм"]
        for row in self.rows:
            if row.result is None:
                lines.append(f"  {row.shape_name:<24} пропущена: {row.skipped_reason}")
                continue
            result = row.result
            values = ", ".join(f"{name} = {value:.4g}" for name, value in result.values.items())
            mark = "" if result.feasible else "  (вне области существования)"
            lines.append(f"  {row.shape_name:<24} loss = {result.loss:.4g}  {values}{mark}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def compare_shapes(
    pore: Pore, profile: DensityProfile, spec: FitSpec | None = None, shape_names: Sequence[str] | None = None
) -> ShapeComparison:
    """Фит профиля каждой формой поры (по умолчанию — всеми) с одними настройками."""
    settings = spec if spec is not None else FitSpec()
    names = list(pore.own_registry()) if shape_names is None else list(shape_names)
    fitted: list[ShapeFit] = []
    skipped: list[ShapeFit] = []
    for name in names:
        model = pore.interface(name)
        try:
            fitted.append(ShapeFit(shape_name=name, result=fit(model, profile, settings)))
        except PandaError as error:  # нет профиля, нет допустимого старта, режим не объявлен, …
            skipped.append(ShapeFit(shape_name=name, skipped_reason=str(error)))
    fitted.sort(key=lambda row: (row.result is None or not row.result.feasible, row.result.loss if row.result else 0.0))
    return ShapeComparison(pore_name=pore.name, rows=(*fitted, *skipped))
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `425 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/analysis/compare.py tests/unit/test_analysis.py
git commit -m "Анализ: перебор форм поры на одном профиле — таблица по ошибке фита, пропущенные формы с причиной" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 3: Частицы и стенки — профиль по молекулам, поправка поверхности стенки

**Files:**
- Modify: `src/panda_md/core/enums.py` (`ParticleGrouping`), `src/panda_md/core/pbc.py` (`molecule_centers`; период — число или по оси), `src/panda_md/core/trajectory.py` (`molecules`, `renumber`), `src/panda_md/core/pore.py` (`expected_wall_gap_nm`), `src/panda_md/pores/slit/pore.py` (`wall_offset_nm`), `src/panda_md/trajectory/arrays.py` (`molecule_ids`, `masses`), `src/panda_md/trajectory/reader.py` (молекулы — остатки), `src/panda_md/trajectory/density.py` (`per`)
- Create: `tests/unit/test_molecules_and_offset.py`

**Interfaces:**
- Consumes: путь данных стадии 4.
- Produces: `ParticleGrouping.ATOM/MOLECULE`; `molecule_centers(positions, box, molecule_indices, masses) -> FloatArray`; абстрактный `Trajectory.molecules(indices) -> (IndexArray, FloatArray)`; `renumber(labels)`; `ArrayTrajectory(…, molecule_ids=None, masses=None)`; `Pore.expected_wall_gap_nm()`; `SlitPore.wall_offset_nm`, `SlitPore.from_trajectory(…, wall_offset_nm=0.0)`; `density_profile(…, per="atom")`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_molecules_and_offset.py`:

```python
"""Профиль по центрам масс молекул и поправка положения поверхности стенки (wall_offset)."""

import warnings
from typing import Any

import numpy as np
import pytest

from panda_md import ArrayTrajectory, SlitPore, density_profile
from panda_md.core.enums import ParticleGrouping
from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import molecule_centers
from panda_md.core.trajectory import renumber
from panda_md.trajectory.reader import MDATrajectory

BOX_NM = np.array([4.0, 4.0, 10.0])


def test_renumber_keeps_order_of_first_appearance() -> None:
    assert renumber(np.array([7, 7, 3, 3, 9, 7])).tolist() == [0, 0, 1, 1, 2, 0]


def test_molecule_cut_by_box_boundary_has_its_center_at_the_boundary() -> None:
    positions = np.array([[1.0, 1.0, 9.9], [1.0, 1.0, 0.1], [2.0, 2.0, 5.0], [2.0, 2.0, 5.4]])
    centers = molecule_centers(positions, BOX_NM, np.array([0, 0, 1, 1]), np.array([1.0, 1.0, 16.0, 1.0]))
    assert min(centers[0, 2], BOX_NM[2] - centers[0, 2]) == pytest.approx(0.0, abs=1e-12)  # а не 5.0
    assert centers[1, 2] == pytest.approx((16.0 * 5.0 + 5.4) / 17.0)
    with pytest.raises(InvalidTrajectoryError, match="масса"):
        molecule_centers(positions, BOX_NM, np.array([0, 0, 1, 1]), np.zeros(4))


def water_like_trajectory() -> ArrayTrajectory:
    """Молекулы из трёх атомов (16 + 1 + 1) в слое |z| < 1 нм вокруг центра поры z = 5; подложка 7.5–12.5 ≡ 2.5."""
    generator = np.random.default_rng(5)
    count = 300
    centers = np.column_stack([generator.uniform(0.0, 4.0, (count, 2)), generator.uniform(4.0, 6.0, count)])
    offsets = np.array([[0.0, 0.0, 0.0], [0.08, 0.06, 0.0], [-0.08, 0.06, 0.0]])
    liquid = (centers[:, np.newaxis, :] + offsets).reshape(-1, 3)
    grid = np.array([[x, y] for x in np.arange(0.0, 4.0, 0.5) for y in np.arange(0.0, 4.0, 0.5)])
    walls = np.vstack([np.column_stack([grid, np.full(len(grid), z)]) for z in np.mod(np.arange(7.5, 12.6, 0.5), 10.0)])
    positions = np.vstack([walls, liquid])
    wall_count = len(walls)
    molecule_ids = np.concatenate([np.arange(wall_count) + 10_000, np.repeat(np.arange(count), 3)])
    masses = np.concatenate([np.full(wall_count, 12.0), np.tile([16.0, 1.0, 1.0], count)])
    return ArrayTrajectory(
        positions_nm=positions[np.newaxis],
        boxes_nm=BOX_NM[np.newaxis],
        times_ps=[0.0],
        groups={"walls": np.arange(wall_count), "water": wall_count + np.arange(3 * count)},
        molecule_ids=molecule_ids,
        masses=masses,
    )


def test_profile_per_molecule_counts_molecules() -> None:
    trajectory = water_like_trajectory()
    pore = SlitPore.from_trajectory(trajectory, walls="walls")
    by_atom = density_profile(trajectory, pore, "water", bins=50)
    by_molecule = density_profile(trajectory, pore, "water", bins=50, per=ParticleGrouping.MOLECULE)
    assert by_atom.block_particle_counts[0] == 900
    assert by_molecule.block_particle_counts[0] == 300
    total = np.sum(by_molecule.block_densities_per_nm3[0] * np.diff(by_molecule.bin_edges_nm)) * 16.0
    assert total == pytest.approx(300.0)
    by_name = density_profile(trajectory, pore, "water", bins=50, per="molecule")
    np.testing.assert_allclose(by_name.block_densities_per_nm3, by_molecule.block_densities_per_nm3)


def test_molecules_need_ids_and_masses() -> None:
    plain = ArrayTrajectory(positions_nm=np.zeros((1, 2, 3)) + 1.0, boxes_nm=BOX_NM[np.newaxis], times_ps=[0.0])
    pore = SlitPore(width_nm=5.0, length_nm=4.0)
    with pytest.raises(InvalidTrajectoryError, match="molecule_ids"):
        density_profile(plain, pore, [0, 1], pore_center_nm=5.0, per="molecule")


def test_wall_offset_moves_surfaces_inward() -> None:
    trajectory = water_like_trajectory()
    bare = SlitPore.from_trajectory(trajectory, walls="walls")
    offset = SlitPore.from_trajectory(trajectory, walls="walls", wall_offset_nm=0.17)
    assert bare.width_nm == pytest.approx(5.0)
    assert offset.width_nm == pytest.approx(5.0 - 2 * 0.17)
    assert offset.wall_offset_nm == pytest.approx(0.17)
    assert offset.expected_wall_gap_nm() == pytest.approx(5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # промежуток стенок = H + 2 · offset — без предупреждения
        profiles = density_profile(trajectory, offset, "water", bins=50)
    np.testing.assert_allclose(profiles.bin_edges_nm[[0, -1]], [-(5.0 - 0.34) / 2, (5.0 - 0.34) / 2])
    with pytest.raises(InvalidTrajectoryError, match="wall_offset_nm"):
        SlitPore.from_trajectory(trajectory, walls="walls", wall_offset_nm=3.0)


def test_mdanalysis_molecules_are_residues() -> None:
    mdanalysis = pytest.importorskip("MDAnalysis")
    universe: Any = mdanalysis.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    universe.add_TopologyAttr("masses", [16.0, 1.0, 12.0, 12.0])
    trajectory = MDATrajectory(universe)
    molecule_indices, masses = trajectory.molecules(np.array([1, 2, 3]))
    assert molecule_indices.tolist() == [0, 1, 1]
    np.testing.assert_allclose(masses, [1.0, 12.0, 12.0])
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ImportError: cannot import name 'ParticleGrouping' from 'panda_md.core.enums'`.

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/core/enums.py` целиком:

`src/panda_md/core/enums.py`:

```python
"""Перечисления всех фиксированных вариантов. Строковые значения используются в CLI и JSON."""

from enum import Enum


class ParameterName(str, Enum):
    """Имена стандартных параметров модели (ключи в optimize / fixed / bounds)."""

    CONTACT_ANGLE = "theta"
    WALL_LAYER_THICKNESS = "delta"
    VOLUME_FRACTION = "phi"
    PROFILE_SHIFT = "shift"


class Unit(str, Enum):
    """Единицы параметра: в чём его задаёт пользователь."""

    DEGREE = "degree"  # внутри формул — радианы
    PORE_SIZE = "pore_size"  # доли характерного размера поры: H у щели, a у многоугольников
    DIMENSIONLESS = "dimensionless"


class Regime(str, Enum):
    """Режим смачивания анализируемой жидкости."""

    WETTING = "wetting"  # θ < 90°
    NON_WETTING = "non_wetting"  # θ > 90°


class Frame(str, Enum):
    """Система отсчёта координаты профиля."""

    PORE = "pore"  # 0 — центр поры (центроид для многоугольников)
    LIQUID = "liquid"  # 0 — центр анализируемой жидкости (среднее по выбранным атомам)


class Normalization(str, Enum):
    """Как данные приводятся к безразмерной плотности."""

    AUTO = "auto"  # BULK, если задана объёмная плотность, иначе MEAN
    BULK = "bulk"  # ρ / ρ_bulk против ρ_model
    MEAN = "mean"  # ρ / n̄ против ρ_model / φ, n̄ = N / V


class LossFunction(str, Enum):
    L1 = "l1"  # среднее модуля отклонения (статьи PANDA)
    L2 = "l2"  # среднее квадрата отклонения


class OptimizerMethod(str, Enum):
    """Локальный метод scipy.optimize.minimize. По умолчанию Nelder–Mead: L1-ошибка негладкая,
    градиентный L-BFGS-B на синтетике застревает (сравнение — tests/README.md, раздел «Оптимизатор»)."""

    NELDER_MEAD = "Nelder-Mead"
    POWELL = "Powell"
    LBFGSB = "L-BFGS-B"


class Axis(str, Enum):
    """Ось прямоугольного ящика моделирования."""

    X = "x"
    Y = "y"
    Z = "z"

    @property
    def column(self) -> int:
        """Номер столбца координаты в массиве позиций (x — 0, y — 1, z — 2)."""
        return AXIS_ORDER.index(self)


AXIS_ORDER = (Axis.X, Axis.Y, Axis.Z)


class ProfileCentering(str, Enum):
    """Что ставится в 0 координаты профиля в каждом кадре."""

    PORE = "pore"  # середина между стенками (или заданный центр поры) — система PORE
    LIQUID = "liquid"  # центр выбранных атомов (без весов) с учётом PBC — система LIQUID


class ParticleGrouping(str, Enum):
    """Что считается частицей профиля."""

    ATOM = "atom"  # каждый выбранный атом
    MOLECULE = "molecule"  # центр масс молекулы (меньше молекулярной «ряби» у стенок)


class Orientation(str, Enum):
    """Отражать ли профиль z → −z в системе поры."""

    AUTO = "auto"  # отразить, если жидкость у верхней стенки (первый момент среднего профиля > 0)
    KEEP = "keep"  # как в траектории


class TimeUnit(str, Enum):
    """Единицы времени в строках вида «2 ns»; внутри всё в пикосекундах."""

    FEMTOSECOND = "fs"
    PICOSECOND = "ps"
    NANOSECOND = "ns"
    MICROSECOND = "us"

    @property
    def picoseconds(self) -> float:
        return TIME_UNIT_PICOSECONDS[self]


TIME_UNIT_PICOSECONDS = {
    TimeUnit.FEMTOSECOND: 1e-3,
    TimeUnit.PICOSECOND: 1.0,
    TimeUnit.NANOSECOND: 1e3,
    TimeUnit.MICROSECOND: 1e6,
}
```

Заменить `src/panda_md/core/pbc.py` целиком:

`src/panda_md/core/pbc.py`:

```python
"""Периодические граничные условия вдоль одной оси ящика: заворот, круговое среднее, наибольший пустой промежуток."""

import math

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.types import FloatArray, IndexArray


def wrap_centered(values: FloatArray, period: float | FloatArray) -> FloatArray:
    """Завернуть координаты в [−L/2, L/2) — ближайший образ относительно нуля (L — число или по оси на столбец)."""
    wrapped: FloatArray = values - period * np.floor(values / period + 0.5)
    return wrapped


def circular_mean(values: FloatArray, period: float) -> float:
    """Центр облака точек на окружности длины L: среднее направление, результат в [0, L).

    Так центр жидкости, пересекающей границу ящика, не попадает в середину пустоты (как в старой PANDA,
    `utils.get_center_pbc`).
    """
    if values.size == 0:
        raise InvalidTrajectoryError("центр по пустой выборке не определён")
    angles = 2.0 * math.pi * values / period
    mean_angle = math.atan2(-float(np.mean(np.sin(angles))), -float(np.mean(np.cos(angles)))) + math.pi
    return period * mean_angle / (2.0 * math.pi)


def largest_gap(values: FloatArray, period: float) -> tuple[float, float]:
    """Наибольший промежуток без точек на окружности длины L: (начало, конец), конец > начала, конец < начало + L.

    Стенки щели — атомы подложки: пора — самый широкий пустой промежуток между ними, в том числе через границу
    ящика (одна подложка и её образ).
    """
    return empty_gaps(values, period)[0]


def empty_gaps(values: FloatArray, period: float) -> list[tuple[float, float]]:
    """Все промежутки между соседними точками на окружности длины L, от широкого к узкому: (начало, конец)."""
    if values.size == 0:
        raise InvalidTrajectoryError("в выборке стенок нет атомов")
    ordered = np.sort(np.mod(values, period))
    widths = np.diff(np.append(ordered, ordered[0] + period))
    order = np.argsort(-widths, kind="stable")
    return [(float(ordered[index]), float(ordered[index] + widths[index])) for index in order]


def molecule_centers(
    positions: FloatArray, box: FloatArray, molecule_indices: IndexArray, masses: FloatArray
) -> FloatArray:
    """Центры масс молекул (n_молекул × 3), в ящике. Атомы каждой молекулы сначала собираются к её первому атому
    по ближайшему образу — молекула, разрезанная границей ящика, не «размазывается» на весь ящик.

    molecule_indices — номер молекулы каждого атома (0 … n_молекул − 1, атомы одной молекулы — где угодно).
    """
    count = int(np.max(molecule_indices)) + 1
    first_atom = np.full(count, -1, dtype=np.int64)
    first_atom[molecule_indices[::-1]] = np.arange(molecule_indices.size)[::-1]
    reference = positions[first_atom[molecule_indices]]
    unwrapped = reference + wrap_centered(positions - reference, box)
    total_mass = np.bincount(molecule_indices, weights=masses, minlength=count)
    if np.any(total_mass <= 0.0):
        raise InvalidTrajectoryError("у молекулы нулевая масса — проверь массы атомов")
    centers = np.column_stack(
        [np.bincount(molecule_indices, weights=masses * unwrapped[:, axis], minlength=count) for axis in range(3)]
    )
    wrapped: FloatArray = np.mod(centers / total_mass[:, np.newaxis], box)
    return wrapped
```

Заменить `src/panda_md/core/trajectory.py` целиком:

`src/panda_md/core/trajectory.py`:

```python
"""Траектория как последовательность кадров: интерфейс, общий для файлов МД и массивов numpy.

Реализации — `panda_md.trajectory` (`ArrayTrajectory`, `MDATrajectory`). Ядро и поры знают только этот интерфейс,
поэтому MDAnalysis нужен лишь для чтения файлов. Все длины — нм, время — пс, ящик прямоугольный.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np
from pydantic import Field

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.timing import TimeValue, to_picoseconds
from panda_md.core.types import Array1D, Array2D, FloatArray, IndexArray, PandaModel

BOX_DIMENSIONS = 3
TIME_TOLERANCE_PS = 1e-6  # кадр на границе интервала времени включается


class TrajectoryFrame(PandaModel):
    """Один кадр: время, ящик и позиции запрошенных групп атомов (в порядке запроса)."""

    index: int = Field(ge=0)  # номер кадра в траектории
    time_ps: float
    box_nm: Array1D  # длины рёбер прямоугольного ящика
    positions_nm: tuple[Array2D, ...]  # по одному массиву (n, 3) на группу


class Trajectory(ABC):
    """Источник кадров. Выборка атомов — строка (язык выборки бэкенда) или явные индексы."""

    @abstractmethod
    def select(self, selection: str | Sequence[int]) -> IndexArray:
        """Индексы атомов выборки. Пустая выборка — InvalidTrajectoryError."""

    @abstractmethod
    def molecules(self, indices: IndexArray) -> tuple[IndexArray, FloatArray]:
        """Для атомов выборки: номер молекулы (0 … n − 1 в порядке первого появления) и масса атома."""

    @abstractmethod
    def times_ps(self) -> FloatArray:
        """Время каждого кадра, пс."""

    @abstractmethod
    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        """Прочитать кадры по номерам (по возрастанию), отдать позиции групп."""

    def frames(
        self,
        groups: Sequence[IndexArray],
        *,
        start: TimeValue = 0.0,
        stop: TimeValue | None = None,
        stride: int = 1,
    ) -> Iterator[TrajectoryFrame]:
        """Кадры с start ≤ t ≤ stop, каждый stride-й; потоково, без загрузки всей траектории в память."""
        indices = self.frame_indices(start=start, stop=stop, stride=stride)
        return self._read(indices, groups)

    def frame_indices(self, *, start: TimeValue = 0.0, stop: TimeValue | None = None, stride: int = 1) -> list[int]:
        if stride < 1:
            raise ValueError(f"stride = {stride}: шаг по кадрам должен быть ≥ 1")
        times = self.times_ps()
        lower = to_picoseconds(start)
        upper = float(times[-1]) if stop is None else to_picoseconds(stop)
        inside = np.flatnonzero((times >= lower - TIME_TOLERANCE_PS) & (times <= upper + TIME_TOLERANCE_PS))
        if inside.size == 0:
            raise InvalidTrajectoryError(
                f"в интервале [{lower:g}, {upper:g}] пс нет кадров; траектория — [{times[0]:g}, {times[-1]:g}] пс"
            )
        return [int(index) for index in inside[::stride]]

    def frame_at(self, time: TimeValue, groups: Sequence[IndexArray]) -> TrajectoryFrame:
        """Первый кадр с t ≥ time."""
        return next(self._read(self.frame_indices(start=time)[:1], groups))


def check_box(box_nm: FloatArray) -> None:
    if box_nm.shape != (BOX_DIMENSIONS,) or np.any(box_nm <= 0.0):
        raise InvalidTrajectoryError(f"ящик {box_nm.tolist()}: нужны три положительные длины прямоугольного ящика")


def renumber(labels: IndexArray) -> IndexArray:
    """Метки молекул → 0 … n − 1 в порядке первого появления."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    renumbered: IndexArray = order[inverse].astype(np.int64)
    return renumbered
```

Заменить `src/panda_md/core/pore.py` целиком:

`src/panda_md/core/pore.py`:

```python
"""Тип поры: геометрия, положение в ящике моделирования, приведение профиля к безразмерному виду, реестр форм."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from panda_md.core.enums import Axis, Normalization
from panda_md.core.errors import NotDerivedError, UnknownShapeError
from panda_md.core.profile import DensityProfile, ReducedProfile, reduce_profile
from panda_md.core.shape import InterfaceShape, PoreGeometry
from panda_md.core.types import FloatArray, PandaModel

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel

# Any: реестр хранит формы с разными подклассами геометрии; конкретный тип известен только самой поре.
ShapeClassT = TypeVar("ShapeClassT", bound=type[InterfaceShape[Any]])


class Pore(PandaModel, ABC):
    """Конкретная пора с размерами. Каждый подкласс имеет свой реестр форм."""

    name: ClassVar[str]
    shape_registry: ClassVar[dict[str, type[InterfaceShape[Any]]]]
    normal_axis: Axis = Axis.Z  # ось ящика, вдоль которой строится профиль
    wall_selection: str | None = None  # атомы стенок: по ним в каждом кадре ищется центр поры

    @classmethod
    def own_registry(cls) -> dict[str, type[InterfaceShape[Any]]]:
        """Реестр именно этого класса поры: подклассы не делят реестр с родителем."""
        if "shape_registry" not in cls.__dict__:
            cls.shape_registry = {}
        return cls.shape_registry

    @classmethod
    def register(cls, shape_class: ShapeClassT) -> ShapeClassT:
        """Декоратор: подключить форму к этому типу поры."""
        registry = cls.own_registry()
        if shape_class.name in registry:
            raise ValueError(f"форма {shape_class.name} уже зарегистрирована в поре {cls.name}")
        registry[shape_class.name] = shape_class
        return shape_class

    @classmethod
    def shape_names(cls) -> tuple[str, ...]:
        return tuple(cls.own_registry())

    @abstractmethod
    def geometry(self) -> PoreGeometry:
        """Безразмерная геометрия для формул форм."""

    @abstractmethod
    def characteristic_size_nm(self) -> float:
        """Размер, к которому приводятся координата и δ: H у щели, a у многоугольников."""

    @abstractmethod
    def volume_nm3(self) -> float:
        """Объём поры, относительно которого задана φ."""

    @abstractmethod
    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Площадь сечения слоя-бина (бин — слой толщиной dz вдоль оси профиля)."""

    def base_area_nm2(self) -> float | None:
        """Площадь основания ящика, которую ожидает пора (для проверки размеров); None — от ящика не зависит."""
        return None

    def coordinate_bounds_nm(self) -> tuple[float, float]:
        """Стенки поры вдоль оси профиля в системе поры, нм."""
        lower, upper = self.geometry().coordinate_bounds()
        size = self.characteristic_size_nm()
        return lower * size, upper * size

    def expected_wall_gap_nm(self) -> float:
        """Пустой промежуток между центрами крайних атомов стенок, который ожидает пора."""
        lower, upper = self.coordinate_bounds_nm()
        return upper - lower

    def locate_center_nm(self, wall_coordinates_nm: FloatArray, period_nm: float) -> float:
        """Положение центра поры вдоль оси профиля по координатам атомов стенок (с учётом PBC)."""
        raise NotDerivedError(
            f"поиск центра поры {self.name} по атомам стенок не реализован: задай центр явно (pore_center_nm)"
        )

    def reduce(
        self, profile: DensityProfile, normalization: Normalization, density_bulk_per_nm3: float | None
    ) -> ReducedProfile:
        return reduce_profile(profile, self.characteristic_size_nm(), normalization, density_bulk_per_nm3)

    def interface(self, shape_name: str) -> "InterfaceModel":
        from panda_md.core.model import InterfaceModel

        registry = self.own_registry()
        if shape_name not in registry:
            raise UnknownShapeError(f"в поре {self.name} нет формы {shape_name}; доступны: {list(registry)}")
        return InterfaceModel(self, registry[shape_name]())

    def available_interfaces(self) -> list["InterfaceModel"]:
        return [self.interface(shape_name) for shape_name in self.own_registry()]
```

Заменить `src/panda_md/pores/slit/pore.py` целиком:

`src/panda_md/pores/slit/pore.py`:

```python
"""Щелевая пора: две плоские стенки на расстоянии H, прямоугольное основание L × kL (0 < k ≤ 1).

Параметр k (squeeze parameter) — Shistko et al., Mol. Simul. (2025), DOI 10.1080/08927022.2025.2574643.
"""

import math
import warnings
from typing import ClassVar

import numpy as np
from pydantic import Field

from panda_md.core.enums import AXIS_ORDER, Axis
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.pbc import empty_gaps, wrap_centered
from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.core.types import FloatArray

SECOND_GAP_WARNING_NM = 1.0  # внутри подложек промежутки < 0.5 нм (графит, SOLD, кальцит); второй шире 1 нм — вакуум?
SLIT_HALF_WIDTH = 0.5  # стенки щели в безразмерной координате: z = ±0.5


class SlitGeometry(PoreGeometry):
    """Безразмерная геометрия щели: l = L / H — длинная сторона основания, k — короткая сторона / длинная."""

    pore_length_ratio: float = Field(gt=0.0)  # l = L / H
    base_aspect_ratio: float = Field(default=1.0, gt=0.0, le=1.0)  # k: основание L × kL

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SLIT_HALF_WIDTH, SLIT_HALF_WIDTH)

    def profile_integral(self) -> float:
        return 1.0  # ∫ρ dz = φ: профиль нормирован на всю площадь основания

    def base_area(self) -> float:
        """k l² — площадь основания ячейки в единицах H²."""
        return self.base_aspect_ratio * self.pore_length_ratio**2

    def short_side_ratio(self) -> float:
        """k l — короткая сторона основания в единицах H."""
        return self.base_aspect_ratio * self.pore_length_ratio

    def equivalent_square(self) -> "SlitGeometry":
        """Квадратная ячейка со стороной kL.

        Осесимметричная форма, не касающаяся боковых границ ячейки, «видит» только её короткую сторону: в ячейке
        L × kL она та же, что в квадратной ячейке kL × kL с объёмной долей, пересчитанной на площадь основания.
        """
        return SlitGeometry(pore_length_ratio=self.short_side_ratio())


class SlitPore(Pore):
    name: ClassVar[str] = "slit"

    width_nm: float = Field(gt=0.0)  # H — расстояние между стенками
    length_nm: float = Field(gt=0.0)  # L — длинная сторона основания ячейки
    base_aspect_ratio: float = Field(default=1.0, gt=0.0, le=1.0)  # k — короткая сторона / длинная
    wall_offset_nm: float = Field(default=0.0, ge=0.0)  # поверхность стенки — дальше центров крайних атомов

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=self.length_nm / self.width_nm, base_aspect_ratio=self.base_aspect_ratio)

    def characteristic_size_nm(self) -> float:
        return self.width_nm

    def volume_nm3(self) -> float:
        return self.base_aspect_ratio * self.length_nm**2 * self.width_nm

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Площадь основания ящика — произведение двух рёбер, перпендикулярных оси профиля."""
        return math.prod(float(box_nm[axis.column]) for axis in AXIS_ORDER if axis is not self.normal_axis)

    def base_area_nm2(self) -> float:
        """k L² — основание ячейки; должно совпадать с основанием ящика."""
        return self.base_aspect_ratio * self.length_nm**2

    def expected_wall_gap_nm(self) -> float:
        """H + 2 · wall_offset: поверхности стенок сдвинуты внутрь от центров крайних атомов."""
        return self.width_nm + 2.0 * self.wall_offset_nm

    def locate_center_nm(self, wall_coordinates_nm: FloatArray, period_nm: float) -> float:
        """Середина пустого промежутка между атомами стенок, ширина которого ближе всего к H, в [0, L).

        Промежуток может проходить через границу ящика; вакуумный слой другой ширины не выбирается.
        """
        start, end = min(
            empty_gaps(wall_coordinates_nm, period_nm),
            key=lambda gap: abs(gap[1] - gap[0] - self.expected_wall_gap_nm()),
        )
        return (0.5 * (start + end)) % period_nm

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        walls: str,
        length_axis: Axis = Axis.X,
        normal_axis: Axis = Axis.Z,
        time: TimeValue = 0.0,
        liquid: str | None = None,
        wall_offset_nm: float = 0.0,
    ) -> "SlitPore":
        """Щель по кадру траектории: H — промежуток между атомами стенок (поверхности — по центрам крайних атомов),
        L — ребро ящика вдоль length_axis, k — второе ребро основания / L.

        Промежуток — наибольший, а если задана выборка жидкости `liquid` — тот, в котором её больше всего (нужно,
        когда в ящике есть вакуумный слой). wall_offset_nm — на сколько поверхность каждой стенки отстоит от центров
        её крайних атомов (например, ван-дер-ваальсов радиус): H = промежуток − 2 · wall_offset_nm.
        """
        if length_axis is normal_axis:
            raise InvalidTrajectoryError(f"ось длины {length_axis.value} совпадает с осью профиля")
        groups = [trajectory.select(walls)] + ([] if liquid is None else [trajectory.select(liquid)])
        frame = trajectory.frame_at(time, groups)
        box = frame.box_nm
        period = float(box[normal_axis.column])
        gaps = empty_gaps(frame.positions_nm[0][:, normal_axis.column], period)
        if liquid is None:
            start, end = gaps[0]
            if len(gaps) > 1 and gaps[1][1] - gaps[1][0] >= SECOND_GAP_WARNING_NM:
                warnings.warn(
                    f"между стенками два широких промежутка ({end - start:.3f} и {gaps[1][1] - gaps[1][0]:.3f} нм) — "
                    f"возможно, в ящике вакуум; выбран наибольший, задай liquid=<выборка>, чтобы выбрать промежуток "
                    f"с жидкостью",
                    PandaWarning,
                    stacklevel=2,
                )
        else:
            heights = frame.positions_nm[1][:, normal_axis.column]

            def liquid_inside(gap: tuple[float, float]) -> int:
                offsets = wrap_centered(heights - 0.5 * (gap[0] + gap[1]), period)
                return int(np.count_nonzero(np.abs(offsets) < 0.5 * (gap[1] - gap[0])))

            start, end = max(gaps, key=liquid_inside)
        (other_axis,) = (axis for axis in AXIS_ORDER if axis not in (length_axis, normal_axis))
        length = float(box[length_axis.column])
        aspect_ratio = float(box[other_axis.column]) / length
        if aspect_ratio > 1.0:
            raise InvalidTrajectoryError(
                f"ребро {other_axis.value} ({box[other_axis.column]:.3f} нм) длиннее {length_axis.value} "
                f"({length:.3f} нм): L — длинная сторона основания, задай length_axis={other_axis.value!r}"
            )
        width = end - start - 2.0 * wall_offset_nm
        if width <= 0.0:
            raise InvalidTrajectoryError(
                f"wall_offset_nm = {wall_offset_nm} нм не меньше половины промежутка {end - start:.3f} нм"
            )
        return cls(
            width_nm=width,
            wall_offset_nm=wall_offset_nm,
            length_nm=length,
            base_aspect_ratio=aspect_ratio,
            normal_axis=normal_axis,
            wall_selection=walls,
        )
```

Заменить `src/panda_md/trajectory/arrays.py` целиком:

`src/panda_md/trajectory/arrays.py`:

```python
"""Траектория из массивов numpy — для синтетических данных и тестов (MDAnalysis не нужен)."""

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.trajectory import BOX_DIMENSIONS, Trajectory, TrajectoryFrame, renumber
from panda_md.core.types import Array1D, Array2D, Array3D, FloatArray, IndexArray, IndexArray1D, PandaModel


class ArrayTrajectory(PandaModel, Trajectory):
    """Кадры из массивов: позиции (кадры × атомы × 3, нм), ящики (кадры × 3, нм), времена (пс).

    Выборка — явные индексы или имя группы из `groups` (например, {"water": [...], "walls": [...]}).
    """

    positions_nm: Array3D
    boxes_nm: Array2D
    times_ps_values: Array1D = Field(alias="times_ps")
    groups: Mapping[str, IndexArray1D] = Field(default_factory=dict)
    molecule_ids: IndexArray1D | None = None  # номер молекулы каждого атома — для per="molecule"
    masses: Array1D | None = None  # масса каждого атома (а. е. м.)

    @model_validator(mode="after")
    def _check_shapes(self) -> "ArrayTrajectory":
        frame_count, atom_count, dimensions = self.positions_nm.shape
        if dimensions != BOX_DIMENSIONS or frame_count == 0:
            raise ValueError(f"позиции формы {self.positions_nm.shape}: нужно (кадры ≥ 1, атомы, 3)")
        if self.boxes_nm.shape != (frame_count, BOX_DIMENSIONS) or np.any(self.boxes_nm <= 0.0):
            raise ValueError(f"ящики формы {self.boxes_nm.shape}: нужно ({frame_count}, 3) с положительными длинами")
        if self.times_ps_values.shape != (frame_count,) or np.any(np.diff(self.times_ps_values) <= 0.0):
            raise ValueError("времена: по одному на кадр, строго по возрастанию")
        for name, values in (("molecule_ids", self.molecule_ids), ("masses", self.masses)):
            if values is not None and values.shape != (atom_count,):
                raise ValueError(f"{name}: по одному значению на атом ({atom_count})")
        for name, indices in self.groups.items():
            if indices.size and int(np.max(indices)) >= atom_count:
                raise ValueError(f"группа {name}: индекс {int(np.max(indices))} при {atom_count} атомах")
        return self

    def select(self, selection: str | Sequence[int]) -> IndexArray:
        if isinstance(selection, str):
            if selection not in self.groups:
                raise InvalidTrajectoryError(f"группы {selection!r} нет; доступны: {list(self.groups)}")
            indices = self.groups[selection]
        else:
            indices = np.asarray(selection, dtype=np.int64)
            if indices.size and (int(np.min(indices)) < 0 or int(np.max(indices)) >= self.positions_nm.shape[1]):
                raise InvalidTrajectoryError(f"индексы выборки вне [0, {self.positions_nm.shape[1]})")
        if indices.size == 0:
            raise InvalidTrajectoryError(f"выборка {selection!r} пуста")
        return indices

    def molecules(self, indices: IndexArray) -> tuple[IndexArray, FloatArray]:
        if self.molecule_ids is None or self.masses is None:
            raise InvalidTrajectoryError('для per="molecule" задай molecule_ids и masses траектории')
        return renumber(self.molecule_ids[indices]), self.masses[indices]

    def times_ps(self) -> FloatArray:
        return self.times_ps_values

    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        for index in frame_indices:
            frame_positions = self.positions_nm[index]
            yield TrajectoryFrame(
                index=index,
                time_ps=float(self.times_ps_values[index]),
                box_nm=self.boxes_nm[index],
                positions_nm=tuple(frame_positions[group] for group in groups),
            )
```

Заменить `src/panda_md/trajectory/reader.py` целиком:

`src/panda_md/trajectory/reader.py`:

```python
"""Чтение файлов МД через MDAnalysis (необязательная зависимость: `pip install panda-md[md]`).

MDAnalysis хранит длины в ångström; наружу отдаются нм. Единственный модуль пакета, импортирующий MDAnalysis.
"""

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError, PandaWarning
from panda_md.core.trajectory import Trajectory, TrajectoryFrame, check_box, renumber
from panda_md.core.types import FloatArray, IndexArray

ANGSTROM_PER_NM = 10.0
RECTANGULAR_ANGLE_DEG = 90.0
ANGLE_TOLERANCE_DEG = 1e-3
TIME_MISMATCH_PS = 1e-3  # время последнего кадра расходится с t0 + dt·(n − 1)


def _import_mdanalysis() -> Any:  # noqa: ANN401 — у MDAnalysis нет аннотаций типов
    try:
        import MDAnalysis
    except ImportError as error:
        raise MissingDependencyError("для чтения файлов МД нужен MDAnalysis: pip install 'panda-md[md]'") from error
    return MDAnalysis


class MDATrajectory(Trajectory):
    """Траектория из объекта MDAnalysis.Universe."""

    universe: Any  # Any: MDAnalysis.Universe без аннотаций типов

    def __init__(self, universe: Any) -> None:  # noqa: ANN401 — MDAnalysis.Universe без аннотаций типов
        self.universe = universe
        self._times: FloatArray | None = None

    def __repr__(self) -> str:
        return f"MDATrajectory({self.universe.atoms.n_atoms} атомов, {len(self.universe.trajectory)} кадров)"

    def select(self, selection: str | Sequence[int]) -> IndexArray:
        if isinstance(selection, str):
            try:
                indices = np.asarray(self.universe.select_atoms(selection).indices, dtype=np.int64)
            except Exception as error:  # MDAnalysis бросает разные исключения на синтаксис выборки
                raise InvalidTrajectoryError(f"выборка {selection!r} не разобрана MDAnalysis: {error}") from error
        else:
            indices = np.asarray(selection, dtype=np.int64)
            atom_count = int(self.universe.atoms.n_atoms)
            if indices.size and (int(np.min(indices)) < 0 or int(np.max(indices)) >= atom_count):
                raise InvalidTrajectoryError(f"индексы выборки вне [0, {atom_count})")
        if indices.size == 0:
            raise InvalidTrajectoryError(f"выборка {selection!r} пуста — проверь имена остатков и атомов")
        return indices

    def molecules(self, indices: IndexArray) -> tuple[IndexArray, FloatArray]:
        atoms = self.universe.atoms[indices]
        masses = np.asarray(atoms.masses, dtype=np.float64)
        return renumber(np.asarray(atoms.resindices, dtype=np.int64)), masses

    def times_ps(self) -> FloatArray:
        """Время кадров: t0 + dt·i, если с ним согласуется время последнего кадра; иначе (склеенные траектории
        с разрывами или перекрытием) — настоящее время каждого кадра, прочитанное один раз."""
        if self._times is not None:
            return self._times
        reader = self.universe.trajectory
        times: FloatArray = reader[0].time + reader.dt * np.arange(len(reader), dtype=np.float64)
        if abs(float(reader[len(reader) - 1].time) - float(times[-1])) > TIME_MISMATCH_PS:
            times = np.array([float(timestep.time) for timestep in reader], dtype=np.float64)
            warnings.warn(
                f"шаг по времени непостоянен (ожидалось {reader.dt:g} пс): время кадров прочитано из траектории",
                PandaWarning,
                stacklevel=2,
            )
        self._times = times
        return times

    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        for timestep in self.universe.trajectory[list(frame_indices)]:
            dimensions = timestep.dimensions
            if dimensions is None or np.any(np.abs(dimensions[3:] - RECTANGULAR_ANGLE_DEG) > ANGLE_TOLERANCE_DEG):
                raise InvalidTrajectoryError(f"кадр {timestep.frame}: ящик не прямоугольный ({dimensions})")
            box_nm = np.asarray(dimensions[:3], dtype=np.float64) / ANGSTROM_PER_NM
            check_box(box_nm)
            yield TrajectoryFrame(
                index=int(timestep.frame),
                time_ps=float(timestep.time),
                box_nm=box_nm,
                positions_nm=tuple(
                    np.asarray(timestep.positions[group], dtype=np.float64) / ANGSTROM_PER_NM for group in groups
                ),
            )


def load_trajectory(topology: str | Path, *trajectory_files: str | Path) -> MDATrajectory:
    """Открыть траекторию МД: топология (.tpr, .gro, .pdb, …) и файлы траектории (.xtc, .trr, …)."""
    mdanalysis = _import_mdanalysis()
    paths = [str(path) for path in (topology, *trajectory_files)]
    for path in paths:
        if not Path(path).exists():
            raise InvalidTrajectoryError(f"файл {path} не найден")
    return MDATrajectory(mdanalysis.Universe(*paths))
```

Заменить `src/panda_md/trajectory/density.py` целиком:

`src/panda_md/trajectory/density.py`:

```python
"""Профиль плотности по траектории: выборка → центр в каждом кадре → координата и бины поры → средние по блокам.

Чтение потоковое: в памяти только текущий кадр и суммы по блокам.
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np

from panda_md.core.enums import Frame, Orientation, ParticleGrouping, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.pbc import circular_mean, molecule_centers, wrap_centered
from panda_md.core.pore import Pore
from panda_md.core.profile import ProfileSet
from panda_md.core.timing import TimeValue, to_picoseconds
from panda_md.core.trajectory import Trajectory, TrajectoryFrame, check_box
from panda_md.core.types import FloatArray, IndexArray

DEFAULT_BIN_COUNT = 200
WALL_GAP_TOLERANCE = 0.02  # промежуток между стенками в кадре отличается от H поры больше чем на 2 % — предупреждение
BOX_AREA_TOLERANCE = 0.01  # площадь основания ящика отличается от основания поры больше чем на 1 % — предупреждение
OUTSIDE_FRACTION_WARNING = 0.01  # больше 1 % выбранных частиц вне поры — предупреждение
BOX_CHANGE_TOLERANCE = 0.01  # в системе жидкости бины — по ящику первого кадра; изменение больше 1 % — предупреждение


class _BlockAccumulator:
    """Суммы по кадрам текущего блока."""

    def __init__(self, bin_count: int) -> None:
        self.density_sum: FloatArray = np.zeros(bin_count)
        self.particle_sum: float = 0.0
        self.frame_count: int = 0
        self.start_ps: float = math.nan
        self.stop_ps: float = math.nan

    def add(self, density: FloatArray, particles: int, time_ps: float) -> None:
        if self.frame_count == 0:
            self.start_ps = time_ps
        self.density_sum += density
        self.particle_sum += particles
        self.frame_count += 1
        self.stop_ps = time_ps


def density_profile(
    trajectory: Trajectory,
    pore: Pore,
    select: str | Sequence[int],
    *,
    centering: ProfileCentering | str = ProfileCentering.PORE,
    bins: int = DEFAULT_BIN_COUNT,
    bin_width_nm: float | None = None,
    start: TimeValue = 0.0,
    stop: TimeValue | None = None,
    block: TimeValue | None = None,
    stride: int = 1,
    pore_center_nm: float | None = None,
    orientation: Orientation | str = Orientation.AUTO,
    per: ParticleGrouping | str = ParticleGrouping.ATOM,
) -> ProfileSet:
    """Профили плотности выбранных частиц вдоль оси поры, усреднённые по блокам времени длины `block`.

    centering=PORE: 0 — центр поры; он берётся из `pore_center_nm` или в каждом кадре из атомов стенок
    (`pore.wall_selection`); бины выровнены по стенкам поры. centering=LIQUID: 0 — центр выборки (среднее
    по атомам без весов) с учётом PBC; бины покрывают весь ящик вдоль оси профиля. Неполный последний блок
    отбрасывается. per=MOLECULE: частица — центр масс молекулы (с учётом PBC), N — число молекул.
    """
    centering = ProfileCentering(centering)
    orientation = Orientation(orientation)
    grouping = ParticleGrouping(per)
    liquid = trajectory.select(select)
    molecules = trajectory.molecules(liquid) if grouping is ParticleGrouping.MOLECULE else None
    groups: list[IndexArray] = [liquid]
    use_walls = centering is ProfileCentering.PORE and pore_center_nm is None
    if use_walls:
        if pore.wall_selection is None:
            raise InvalidTrajectoryError(
                "центрирование по поре: задай pore_center_nm или выборку стенок (wall_selection у поры)"
            )
        groups.append(trajectory.select(pore.wall_selection))
    frame_indices = trajectory.frame_indices(start=start, stop=stop, stride=stride)
    times = trajectory.times_ps()[frame_indices]
    block_ps = None if block is None else to_picoseconds(block)
    block_of_frame, block_count = _assign_blocks(times, block_ps)

    axis = pore.normal_axis.column
    edges: FloatArray | None = None
    first_period = 0.0
    blocks: list[_BlockAccumulator] = []
    warned: set[str] = set()
    for frame, block_index in zip(
        trajectory.frames(groups, start=start, stop=stop, stride=stride), block_of_frame, strict=True
    ):
        if block_index < 0:
            continue
        check_box(frame.box_nm)
        period = float(frame.box_nm[axis])
        heights = _particle_heights(frame.positions_nm[0], frame.box_nm, axis, molecules)
        center = _frame_center(frame, heights, pore, centering, pore_center_nm, period, warned)
        coordinate = wrap_centered(heights - center, period)
        if edges is None:
            first_period = period
            edges = _bin_edges(pore, centering, period, bins, bin_width_nm)
            blocks = [_BlockAccumulator(edges.size - 1) for _ in range(block_count)]
        counts, _ = np.histogram(coordinate, bins=edges)
        if centering is ProfileCentering.LIQUID:
            _check_box_change(frame.index, period, first_period, int(counts.sum()) < coordinate.size, warned)
        area = pore.bin_cross_section_nm2(frame.box_nm)
        _check_area(pore, area, warned)
        density = counts / (area * np.diff(edges))
        inside = int(counts.sum()) if centering is ProfileCentering.PORE else coordinate.size
        if centering is ProfileCentering.PORE and "outside" not in warned:
            outside_fraction = 1.0 - inside / coordinate.size
            if outside_fraction > OUTSIDE_FRACTION_WARNING:
                warned.add("outside")
                warnings.warn(
                    f"{outside_fraction:.1%} выбранных частиц вне поры — проверь выборку и стенки",
                    PandaWarning,
                    stacklevel=2,
                )
        blocks[block_index].add(density, inside, frame.time_ps)

    assert edges is not None  # frame_indices не пуст
    densities = np.array([accumulator.density_sum / accumulator.frame_count for accumulator in blocks])
    frame_reference = Frame.PORE if centering is ProfileCentering.PORE else Frame.LIQUID
    flipped = False
    if orientation is Orientation.AUTO and frame_reference is Frame.PORE and math.isclose(edges[0], -edges[-1]):
        centers = 0.5 * (edges[:-1] + edges[1:])
        if float(np.sum(centers * densities.mean(axis=0))) > 0.0:
            densities = densities[:, ::-1]
            flipped = True
    return ProfileSet(
        bin_edges_nm=edges,
        block_densities_per_nm3=densities,
        block_start_ps=np.array([accumulator.start_ps for accumulator in blocks]),
        block_stop_ps=np.array([accumulator.stop_ps for accumulator in blocks]),
        block_frame_counts=tuple(accumulator.frame_count for accumulator in blocks),
        block_particle_counts=np.array([accumulator.particle_sum / accumulator.frame_count for accumulator in blocks]),
        frame=frame_reference,
        centering=centering,
        selection=select if isinstance(select, str) else f"{len(liquid)} индексов",
        pore_volume_nm3=pore.volume_nm3(),
        flipped=flipped,
    )


def _assign_blocks(times: FloatArray, block_ps: float | None) -> tuple[list[int], int]:
    """Номер блока каждого кадра (−1 — кадр неполного последнего блока) и число полных блоков."""
    if block_ps is None:
        return [0] * times.size, 1
    if block_ps <= 0.0:
        raise ValueError(f"длина блока {block_ps} пс должна быть положительной")
    if times.size == 1:
        warnings.warn(
            f"один кадр: длину блока {block_ps:g} пс проверить не по чему — профиль из одного блока",
            PandaWarning,
            stacklevel=3,
        )
        return [0], 1
    spacing = float(np.median(np.diff(times)))
    elapsed = times - times[0]
    block_count = math.floor((elapsed[-1] + spacing) / block_ps + 1e-9)
    if block_count == 0:
        raise InvalidTrajectoryError(
            f"интервал {elapsed[-1] + spacing:g} пс короче одного блока {block_ps:g} пс — уменьши block"
        )
    assignment = [math.floor(value / block_ps + 1e-9) for value in elapsed]
    dropped = sum(1 for index in assignment if index >= block_count)
    if dropped > 1:  # один кадр ровно на границе — начало следующего блока, а не неполный блок
        warnings.warn(f"{dropped} последних кадров не составляют полного блока и не учтены", PandaWarning, stacklevel=3)
    return [index if index < block_count else -1 for index in assignment], block_count


def _particle_heights(
    positions: FloatArray, box: FloatArray, axis: int, molecules: tuple[IndexArray, FloatArray] | None
) -> FloatArray:
    """Координата частиц вдоль оси профиля: атомов или центров масс молекул."""
    if molecules is None:
        heights: FloatArray = positions[:, axis]
        return heights
    molecule_indices, masses = molecules
    centers: FloatArray = molecule_centers(positions, box, molecule_indices, masses)[:, axis]
    return centers


def _frame_center(
    frame: TrajectoryFrame,
    heights: FloatArray,
    pore: Pore,
    centering: ProfileCentering,
    pore_center_nm: float | None,
    period: float,
    warned: set[str],
) -> float:
    axis = pore.normal_axis.column
    if centering is ProfileCentering.LIQUID:
        return circular_mean(heights, period)
    if pore_center_nm is not None:
        return pore_center_nm
    walls = frame.positions_nm[1][:, axis]
    center = pore.locate_center_nm(walls, period)
    gap = _wall_gap(walls, period, center)
    expected = pore.expected_wall_gap_nm()
    if abs(gap - expected) > WALL_GAP_TOLERANCE * expected and "gap" not in warned:
        warned.add("gap")
        warnings.warn(
            f"кадр {frame.index}: промежуток между стенками {gap:.3f} нм, а размер поры {expected:.3f} нм — "
            f"стенки сместились или пора задана неверно",
            PandaWarning,
            stacklevel=3,
        )
    return center


def _wall_gap(walls: FloatArray, period: float, center: float) -> float:
    """Ширина пустого промежутка вокруг center: ближайшие атомы стенок снизу и сверху."""
    offsets = wrap_centered(walls - center, period)
    above, below = offsets[offsets > 0], offsets[offsets <= 0]
    if above.size == 0 or below.size == 0:
        return period  # стенки в одной плоскости: пора — весь ящик
    return float(np.min(above) - np.max(below))


def _bin_edges(
    pore: Pore, centering: ProfileCentering, period: float, bins: int, bin_width_nm: float | None
) -> FloatArray:
    if centering is ProfileCentering.PORE:
        lower, upper = pore.coordinate_bounds_nm()
        if upper - lower > period:
            raise InvalidTrajectoryError(
                f"пора ({upper - lower:.3f} нм) шире ящика вдоль оси профиля ({period:.3f} нм)"
            )
    else:
        lower, upper = -0.5 * period, 0.5 * period
    count = bins if bin_width_nm is None else max(1, round((upper - lower) / bin_width_nm))
    if count < 2:
        raise ValueError(f"бинов {count}: нужно хотя бы 2")
    edges: FloatArray = np.linspace(lower, upper, count + 1)
    return edges


def _check_box_change(frame_index: int, period: float, first_period: float, lost: bool, warned: set[str]) -> None:
    """В системе жидкости бины построены по ящику первого кадра: ящик заметно изменился — предупредить один раз."""
    if "box" in warned:
        return
    if abs(period - first_period) > BOX_CHANGE_TOLERANCE * first_period or lost:
        warned.add("box")
        warnings.warn(
            f"кадр {frame_index}: ящик вдоль оси профиля {period:.3f} нм, а бины построены по первому кадру "
            f"({first_period:.3f} нм) — часть частиц вне бинов или крайние бины недозаполнены",
            PandaWarning,
            stacklevel=3,
        )


def _check_area(pore: Pore, area: float, warned: set[str]) -> None:
    base = pore.base_area_nm2()
    if base is None:
        return
    if abs(area - base) > BOX_AREA_TOLERANCE * base and "area" not in warned:
        warned.add("area")
        warnings.warn(
            f"площадь бина {area:.3f} нм² отличается от основания поры {base:.3f} нм² — проверь размеры поры",
            PandaWarning,
            stacklevel=3,
        )
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `431 passed, 28 skipped`; md-тесты на реальных кадрах по-прежнему зелёные (путь по атомам не изменился).

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests/unit/test_molecules_and_offset.py
git commit -m "Путь данных: профиль по центрам масс молекул (per=\"molecule\") и поправка поверхности стенки (wall_offset_nm)" \
  -m "Атомы молекулы собираются по ближайшему образу; путь данных сверяет промежуток стенок с H + 2 · offset." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 4: Публичный API и документация

**Files:**
- Modify: `src/panda_md/__init__.py`, `tests/test_package.py`, `tests/README.md`, `docs/superpowers/specs/2026-10-01-panda-architecture-design.md`, `ROADMAP.md`, `README.md`

- [ ] **Шаг 1: Тест**

Заменить `tests/test_package.py` целиком:

`tests/test_package.py`:

```python
"""Проверка упаковки: пакет импортируется, версия берётся из метаданных, типы объявлены."""

import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files

import panda_md


def test_version_matches_metadata() -> None:
    assert panda_md.__version__ == version("panda-md")


def test_package_is_typed() -> None:
    assert files("panda_md").joinpath("py.typed").is_file()


def test_importing_package_does_not_import_mdanalysis() -> None:
    """MDAnalysis — необязательная зависимость: нужен только для чтения файлов МД (load_trajectory)."""
    check = "import sys, panda_md; print('MDAnalysis' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"


def test_data_path_is_public() -> None:
    expected = {"ArrayTrajectory", "Axis", "Orientation", "ProfileCentering", "ProfileSet", "density_profile"}
    assert expected | {"load_trajectory"} <= set(panda_md.__all__)


def test_analysis_is_public() -> None:
    expected = {"FitResultSet", "ParticleGrouping", "ShapeComparison", "ShapeFit", "StatisticalInefficiency"}
    assert expected | {"compare_shapes", "fit_blocks", "statistical_inefficiency"} <= set(panda_md.__all__)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Run: `uv run pytest -q tests/test_package.py`
Expected: `1 failed` — `test_analysis_is_public`.

- [ ] **Шаг 3: Реализация и документация**

Заменить `src/panda_md/__init__.py` целиком:

`src/panda_md/__init__.py`:

```python
"""PANDA: Predicting Angle from Nanoscale Density Analysis."""

from importlib.metadata import PackageNotFoundError, version

from panda_md.analysis.blocks import FitResultSet, StatisticalInefficiency, fit_blocks, statistical_inefficiency
from panda_md.analysis.compare import ShapeComparison, ShapeFit, compare_shapes
from panda_md.core.enums import (
    Axis,
    Frame,
    LossFunction,
    Normalization,
    OptimizerMethod,
    Orientation,
    ParameterName,
    ParticleGrouping,
    ProfileCentering,
    Regime,
    Unit,
)
from panda_md.core.errors import PandaError, PandaWarning
from panda_md.core.model import InterfaceModel
from panda_md.core.parameters import Parameter
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile
from panda_md.trajectory.reader import load_trajectory

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = [
    "ArrayTrajectory",
    "Axis",
    "DensityProfile",
    "FitResult",
    "FitResultSet",
    "FitSpec",
    "Frame",
    "InterfaceModel",
    "LossFunction",
    "Normalization",
    "OptimizerMethod",
    "Orientation",
    "PandaError",
    "PandaWarning",
    "Parameter",
    "ParameterName",
    "ParticleGrouping",
    "ProfileCentering",
    "ProfileSet",
    "Regime",
    "ShapeComparison",
    "ShapeFit",
    "SlitPore",
    "SquarePore",
    "StatisticalInefficiency",
    "TrianglePore",
    "Unit",
    "__version__",
    "compare_shapes",
    "density_profile",
    "fit_blocks",
    "load_trajectory",
    "statistical_inefficiency",
]
```

Заменить `tests/README.md` целиком:

````markdown
# Тесты PANDA

Ответ на вопрос «почему мы считаем, что эта часть PANDA работает правильно». Для каждого набора: что проверяется,
на каких данных, какой результат ожидается, какой допуск и к какой категории относится проверка
(математика / оптимизация / данные / физическая модель).

Запуск: `make test` (всё), `uv run pytest -m "not slow"` (без долгой синтетики), `make test-env` (чистое окружение).

## Наборы

| Набор | Файл | Категория | Что проверяется | Данные | Ожидание | Допуск |
|-------|------|-----------|-----------------|--------|----------|--------|
| параметры | `unit/test_parameters.py` | оптимизация | описание параметра (границы, умолчание, лишние поля), градусы ↔ радианы, разбор `optimize` / `fixed` / `bounds` / `initial` (границы только для свободных, конечные; старт внутри границ), упаковка в вектор и обратно | стандартные θ, δ, φ, shift | ошибки с понятным текстом; порядок параметров сохраняется | точное равенство |
| профиль | `unit/test_profile.py` | данные | проверка массивов (длины, NaN, возрастание координаты, знак плотности), неизменяемость, JSON, приведение по ρ_bulk и по средней плотности | ровный профиль 50 точек | безразмерная координата в долях H, плотность / ρ_bulk или / n̄ | 1e-12 |
| реестр | `unit/test_pore_registry.py` | расширяемость | форма, объявленная в тесте, подключается декоратором без правки ядра; повтор — ошибка; у каждого типа поры свой реестр | `FlatLayer`, `OtherPore` из теста | форма видна только своей поре | — |
| модель | `unit/test_model.py` | математика | параметры модели, единицы, ошибки имён, профиль валика в центре поры против формулы, допустимость по интервалу φ, капиллярный функционал (в т. ч. слоя без θ), набор форм каждой поры | щель 8 × 16 нм, квадрат и треугольник 6 × 12 нм | совпадение с формулой | 1e-12 |
| ячейка L × kL | `unit/test_slit_rectangular_cell.py` | математика | все шесть форм щели против формул Shistko et al. (2025) при θ = 180°, δ = 0: размеры, пределы φ, профиль, площадь; ячейка L × kL для осесимметричной формы = квадратная kL × kL; объём поры | k ∈ {0.2, 0.6, 1}, l ∈ {2, 4} | совпадение с формулами статьи (пределы перфорации — выведены интегрированием, в тексте статьи уравнение (23) неоднозначно) | 1e-12 |
| контракт | `contract/test_shapes.py` | математика | для **каждой** зарегистрированной формы каждой поры (щель — и с k = 0.4; квадрат; треугольник): атрибуты контракта (у формы без θ режимов нет); допустимые точки в каждом объявленном режиме; ∫ρ dz = φ · (A / ly); 0 ≤ ρ ≤ 1 и ρ = 0 вне поры; центр масс формулы = численному; S_LV > 0, S_SL ≥ 0. Проверки профиля пропускаются у форм без профиля (10 форм треугольника — только для диаграммы устойчивости) | сетка θ (по 3 угла в режиме), δ ∈ {0, 0.03}, 400 значений φ (droplet_w3c0 треугольника существует в окне ±1.5 % по φ); сетка интегрирования — между стенками поры | все проверки на всех допустимых точках | ∫ — 1e-4 (ступенька на краю), центр масс — 1e-4 |
| регрессия | `regression/test_reference_formulas.py` | математика | перенос формул без изменений: шесть форм щели (k = 1), 12 форм треугольника, 7 форм квадрата; диаграммы устойчивости треугольника и квадрата | профиль по траектории со старой PANDA (центрирование по жидкости, бины по ящику; облако точек через границу ящика, `generate_density_reference.py`, нужно окружение старой PANDA); эталоны из старой PANDA (`alpha.py`, 93c6dbb) и PANDA_Polygon (`triangle.py`, `square.py`, 38ff491); генераторы `regression/generate_*.py`; диаграммы — 10 пар (θ, δ) на пору, сетка 30 × 30 по (l, φ), как в `computePhase` PANDA_Polygon | профиль (где выведен), интервал φ, условие существования, энергия и сумма площадей совпадают; раскраска диаграмм совпадает во всех точках | 1e-12 |
| фит | `unit/test_fitting.py` | оптимизация | один оптимизатор при 1, 2, 3 свободных параметрах; фиксированные не меняются; флаги «у границы» и «у 90°»; пустое окно, профиль не соответствует поре (пора шире данных, профиль в других единицах, профиль в системе жидкости шире поры), фиксированное значение или φ из данных вне границ, нет φ, нет допустимого старта, не объявленный режим, θ в зазоре у 90°, параметр без влияния — ошибки; деление по режимам смачивания | синтетика без шума, 200 бинов | восстановление в пределах дискретизации | θ — 0.5°, δ — 0.0025 (полбина), φ — 0.003 |
| детали фита | `unit/test_fitting_details.py` | оптимизация | барьер выше любой допустимой ошибки и растёт с выходом φ за предел; счётчик вычислений у каждого режима свой; ошибка «нет старта» перечисляет диапазоны; `regime=` у формы без θ — ошибка; φ слоя без систематического сдвига; нормировка mean у формы без φ — ошибка; фит формы без профиля — `NotDerivedError` до оптимизации со списком форм поры, у которых профиль есть | валик, нить квадрата, слой на 35 значениях φ, капли треугольника, форма из теста | точные значения; φ слоя в пределах бина, среднее смещение < 0.0025 | бин 0.005 |
| защита модели | `unit/test_model_guards.py` | математика | формулы вне объявленного режима (θ < 90° у щели, зазор у 90°) не вызываются — ошибка, `is_feasible` = False (раньше капля при θ = 0 падала с ZeroDivisionError); синтетика вне режима — ошибка; неизвестная форма — `UnknownShapeError` (и `KeyError`); энергия формы без θ, касающейся стенок, — ошибка; слой при φ = 1 − 2δ не существует | щель 8 × 16 нм, формы из теста | ожидаемые исключения | — |
| отступления многоугольников | `unit/test_polygon_shapes.py` | физическая модель | формы треугольника с неверной площадью свободной поверхности (droplet_w2c1, capsule_w3c3) не перенесены; граница droplet_w2c1 квадрата строгая (условие источника допускает θ = 3π/4, функция площади — нет) | треугольник и квадрат 1 нм | `UnknownShapeError`; `is_feasible` = False | — |
| устойчивость | `unit/test_stability.py` | математика | в каждой точке (l, φ) — допустимая форма с наименьшим S_LV − cos θ · S_SL; ограничение списка форм; формы без площадей (пончик, перфорация щели) не участвуют; нет допустимых форм — пусто | квадрат, щель, треугольник при a′ = 0 | совпадение с перебором | точное равенство |
| неизменяемость | `unit/test_immutability.py` | данные | словари в `FitResult` и `FitSpec` нельзя изменить; одинаковые `FitSpec` равны и хешируются; JSON туда и обратно | фит валика | `TypeError` при записи; равенство | — |
| PBC и время | `unit/test_pbc_and_time.py` | данные | заворот в [−L/2, L/2); круговое среднее облака через границу ящика (как `get_center_pbc` старой PANDA); наибольший пустой промежуток между атомами стенок через границу; «2 ns» → пс | точки на окружности, строки времени | точные значения; пустые выборки и неизвестные единицы — ошибки | 1e-12 |
| траектория | `unit/test_trajectory.py` | данные | `ArrayTrajectory`: формы массивов, порядок времён, группы и индексы; выбор кадров по времени и шагу; `MDATrajectory` на Universe в памяти: Å → нм, время, выборка, непрямоугольный ящик — ошибка; нет файла или MDAnalysis — понятная ошибка; `SlitPore.from_trajectory`: H, L, k по атомам стенок через границу ящика, неверные оси — ошибка; вакуумный слой — предупреждение, `liquid=` выбирает промежуток с жидкостью, поиск центра в кадре — промежуток ширины H | решётки атомов, Universe из MDAnalysis | точные значения | 1e-12 |
| профиль по траектории | `unit/test_density_profile.py` | данные | центрирование по поре (стенки в каждом кадре или заданный центр) и по жидкости; бины выровнены по стенкам; блоки по времени, неполный блок отброшен; среднее, std, sem; отражение профиля жидкости у верхней стенки; N и V для нормировки mean; предупреждения (частицы вне поры, промежуток стенок ≠ H, основание ящика ≠ kL², ящик вдоль оси меняется в системе жидкости); один граничный кадр — не неполный блок; стенки в одной плоскости — пора на весь ящик; ошибки (нет стенок и центра, пора шире ящика, треугольник без центра); сохранение .npz и .csv | равномерный слой, дрейфующая пора через границу ящика | плотность N / (A · ширина) | 1 % (статистика) |
| путь данных | `synthetic/test_data_path.py` | данные | частицы, разыгранные по профилю модели (z по обратной функции распределения, x, y равномерно), стенки через границу ящика с дрейфом → `density_profile` → фит: валик и капля в системе поры, капля в системе жидкости | щель 4 × 8 нм, k = 0.5, 40 кадров, 4 блока | θ восстановлен | 1.5° |
| МД | `md/test_real_systems.py` | физическая модель | профили по реальным траекториям, посчитанные новым путём данных (`generate_md_profiles.py`, нужны данные и `panda-md[md]`): стенки найдены как в ручном расчёте по .gro; по одному реальному кадру каждой системы (высоты атомов подложки и жидкости, 34–46 КБ) поиск стенок и профиль пересчитываются заново — код, а не только сохранённые результаты; вся жидкость в поре; вода в щели графита — несмачивающий валик; бензол на подложке SOLD смачивает её — фит упирается в границу режима и поднимает флаг (формул θ < 90° нет, O1) | графит (вода, 8 блоков × 1 нс), бензол/SOLD (5 блоков × 1.2 нс), кальцит (стартовый кадр) | вода: θ 165°–180° (задача 1: 171.8° ± 7.1°), φ = 0.487; бензол: у стенок плотнее, чем в середине, `at_regime_boundary` при ρ_bulk 10–25 нм⁻³ | H — 1e-3 нм |
| анализ | `unit/test_analysis.py` | оптимизация | фит по блокам с одними настройками: значения, среднее, std, ошибка среднего, таблица; статистическая неэффективность — формула блочной дисперсии (ряд 1…6: s(2) = 2 · 4 / 3.5), белый шум s ≈ 1, AR(1) с ρ = 0.8 — плато (1 + ρ)/(1 − ρ) = 9; перебор форм: истинная форма первая, ошибки по возрастанию, формы без профиля — с причиной, «лучшая» без допустимых — ошибка | 4 блока синтетики валика (шум 0.5 %), ряды 20 000 значений, капли треугольника | точные значения; плато ± 25 % | 1e-12 |
| молекулы и стенка | `unit/test_molecules_and_offset.py` | данные | центр масс молекулы, разрезанной границей ящика, — у границы, а не в середине ящика; профиль по молекулам считает молекулы (∫ρ · A = N); без номеров молекул и масс — ошибка; MDAnalysis: молекула = остаток; `wall_offset_nm`: H = промежуток − 2 · offset, без ложного предупреждения о промежутке | молекулы из 3 атомов (16 + 1 + 1), сплошная подложка | точные значения | 1e-12 |
| упаковка | `test_package.py` | данные | версия из метаданных, `py.typed`, путь данных в публичном API; `import panda_md` не импортирует MDAnalysis | — | — | — |
| синтетика | `synthetic/test_recovery.py` | физическая модель | полная цепочка «форма → профиль с шумом → фит» для 16 случаев (пять форм щели — все, кроме слоя: валик, капля, пончик, червь, перфорация; капля и валик при k = 0.4; капли droplet_w1c0, droplet_w2c0, droplet_w2c1, нить filament_w1c0 (и при 60°) и капсула квадрата; капля, нить filament_w3c0 и капсула треугольника); три режима ТЗ; нормировка по средней плотности (O2); ошибка положения стенок (O3) | шум 0.5 % ρ_bulk на бин, 5 зерён | см. ниже | максимум 1.5°, среднее 1° |

## Что выяснилось на прототипе (15.10.2026)

**Оптимизатор.** На L1-ошибке градиентный L-BFGS-B с конечными разностями застревает: без шума валик
(θ, δ, φ) — 118.6° вместо 120°, нить при 60° — 68.6°. Nelder–Mead и Powell восстанавливают точно; Nelder–Mead
быстрее (0.5 с против 1.6 с на три свободных параметра). Метод по умолчанию — Nelder–Mead.

**δ и дискретизация.** δ задаётся только положением края профиля, поэтому определяется с точностью до
полуширины бина (при 200 бинах — 0.0025). Когда свободны θ, δ и φ, θ связан с δ через радиус мениска,
и даже без шума разброс θ — до 0.5°. Для статьи: точность δ ограничена шириной бина, а не шумом.

**Шум.** При шуме 1 % ρ_bulk на бин ошибка θ — до 1.7°, при 0.5 % — до 1.1° (10 зерён). Критерий ТЗ
|Δθ| < 1° выполняется в среднем по зёрнам при 0.5 %. Зависимость Δθ(σ) — задача фазы 6 (`validation/`).

**O2 — нормировка по средней плотности.** Для капель и нити φ и θ определяются вместе без ρ_bulk
(|Δφ| < 0.005, ρ_bulk восстанавливается в пределах 10 %). Для валика — нет: при свободной φ разброс θ до 19°,
φ — до 0.28; профиль валика, делённый на φ, почти не меняется вдоль линии в пространстве (θ, δ, φ).
Валику в нормировке mean нужна известная φ (из построения системы) — тогда θ восстанавливается точно.

**O3 — центрирование.** Ошибка положения стенок на 5 % размера поры:
- формы на одной стенке (капли, нить) — сдвиг поглощает δ, θ не меняется (центрирование по поре работает);
- валик касается обеих стенок — без свободного `shift` ошибка θ 5°, со свободным `shift` — 0.2°;
- центрирование по жидкости при известной δ восстанавливает θ для форм на одной стенке; δ при этом не
  определяется (проверка «параметр не влияет»).
Рекомендация: по умолчанию центрирование по поре; для валика при неточных стенках — `optimize=[..., "shift"]`.

## Что выяснилось на стадии 2 — щель полностью (16.10.2026)

**Ячейка L × kL.** Осесимметричная форма, не касающаяся боковых границ ячейки, видит только короткую сторону:
ρ(z; l, k, φ) = k · ρ(z; kl, 1, φ/k) (у перфорации сохраняется объём отверстия). Червь и валик лежат вдоль короткой
стороны, их профиль от k не зависит; у слоя тоже. При θ = 180°, δ = 0 это даёт ровно формулы Shistko et al. (2025)
для всех шести форм; при произвольных θ, δ — контрактные тесты с k = 0.4 (∫ρ = φ, центр масс, пределы).

**Слой.** Профиль — ступенька, φ двигает только её край: малый пробный шаг проверки «параметр влияет» (0.1 % границ)
не выходит за бин, и φ ошибочно объявлялся не влияющим. Проверка теперь пробует шаги 0.1 %, 1 % и 5 % ширины границ.
φ слоя определяется с точностью до одного бина.

**Синтетика новых форм** (шум 0.5 %, 5 зёрен, θ = 130°, свободны θ и δ): пончик, червь, перфорация — |Δθ| до 0.6°;
при k = 0.4 капля 0.6°, пончик, перфорация и валик — до 1.2° (среднее меньше 0.6°).

## Что выяснилось на стадии 3 — треугольная и квадратная поры (16.10.2026)

**Перенос.** 12 форм треугольника и 7 форм квадрата (девять форм квадрата и две треугольника с дефектами источника не перенесены,
`docs/architecture/05_polygon_pores.md` §4) совпадают с PANDA_Polygon до 1e-12; диаграммы устойчивости совпадают
во всех точках. У capsule_w3c0 треугольника длина вставки в источнике записана тремя разными выражениями
(условие, площади, профиль) — они тождественны, оставлено одно.

**Новые дефекты источника** (вопросы автору, `05_polygon_pores.md` §4):
- droplet_w2c1 треугольника: S_LV = 2π R² (1 + cos θ − ⅔ sin θ) расходится с геометрией на всём диапазоне
  90°–120° (Монте-Карло: 3.29 R² против 1.07 R² при 100°), при θ > 112.6° формула отрицательна. Найдено сначала
  по отрицательной площади, затем итоговым ревью; форма не перенесена.
- capsule_w3c3 треугольника: площадь свободных торцов расходится с геометрией уже при δ = 0 (Монте-Карло 0.92 a′²
  против 1.25 a′² при 110°), а при δ > 0 длина вставки записана для a′ = 1. Не перенесена.
- filament_w2c1 треугольника при θ > 120°: геометрия переходит в w2c0, площадь сечения, по-видимому, неверна;
  на диаграммах не побеждает, перенесена как есть — вопрос автору.

**Синтетика новых форм** (шум 0.5 %, 5 зёрен, свободны θ и δ): капля, нить w3c0 и капсула треугольника — |Δθ| до 1.4°
(среднее до 0.6°); капли в углу и капсула квадрата — до 1°. Капсулы восстанавливаются лучше всего (< 0.35°).

**θ у 180° и шум (открытый вопрос O11; уточнено 16.10.2026).** droplet_w2c0 квадрата при θ = 160°: в одном зерне
из пяти фит даёт θ = 168.9° (δ = 0.0075). Это не локальный минимум: все 25 стартов сходятся туда же, перебор δ
по ступенькам бинов ниже ошибку не даёт, а со свободной φ получается 172°. При θ, близком к 180°, профиль капли
слабо зависит от θ, и шум 0.5 % сдвигает уже глобальный минимум. При θ = 150° такого нет. Ранняя запись
«локальный минимум, ошибка на 4 % выше истинной точки» неверна: там сравнивалось с фитом при точно известной φ.
Закрывается оценкой Δθ(шум, θ) в `validation/` (фаза 6).

## Что выяснилось на стадии 4 — путь данных (16.10.2026)

**Совпадение со старой PANDA.** Центрирование по жидкости (круговое среднее) и бины по ящику дают тот же профиль, что `get_center_pbc` + `get_numerical_density_profile`, точно (разница 0) — на облаке, пересекающем границу ящика. Старая версия отдавала координатой левые края бинов, новая — центры.

**Реальная МД.** Щель графита находится по атомам подложки автоматически: H = 8.269 нм, как в ручном расчёте задачи 1 (пора проходит через границу ящика). 4000 кадров (85 тыс. атомов) обрабатываются за 39 с потоково. Вода: θ = 171.1° по среднему профилю, 170.7° ± 7.9° по 8 блокам (задача 1 — 171.8° ± 7.1°).

**Другие подложки.** Бензол на SOLD (5001 кадр, 15 с): стенки найдены (H = 10.314 нм, как в задаче 1), плотность у стенок 8.8 нм⁻³ против 5.2 в середине — бензол смачивает подложку, фит валика при любой разумной ρ_bulk останавливается на границе режима 90.5° с флагом `at_regime_boundary` (в задаче 1 старая версия молча давала 90.57°). Кальцит — только стартовый кадр (равновесной траектории нет): стенки толстой подложки из ионов найдены верно (H = 9.139 нм; ноутбук-пример старой PANDA задавал H = 6 нм вручную).

**Ориентация.** В системе поры профиль отражается, если жидкость у верхней стенки (первый момент > 0). У симметричных форм (валик) знак момента определяется шумом — отражение безвредно, но флаг `flipped` тогда ничего не значит.

## Что выяснилось на стадии 5 — анализ (16.10.2026)

**Вода в щели графита по блокам.** `fit_blocks`: θ = 170.7° ± 3.0° (ошибка среднего по 8 блокам по 1 нс), в одном блоке θ упирается в 180° — флаг «у границы θ». Перебор форм ставит первым валик (ошибка 0.026), червь хуже втрое (0.084), слой — в 14 раз; капля, пончик и перфорация при φ = 0.49 не существуют — пропущены с причиной.

**Профиль по молекулам.** Вода TIP4P по центрам масс молекул: те же 11 807 частиц, профиль отличается от профиля кислородов не больше чем на 0.8 нм⁻³ (у стенок).
````

Спецификация: в таблице решений после строки D23 добавить:

```markdown
| D24 | Анализ — функции слоя `analysis`, а не методы поры: `fit_blocks(model, profiles, spec) -> FitResultSet`, `compare_shapes(pore, profile, spec) -> ShapeComparison` (вместо `pore.fit_all` — ядро не импортирует `analysis`), `statistical_inefficiency(series)`. Настройки фита передаются одним `FitSpec` (стадия 5) |
| D25 | `per="molecule"`: частица — центр масс молекулы (остаток MDAnalysis или `molecule_ids` и `masses` у `ArrayTrajectory`), атомы молекулы собираются по ближайшему образу; `SlitPore.wall_offset_nm`: поверхность стенки отстоит от центров крайних атомов, H = промежуток − 2 · offset (стадия 5) |
```

`ROADMAP.md`, «Фаза 6»: три строки списка (`analysis/`…, `synthetic/points`…, `validation/`…) заменить на:

```markdown
Делится на три стадии:

- **6а, анализ — выполнено** (план `docs/superpowers/plans/2026-10-16-stage5-analysis.md`): фит по блокам (`fit_blocks`, среднее ± ошибка), перебор форм (`compare_shapes`), статистическая неэффективность, профиль по центрам масс молекул (`per="molecule"`), `wall_offset_nm`. Диаграммы устойчивости — стадия 3.
- **6б, трёхмерные формы:** `contains` у форм, облака точек (`synthetic/points`), осевое центрирование (`SymmetryAxis`), 2D-карты плотности.
- **6в, графики и валидация:** `plotting/` (профиль и модель, блоки, диаграммы), `validation/`: Δθ от шума, числа бинов и θ (в том числе O11 — точность у 180°), сравнение центрирований.
```

`README.md`: в примере «From an MD trajectory» после строки `result = pore.interface("roll").fit(...)` добавить:

```python
spec = panda.FitSpec(density_bulk_per_nm3=33.1)
print(panda.fit_blocks(pore.interface("roll"), profiles, spec))  # θ по блокам, среднее ± ошибка
print(panda.compare_shapes(pore, profiles.mean(), spec))  # все формы поры по ошибке фита
```

- [ ] **Шаг 4: Полная проверка и коммит**

Run: `make check`, `make test-env`, `make wheel-check`; Python 3.10: `UV_PROJECT_ENVIRONMENT=.venv-py310 uv run --no-sync pytest -q`. Всё зелёное, `432 passed, 28 skipped`. Пример из README — на реальной траектории графита.

```bash
git add src/panda_md/__init__.py tests docs ROADMAP.md README.md
git commit -m "Стадия 5 (фаза 6а) закрыта: анализ в публичном API, описание тестов, D24–D25, ROADMAP, README" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
