# Стадия 7 — графики и валидация (фаза 6в): план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Оценка точности метода на синтетике (ошибка θ от шума, числа бинов, угла и центрирования) и графики результатов — рисунки и таблицы для документации и статьи; закрытие O11 (точность у 180°).

**Architecture:** `analysis.accuracy` — исследование точности: синтетический профиль с шумом → фит той же формой → Δθ по зёрнам шума (`AccuracyCase`, `AccuracyStudy`, `angle_errors`, `accuracy_study`). `panda_md.plotting` — функции рисования от готовых объектов; matplotlib — необязательная зависимость `[plot]`, импортируется только там. `validation/run_accuracy.py` — скрипт вне пакета: четыре исследования → `validation/figures/*.png` и `validation/REPORT.md`; выводы — `validation/README.md`.

**Tech Stack:** Python ≥ 3.10, numpy, pydantic v2, matplotlib ≥ 3.8 (extra `plot`), pytest, uv, black, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (§2 «matplotlib — только в plotting/», §9 «validation/ строит графики: Δθ от уровня шума и от числа бинов, сравнение центрирований»; O10, O11). Решение D27 — в задаче 4.

**Проверено до написания плана.** Код собран и прогнан по задачам в чистом клоне (коммит `a624dc8`): 482 теста, 28 пропусков; `validation/run_accuracy.py` — около минуты, детерминирован (зёрна фиксированы), повторный запуск даёт тот же `REPORT.md`. Код переносить дословно.

## Global Constraints

- Ограничения стадий 1–6 в силе (pydantic, `Enum`, типы, имена, black 120, ruff, mypy strict, русские сообщения).
- Слои (§2): matplotlib импортируется только в `panda_md/plotting`; `import panda_md` его не тянет (тест); ядро не импортирует `plotting`.
- Разработка локальная, в ветке `stage1-core-prototype`; в GitHub не отправлять.
- Многострочный текст — через Write/Edit; коммиты — отдельными `-m`, последний — `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Проверка перед коммитом: `make check`.

## Review Focus

1. Пересчёт шума по числу бинов: сравнивать разное число бинов можно только при одной статистике траектории — σ · √(бинов / 200). Тест — `test_noise_is_rescaled_with_the_bin_count` (задача 1).
2. Окружение без matplotlib: `import panda_md` работает, `import panda_md.plotting` — понятная ошибка с `pip install 'panda-md[plot]'`. Тесты — `test_plotting_without_matplotlib_is_a_clear_error`, `test_importing_package_does_not_import_matplotlib` (задача 2).
3. Выводы в `validation/README.md` должны совпадать с числами `REPORT.md` (проверить по таблицам) — задача 3.
4. Фиты, упёршиеся в 180°, в исследовании угла: максимальная |Δθ| 5° у валика при 175° — это граница, а не ошибка оптимизатора (флаг `at_bounds`). Описано в выводах — задача 3.
5. Детерминированность валидации: фиксированные зёрна, повторный запуск — тот же отчёт — задача 3.

---

### Задача 1: Исследование точности на синтетике

**Files:**
- Create: `src/panda_md/analysis/accuracy.py`, `tests/unit/test_accuracy_and_plots.py` (без графиков — задача 2)

**Interfaces:**
- Consumes: `synthetic_profile`, `InterfaceModel.fit`.
- Produces: `AccuracyCase(truth, noise_std, bin_count=200, offset=0.0, frame=PORE, optimize=None, fixed={})`; `effective_noise_std(case)`; `angle_errors(model, case, *, density_bulk_per_nm3, seeds) -> FloatArray`; `AccuracyStudy(shape_name, varied, labels, values, errors_deg)` с `mean_absolute`, `maximum_absolute`, `bias`, `table()`; `accuracy_study(model, varied, cases, *, density_bulk_per_nm3, seeds)`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_accuracy_and_plots.py`:

```python
"""Исследование точности на синтетике."""

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md import SlitPore
from panda_md.analysis.accuracy import (
    AccuracyCase,
    AccuracyStudy,
    accuracy_study,
    angle_errors,
    effective_noise_std,
)

DENSITY_BULK = 30.0
PORE = SlitPore(width_nm=8.0, length_nm=16.0)
ROLL_TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}


def test_without_noise_the_angle_is_recovered() -> None:
    errors = angle_errors(
        PORE.interface("roll"),
        AccuracyCase(truth=ROLL_TRUTH, noise_std=0.0),
        density_bulk_per_nm3=DENSITY_BULK,
        seeds=[0],
    )
    assert abs(errors[0]) < 0.5


def test_study_summarises_errors_per_condition() -> None:
    cases = {
        "0.5": (0.5, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.005)),
        "2": (2.0, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.02)),
    }
    study = accuracy_study(PORE.interface("roll"), "шум, %", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=range(3))
    assert study.errors_deg.shape == (2, 3)
    assert study.mean_absolute()[1] > study.mean_absolute()[0]  # больше шума — больше ошибка
    np.testing.assert_allclose(study.maximum_absolute(), np.max(np.abs(study.errors_deg), axis=1))
    table = study.table()
    assert table.splitlines()[0].startswith("| шум, %") and "| 2 |" in table
    with pytest.raises(ValidationError, match="подписи"):
        AccuracyStudy(
            shape_name="roll", varied="x", labels=("a",), values=np.array([1.0, 2.0]), errors_deg=np.zeros((2, 1))
        )


def test_noise_is_rescaled_with_the_bin_count() -> None:
    """Та же траектория, разбитая мельче, — шум на бин растёт как √(бинов / 200)."""
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01)) == pytest.approx(0.01)
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01, bin_count=400)) == pytest.approx(
        0.01 * np.sqrt(2.0)
    )
    with pytest.raises(ValidationError):
        AccuracyCase(truth=ROLL_TRUTH, noise_std=-0.1)
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.analysis.accuracy'`.

- [ ] **Шаг 3: Реализация**

`src/panda_md/analysis/accuracy.py`:

```python
"""Точность метода на синтетике: ошибка восстановленного θ при заданных шуме, числе бинов, угле и центрировании.

Профиль строится по модели (`synthetic_profile`) с гауссовым шумом в долях ρ_bulk на бин, затем фитируется той же
формой. Ошибка зависит только от метода (фит, дискретизация, шум), а не от адекватности формы — это нижняя граница
ошибки на реальных данных. Шум задаётся для 200 бинов; при другом числе бинов он пересчитывается как √(бинов / 200):
та же траектория, разбитая мельче, даёт меньше частиц на бин.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.model import InterfaceModel
from panda_md.core.types import Array1D, Array2D, FloatArray, PandaModel
from panda_md.synthetic.profiles import synthetic_profile

REFERENCE_BIN_COUNT = 200  # шум задаётся для этого числа бинов


class AccuracyCase(PandaModel):
    """Одна точка исследования: истинные параметры и условия синтетики."""

    truth: dict[str, float]  # единицы пользователя
    noise_std: float = Field(ge=0.0)  # доля ρ_bulk на бин при REFERENCE_BIN_COUNT бинах
    bin_count: int = Field(default=REFERENCE_BIN_COUNT, ge=10)
    offset: float = 0.0  # сдвиг интерфейса вдоль оси профиля в долях размера поры (ошибка положения стенок)
    frame: Frame = Frame.PORE
    optimize: tuple[str, ...] | None = None
    fixed: dict[str, float] = Field(default_factory=dict)


class AccuracyStudy(PandaModel):
    """Ошибки θ (градусы, восстановленное − истинное) по зёрнам шума для ряда значений одного условия."""

    shape_name: str
    varied: str  # что меняется: «шум, % ρ_bulk», «бинов», «θ, °», «центрирование»
    labels: tuple[str, ...]  # подпись каждого значения
    values: Array1D  # численное значение условия (для оси графика)
    errors_deg: Array2D  # значения × зёрна

    @model_validator(mode="after")
    def _check_shapes(self) -> "AccuracyStudy":
        if self.errors_deg.shape[0] != self.values.size or len(self.labels) != self.values.size:
            raise ValueError("по строке ошибок и подписи на каждое значение условия")
        return self

    def mean_absolute(self) -> FloatArray:
        mean: FloatArray = np.mean(np.abs(self.errors_deg), axis=1)
        return mean

    def maximum_absolute(self) -> FloatArray:
        maximum: FloatArray = np.max(np.abs(self.errors_deg), axis=1)
        return maximum

    def bias(self) -> FloatArray:
        """Систематический сдвиг: среднее (восстановленное − истинное)."""
        bias: FloatArray = np.mean(self.errors_deg, axis=1)
        return bias

    def table(self) -> str:
        """Таблица markdown: условие, средняя и максимальная |Δθ|, смещение."""
        lines = [f"| {self.varied} | среднее \\|Δθ\\|, ° | максимум \\|Δθ\\|, ° | смещение, ° |", "|---|---|---|---|"]
        for label, mean, maximum, bias in zip(
            self.labels, self.mean_absolute(), self.maximum_absolute(), self.bias(), strict=True
        ):
            lines.append(f"| {label} | {mean:.2f} | {maximum:.2f} | {bias:+.2f} |")
        return "\n".join(lines)


def effective_noise_std(case: AccuracyCase) -> float:
    """Шум на бин при case.bin_count бинах: та же статистика, разбитая мельче, — σ · √(бинов / 200)."""
    return case.noise_std * math.sqrt(case.bin_count / REFERENCE_BIN_COUNT)


def angle_errors(
    model: InterfaceModel, case: AccuracyCase, *, density_bulk_per_nm3: float, seeds: Sequence[int]
) -> FloatArray:
    """θ_фит − θ_истинный (градусы) для каждого зерна шума."""
    noise = effective_noise_std(case)
    errors = []
    for seed in seeds:
        profile = synthetic_profile(
            model,
            case.truth,
            density_bulk_per_nm3=density_bulk_per_nm3,
            bin_count=case.bin_count,
            noise_std=noise,
            frame=case.frame,
            offset=case.offset,
            seed=seed,
        )
        result = model.fit(profile, density_bulk_per_nm3=density_bulk_per_nm3, optimize=case.optimize, fixed=case.fixed)
        errors.append(result.values[ParameterName.CONTACT_ANGLE.value] - case.truth[ParameterName.CONTACT_ANGLE.value])
    angles: FloatArray = np.array(errors)
    return angles


def accuracy_study(
    model: InterfaceModel,
    varied: str,
    cases: Mapping[str, tuple[float, AccuracyCase]],
    *,
    density_bulk_per_nm3: float,
    seeds: Sequence[int],
) -> AccuracyStudy:
    """Исследование: для каждой подписи — (значение условия, условия синтетики)."""
    errors = [
        angle_errors(model, case, density_bulk_per_nm3=density_bulk_per_nm3, seeds=seeds) for _, case in cases.values()
    ]
    return AccuracyStudy(
        shape_name=model.shape.name,
        varied=varied,
        labels=tuple(cases),
        values=np.array([value for value, _ in cases.values()]),
        errors_deg=np.array(errors),
    )
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `479 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/analysis/accuracy.py tests/unit/test_accuracy_and_plots.py
git commit -m "Точность метода на синтетике: ошибка θ по зёрнам шума при заданных условиях" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 2: Графики (extra `plot`)

**Files:**
- Modify: `pyproject.toml` (extra `plot` = matplotlib ≥ 3.8), `Makefile` и `.github/workflows/ci.yml` (тесты с `--extra plot`)
- Create: `src/panda_md/plotting/__init__.py`, `src/panda_md/plotting/figures.py`
- Modify: `tests/unit/test_accuracy_and_plots.py`

**Interfaces:**
- Consumes: `FitResult`, `FitResultSet`, `DensityMap`, `StabilityDiagram`, `AccuracyStudy`.
- Produces: `plot_fit(result, ax=None)`, `plot_blocks(blocks, parameter="theta", ax=None)`, `plot_density_map(density, ax=None)`, `plot_stability(diagram, ax=None)`, `plot_accuracy(study, ax=None)` — каждая возвращает `matplotlib.axes.Axes`.

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_accuracy_and_plots.py` целиком:

`tests/unit/test_accuracy_and_plots.py`:

```python
"""Исследование точности на синтетике и графики (matplotlib — необязательная зависимость)."""

import importlib
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md import FitSpec, SlitPore
from panda_md.analysis.accuracy import (
    AccuracyCase,
    AccuracyStudy,
    accuracy_study,
    angle_errors,
    effective_noise_std,
)
from panda_md.analysis.blocks import fit_blocks
from panda_md.analysis.stability import stability_diagram
from panda_md.core.errors import MissingDependencyError
from panda_md.synthetic.points import synthetic_trajectory
from panda_md.synthetic.profiles import synthetic_profile
from panda_md.trajectory.maps import density_map

DENSITY_BULK = 30.0
PORE = SlitPore(width_nm=8.0, length_nm=16.0)
ROLL_TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}


def test_without_noise_the_angle_is_recovered() -> None:
    errors = angle_errors(
        PORE.interface("roll"),
        AccuracyCase(truth=ROLL_TRUTH, noise_std=0.0),
        density_bulk_per_nm3=DENSITY_BULK,
        seeds=[0],
    )
    assert abs(errors[0]) < 0.5


def test_study_summarises_errors_per_condition() -> None:
    cases = {
        "0.5": (0.5, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.005)),
        "2": (2.0, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.02)),
    }
    study = accuracy_study(PORE.interface("roll"), "шум, %", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=range(3))
    assert study.errors_deg.shape == (2, 3)
    assert study.mean_absolute()[1] > study.mean_absolute()[0]  # больше шума — больше ошибка
    np.testing.assert_allclose(study.maximum_absolute(), np.max(np.abs(study.errors_deg), axis=1))
    table = study.table()
    assert table.splitlines()[0].startswith("| шум, %") and "| 2 |" in table
    with pytest.raises(ValidationError, match="подписи"):
        AccuracyStudy(
            shape_name="roll", varied="x", labels=("a",), values=np.array([1.0, 2.0]), errors_deg=np.zeros((2, 1))
        )


def test_noise_is_rescaled_with_the_bin_count() -> None:
    """Та же траектория, разбитая мельче, — шум на бин растёт как √(бинов / 200)."""
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01)) == pytest.approx(0.01)
    assert effective_noise_std(AccuracyCase(truth=ROLL_TRUTH, noise_std=0.01, bin_count=400)) == pytest.approx(
        0.01 * np.sqrt(2.0)
    )
    with pytest.raises(ValidationError):
        AccuracyCase(truth=ROLL_TRUTH, noise_std=-0.1)


@pytest.fixture
def pyplot() -> object:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_figures_are_drawn_from_panda_objects(pyplot: object) -> None:
    from panda_md.plotting import plot_accuracy, plot_blocks, plot_density_map, plot_fit, plot_stability

    roll = PORE.interface("roll")
    profile = synthetic_profile(roll, ROLL_TRUTH, density_bulk_per_nm3=DENSITY_BULK, noise_std=0.005, seed=0)
    result = roll.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert "θ = 1" in plot_fit(result).get_legend_handles_labels()[1][1]

    trajectory = synthetic_trajectory(
        SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5).interface("roll"),
        {"theta": 120.0, "delta": 0.03, "phi": 0.3},
        density_bulk_per_nm3=33.0,
        frame_count=4,
        seed=0,
    )
    walls = SlitPore.from_trajectory(trajectory, walls="walls")
    assert (
        "ось формы"
        in plot_density_map(density_map(trajectory, walls, "liquid", shape="roll", bins=(20, 20))).get_xlabel()
    )

    from panda_md import density_profile

    profiles = density_profile(trajectory, walls, "liquid", bins=40, block="10 ps")
    blocks = fit_blocks(walls.interface("roll"), profiles, FitSpec(density_bulk_per_nm3=33.0))
    assert plot_blocks(blocks).get_ylabel() == "theta"

    diagram = stability_diagram(
        PORE, contact_angle_deg=150.0, wall_layer=0.0, volume_fractions=(0.1, 0.4, 0.7), length_ratios=(1.0, 2.0, 4.0)
    )
    assert plot_stability(diagram).get_title().startswith("slit")

    cases = {"0.5": (0.5, AccuracyCase(truth=ROLL_TRUTH, noise_std=0.005))}
    study = accuracy_study(roll, "шум, %", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=range(2))
    assert plot_accuracy(study).get_xlabel() == "шум, %"


def test_plotting_without_matplotlib_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [module for module in sys.modules if module.startswith(("matplotlib", "panda_md.plotting"))]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(MissingDependencyError, match=r"panda-md\[plot\]"):
        importlib.import_module("panda_md.plotting")


def test_importing_package_does_not_import_matplotlib() -> None:
    import subprocess

    check = "import sys, panda_md; print('matplotlib' in sys.modules)"
    output = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "False"
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: `2 failed` — `ModuleNotFoundError: No module named 'panda_md.plotting'`.

- [ ] **Шаг 3: Реализация**

Заменить `pyproject.toml` целиком:

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.26"]
build-backend = "hatchling.build"

[project]
name = "panda-md"
version = "0.0.1"
description = "PANDA (Predicting Angle from Nanoscale Density Analysis): contact angle and interface shape of confined liquids from 1D density profiles of molecular simulations"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Ilya Garanov" }]
keywords = ["molecular dynamics", "contact angle", "wetting", "nanopores", "density profile"]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Physics",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pydantic>=2.6",
]

[project.urls]
Repository = "https://github.com/Ilya-Garanov-WorkingAcc/PANDA_2.0"

[project.optional-dependencies]
md = [
    "mdanalysis>=2.7",
]
plot = [
    "matplotlib>=3.8",
]

# Группы зависимостей (PEP 735) — в опубликованный пакет не попадают.
#   uv sync: группа dev (всё для разработки)
#   uv sync --no-default-groups --group test: только пакет и то, что нужно для запуска тестов
[dependency-groups]
test = [
    "pytest>=8",
    "pytest-cov>=5",
]
lint = [
    "black>=24",
    "ruff>=0.6",
    "mypy>=1.11",
    "scipy-stubs>=1.14; python_version >= '3.11'",
]
dev = [
    { include-group = "test" },
    { include-group = "lint" },
]

[tool.hatch.build.targets.wheel]
packages = ["src/panda_md"]

[tool.hatch.build.targets.sdist]
include = ["src", "tests", "README.md", "LICENSE", "pyproject.toml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config --import-mode=importlib"
markers = ["slow: долгие тесты (синтетическая валидация, МД)"]

[tool.black]
target-version = ["py310", "py311", "py312", "py313"]
line-length = 120
extend-exclude = "/(PANDA|smoke|docs)/"

[tool.ruff]
line-length = 120
target-version = "py310"
src = ["src", "tests"]
extend-exclude = ["PANDA", "smoke", "docs", "*.md"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "ANN", "RUF", "NPY"]
ignore = ["RUF001", "RUF002", "RUF003"]  # кириллица в строках и комментариях

[tool.mypy]
python_version = "3.10"
strict = true
files = ["src", "tests"]
warn_unreachable = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["MDAnalysis", "MDAnalysis.*"]  # у MDAnalysis нет аннотаций типов
ignore_missing_imports = true
```

Заменить `Makefile` целиком:

`Makefile`:

```
.DEFAULT_GOAL := help
.PHONY: help run push lint deps install test test-env lint-check typecheck check build wheel-check clean

help:  ## список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

run:
	uv run -m src.main
push:
	git add .
	@if [ -z "$(msg)" ]; then \
		read -p "Enter commit message: " user_msg; \
		git commit -m "$$user_msg"; \
	else \
		git commit -m "$(msg)"; \
	fi
	git push
lint:  ## black + ruff с автоисправлением
	uv run black . -l 120
	uv run ruff check --fix
deps:
	sed -i 's/>/~/g' pyproject.toml

install:  ## окружение разработки (.venv): пакет с [md] + группы test и lint
	uv sync --all-extras

test:  ## тесты
	uv run pytest

test-env:  ## тесты в отдельном чистом окружении: только пакет с [md, plot] и группа test
	UV_PROJECT_ENVIRONMENT=.venv-test uv sync --no-default-groups --group test --extra md --extra plot
	UV_PROJECT_ENVIRONMENT=.venv-test uv run --no-sync pytest

lint-check:  ## black + ruff без изменения файлов (как в CI)
	uv run black . -l 120 --check
	uv run ruff check

typecheck:  ## mypy (strict)
	uv run mypy

check: lint-check typecheck test  ## всё, что проверяет CI

build: clean  ## собрать sdist и wheel в dist/
	uv build

wheel-check: build  ## установить собранный wheel в чистое окружение и импортировать
	uv run --isolated --no-project --with dist/*.whl python -c "import panda_md; print('panda_md', panda_md.__version__)"

clean:  ## удалить артефакты сборки и кэши
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
```

Заменить `.github/workflows/ci.yml` целиком:

`.github/workflows/ci.yml`:

```
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --all-extras
      - run: uv run black . -l 120 --check
      - run: uv run ruff check
      - run: uv run mypy

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: ${{ matrix.python-version }}
      # только пакет с [md, plot] и группа test — тесты не должны зависеть от инструментов разработки
      - run: uv sync --no-default-groups --group test --extra md --extra plot
      - run: uv run --no-sync pytest

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv build
      - run: uv run --isolated --no-project --with dist/*.whl python -c "import panda_md; print(panda_md.__version__)"
```

`src/panda_md/plotting/__init__.py`:

```python
"""Графики результатов (необязательная зависимость: `pip install 'panda-md[plot]'`).

Единственный пакет, импортирующий matplotlib; функции принимают готовые объекты PANDA и рисуют на переданной оси
(или создают новую) — ничего не считают сами.
"""

from panda_md.plotting.figures import (
    plot_accuracy,
    plot_blocks,
    plot_density_map,
    plot_fit,
    plot_stability,
)

__all__ = ["plot_accuracy", "plot_blocks", "plot_density_map", "plot_fit", "plot_stability"]
```

`src/panda_md/plotting/figures.py`:

```python
"""Графики: профиль и модель, фит по блокам, 2D-карта плотности, диаграмма устойчивости, точность метода."""

import numpy as np

from panda_md.analysis.accuracy import AccuracyStudy
from panda_md.analysis.blocks import FitResultSet
from panda_md.analysis.stability import StabilityDiagram
from panda_md.core.errors import MissingDependencyError
from panda_md.core.result import FitResult
from panda_md.trajectory.maps import DensityMap

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.colors import ListedColormap
except ImportError as error:  # pragma: no cover — проверяется тестом без matplotlib
    raise MissingDependencyError("для графиков нужен matplotlib: pip install 'panda-md[plot]'") from error

EMPTY_CELL_COLOR = "#f0f0f0"  # точки диаграммы, где ни одна форма не существует


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created = plt.subplots()
    return created


def plot_fit(result: FitResult, ax: Axes | None = None) -> Axes:
    """Данные и модель в безразмерных координатах окна фита."""
    axes = _axes(ax)
    axes.plot(result.coordinate, result.data, ".", markersize=3, label="данные")
    axes.plot(
        result.coordinate,
        result.model_curve,
        "-",
        label=(
            f"{result.shape_name}: θ = {result.values['theta']:.1f}°" if "theta" in result.values else result.shape_name
        ),
    )
    axes.set_xlabel("координата / размер поры")
    axes.set_ylabel("плотность (нормированная)")
    axes.legend()
    return axes


def plot_blocks(blocks: FitResultSet, parameter: str = "theta", ax: Axes | None = None) -> Axes:
    """Значение параметра в каждом блоке, среднее и ошибка среднего."""
    axes = _axes(ax)
    times_ns = 0.5 * (blocks.block_start_ps + blocks.block_stop_ps) / 1000.0
    values = blocks.values(parameter)
    axes.plot(times_ns, values, "o-", label="блоки")
    mean = blocks.mean(parameter)
    axes.axhline(mean, color="black", linewidth=1, label=f"среднее {mean:.4g} ± {blocks.sem(parameter):.2g}")
    axes.set_xlabel("время, нс")
    axes.set_ylabel(parameter)
    axes.legend()
    return axes


def plot_density_map(density: DensityMap, ax: Axes | None = None) -> Axes:
    """ρ(x, z) в нм⁻³: x — вдоль длинной стороны (ось формы в 0, если центрировано), z — ось профиля."""
    axes = _axes(ax)
    mesh = axes.pcolormesh(density.length_edges_nm, density.height_edges_nm, density.density_per_nm3.T, shading="flat")
    axes.figure.colorbar(mesh, ax=axes, label="ρ, нм⁻³")
    axes.set_xlabel("x, нм" + (" (ось формы в 0)" if density.axis_centered else ""))
    axes.set_ylabel(f"z, нм (система {density.frame.value})")
    axes.set_aspect("equal")
    return axes


def plot_stability(diagram: StabilityDiagram, ax: Axes | None = None) -> Axes:
    """Устойчивая форма в каждой точке (φ, l); пустые клетки — ни одна форма не существует."""
    axes = _axes(ax)
    names = list(diagram.shape_names)
    codes = np.array(
        [[names.index(name) if name is not None else -1 for name in row] for row in diagram.stable_shapes], dtype=float
    )
    palette = plt.get_cmap("tab20", max(len(names), 1))
    colors = [EMPTY_CELL_COLOR, *[palette(index) for index in range(len(names))]]
    axes.pcolormesh(
        np.asarray(diagram.volume_fractions),
        np.asarray(diagram.length_ratios),
        codes + 1,
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=len(names) + 0.5,
        shading="nearest",
    )
    present = sorted({name for row in diagram.stable_shapes for name in row if name is not None}, key=names.index)
    for name in present:
        axes.plot([], [], "s", color=palette(names.index(name)), label=name)
    axes.set_xlabel("φ")
    axes.set_ylabel("l = L / размер поры")
    axes.set_title(f"{diagram.pore_name}: θ = {diagram.contact_angle_deg:g}°, δ = {diagram.wall_layer:g}")
    if present:
        axes.legend(fontsize="small")
    return axes


def plot_accuracy(study: AccuracyStudy, ax: Axes | None = None) -> Axes:
    """Средняя и максимальная |Δθ| и смещение по значениям условия (значения — равномерно, с подписями)."""
    axes = _axes(ax)
    positions = np.arange(len(study.labels))
    axes.plot(positions, study.mean_absolute(), "o-", label="среднее |Δθ|")
    axes.plot(positions, study.maximum_absolute(), "s--", label="максимум |Δθ|")
    axes.plot(positions, study.bias(), "^:", label="смещение")
    axes.axhline(0.0, color="grey", linewidth=0.5)
    axes.set_xticks(positions, study.labels)
    axes.set_xlabel(study.varied)
    axes.set_ylabel("Δθ, °")
    axes.set_title(study.shape_name)
    axes.legend()
    return axes
```

Run: `uv sync --all-extras`.

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `482 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add pyproject.toml Makefile .github/workflows/ci.yml src/panda_md/plotting tests/unit/test_accuracy_and_plots.py
git commit -m "Графики: профиль и модель, фит по блокам, 2D-карта, диаграмма устойчивости, точность (panda-md[plot])" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 3: Валидация — скрипт, рисунки, отчёт и выводы

Проверочная задача: кода пакета нет, тестов нет; критерий — скрипт воспроизводит отчёт.

**Files:**
- Modify: `pyproject.toml` (mypy проверяет и `validation/`)
- Create: `validation/run_accuracy.py`, `validation/README.md`, `validation/REPORT.md`, `validation/figures/*.png`

- [ ] **Шаг 1: Скрипт и mypy**

Заменить `pyproject.toml` целиком:

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.26"]
build-backend = "hatchling.build"

[project]
name = "panda-md"
version = "0.0.1"
description = "PANDA (Predicting Angle from Nanoscale Density Analysis): contact angle and interface shape of confined liquids from 1D density profiles of molecular simulations"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Ilya Garanov" }]
keywords = ["molecular dynamics", "contact angle", "wetting", "nanopores", "density profile"]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Physics",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",
    "pydantic>=2.6",
]

[project.urls]
Repository = "https://github.com/Ilya-Garanov-WorkingAcc/PANDA_2.0"

[project.optional-dependencies]
md = [
    "mdanalysis>=2.7",
]
plot = [
    "matplotlib>=3.8",
]

# Группы зависимостей (PEP 735) — в опубликованный пакет не попадают.
#   uv sync: группа dev (всё для разработки)
#   uv sync --no-default-groups --group test: только пакет и то, что нужно для запуска тестов
[dependency-groups]
test = [
    "pytest>=8",
    "pytest-cov>=5",
]
lint = [
    "black>=24",
    "ruff>=0.6",
    "mypy>=1.11",
    "scipy-stubs>=1.14; python_version >= '3.11'",
]
dev = [
    { include-group = "test" },
    { include-group = "lint" },
]

[tool.hatch.build.targets.wheel]
packages = ["src/panda_md"]

[tool.hatch.build.targets.sdist]
include = ["src", "tests", "README.md", "LICENSE", "pyproject.toml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config --import-mode=importlib"
markers = ["slow: долгие тесты (синтетическая валидация, МД)"]

[tool.black]
target-version = ["py310", "py311", "py312", "py313"]
line-length = 120
extend-exclude = "/(PANDA|smoke|docs)/"

[tool.ruff]
line-length = 120
target-version = "py310"
src = ["src", "tests"]
extend-exclude = ["PANDA", "smoke", "docs", "*.md"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "ANN", "RUF", "NPY"]
ignore = ["RUF001", "RUF002", "RUF003"]  # кириллица в строках и комментариях

[tool.mypy]
python_version = "3.10"
strict = true
files = ["src", "tests", "validation"]
warn_unreachable = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["MDAnalysis", "MDAnalysis.*"]  # у MDAnalysis нет аннотаций типов
ignore_missing_imports = true
```

`validation/run_accuracy.py`:

```python
"""Точность метода на синтетике — рисунки и таблицы для документации и статьи.

Запуск из корня репозитория (нужен panda-md[plot]):
    uv run python validation/run_accuracy.py
Результат: validation/figures/*.png и validation/REPORT.md (таблицы и выводы пересобираются при каждом запуске).
Время — несколько минут: каждое значение условия — 10 зёрен шума, каждый фит — мультистарт.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без окна: только файлы

import matplotlib.pyplot as plt

from panda_md import Frame, SlitPore
from panda_md.analysis.accuracy import AccuracyCase, AccuracyStudy, accuracy_study
from panda_md.plotting import plot_accuracy

HERE = Path(__file__).parent
FIGURES = HERE / "figures"
REPORT = HERE / "REPORT.md"
DENSITY_BULK = 30.0
SEEDS = range(10)
PORE = SlitPore(width_nm=8.0, length_nm=16.0)
TRUTHS = {
    "roll": {"theta": 120.0, "delta": 0.03, "phi": 0.4},
    "droplet": {"theta": 130.0, "delta": 0.03, "phi": 0.05},
    "worm": {"theta": 130.0, "delta": 0.03, "phi": 0.1},
}
NOISE_LEVELS = (0.0025, 0.005, 0.01, 0.02)
BIN_COUNTS = (50, 100, 200, 400)
ANGLES = (100.0, 120.0, 140.0, 160.0, 170.0, 175.0)
BASE_NOISE = 0.005
WALL_ERROR = 0.05  # ошибка положения стенок — 5 % ширины щели (O3)


def noise_study(shape: str) -> AccuracyStudy:
    cases = {
        f"{100 * noise:g}": (100 * noise, AccuracyCase(truth=TRUTHS[shape], noise_std=noise)) for noise in NOISE_LEVELS
    }
    return accuracy_study(
        PORE.interface(shape), "шум, % ρ_bulk на бин", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS
    )


def bins_study(shape: str) -> AccuracyStudy:
    cases = {
        f"{bins}": (float(bins), AccuracyCase(truth=TRUTHS[shape], noise_std=BASE_NOISE, bin_count=bins))
        for bins in BIN_COUNTS
    }
    return accuracy_study(PORE.interface(shape), "бинов", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS)


def angle_study(shape: str) -> AccuracyStudy:
    cases = {}
    for angle in ANGLES:
        truth = {**TRUTHS[shape], "theta": angle}
        if not PORE.interface(shape).is_feasible(**truth):
            continue
        cases[f"{angle:g}"] = (angle, AccuracyCase(truth=truth, noise_std=BASE_NOISE))
    return accuracy_study(PORE.interface(shape), "θ, °", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS)


def centering_study(shape: str) -> AccuracyStudy:
    truth = TRUTHS[shape]
    cases = {
        "поре": (0.0, AccuracyCase(truth=truth, noise_std=BASE_NOISE, offset=WALL_ERROR)),
        "поре + shift": (
            1.0,
            AccuracyCase(truth=truth, noise_std=BASE_NOISE, offset=WALL_ERROR, optimize=("theta", "delta", "shift")),
        ),
        "жидкости, δ известна": (
            2.0,
            AccuracyCase(
                truth=truth,
                noise_std=BASE_NOISE,
                offset=WALL_ERROR,
                frame=Frame.LIQUID,
                optimize=("theta",),
                fixed={"delta": truth["delta"]},
            ),
        ),
    }
    return accuracy_study(
        PORE.interface(shape), "центрирование по", cases, density_bulk_per_nm3=DENSITY_BULK, seeds=SEEDS
    )


def save_figure(study: AccuracyStudy, name: str) -> str:
    figure, axes = plt.subplots(figsize=(5.0, 3.6))
    plot_accuracy(study, axes)
    figure.tight_layout()
    path = FIGURES / f"{name}.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path.name


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    started = time.perf_counter()
    sections = [
        "# Точность метода на синтетике\n",
        f"Сгенерировано `validation/run_accuracy.py`. Щель {PORE.width_nm:g} × {PORE.length_nm:g} нм, ρ_bulk = "
        f"{DENSITY_BULK:g} нм⁻³, {len(SEEDS)} зёрен шума на точку, фит — настройки по умолчанию (θ и δ свободны, "
        f"φ из ∫ρ). Шум задан для 200 бинов и пересчитывается как √(бинов / 200). Δθ = восстановленный − истинный.\n",
    ]
    studies = [
        ("Шум", "noise", noise_study, ("roll", "droplet", "worm")),
        ("Число бинов (шум 0.5 % при 200 бинах)", "bins", bins_study, ("roll", "droplet")),
        ("Угол (шум 0.5 %): точность у 180° — O11", "angle", angle_study, ("roll", "droplet", "worm")),
        ("Центрирование при ошибке положения стенок 5 % H — O3", "centering", centering_study, ("roll", "droplet")),
    ]
    for title, key, build, shapes in studies:
        sections.append(f"\n## {title}\n")
        for shape in shapes:
            study = build(shape)
            figure = save_figure(study, f"{key}_{shape}")
            sections.append(
                f"\n### {shape} (истинные {TRUTHS[shape]})\n\n{study.table()}\n\n![{shape}](figures/{figure})\n"
            )
            print(f"{key}/{shape}: {time.perf_counter() - started:.0f} с", flush=True)
    REPORT.write_text("".join(sections))
    print(f"записано: {REPORT}")


if __name__ == "__main__":
    main()
```

- [ ] **Шаг 2: Отчёт**

Run: `uv run python validation/run_accuracy.py`
Expected: около минуты, последняя строка `записано: .../validation/REPORT.md`; 10 рисунков в `validation/figures/`. Отчёт:

````markdown
# Точность метода на синтетике
Сгенерировано `validation/run_accuracy.py`. Щель 8 × 16 нм, ρ_bulk = 30 нм⁻³, 10 зёрен шума на точку, фит — настройки по умолчанию (θ и δ свободны, φ из ∫ρ). Шум задан для 200 бинов и пересчитывается как √(бинов / 200). Δθ = восстановленный − истинный.

## Шум

### roll (истинные {'theta': 120.0, 'delta': 0.03, 'phi': 0.4})

| шум, % ρ_bulk на бин | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 0.25 | 0.17 | 0.40 | -0.03 |
| 0.5 | 0.35 | 0.81 | -0.07 |
| 1 | 0.69 | 1.63 | -0.13 |
| 2 | 1.39 | 3.29 | -0.27 |

![roll](figures/noise_roll.png)

### droplet (истинные {'theta': 130.0, 'delta': 0.03, 'phi': 0.05})

| шум, % ρ_bulk на бин | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 0.25 | 0.46 | 0.63 | +0.46 |
| 0.5 | 0.45 | 0.86 | +0.42 |
| 1 | 0.68 | 1.68 | +0.29 |
| 2 | 0.95 | 1.69 | +0.13 |

![droplet](figures/noise_droplet.png)

### worm (истинные {'theta': 130.0, 'delta': 0.03, 'phi': 0.1})

| шум, % ρ_bulk на бин | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 0.25 | 0.64 | 0.83 | +0.64 |
| 0.5 | 0.30 | 0.81 | +0.29 |
| 1 | 0.78 | 1.58 | -0.70 |
| 2 | 1.93 | 3.93 | -1.78 |

![worm](figures/noise_worm.png)

## Число бинов (шум 0.5 % при 200 бинах)

### roll (истинные {'theta': 120.0, 'delta': 0.03, 'phi': 0.4})

| бинов | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 50 | 0.81 | 1.24 | -0.81 |
| 100 | 0.30 | 0.69 | +0.05 |
| 200 | 0.35 | 0.81 | -0.07 |
| 400 | 0.31 | 0.81 | +0.09 |

![roll](figures/bins_roll.png)

### droplet (истинные {'theta': 130.0, 'delta': 0.03, 'phi': 0.05})

| бинов | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 50 | 1.27 | 2.25 | -1.21 |
| 100 | 0.92 | 1.55 | +0.92 |
| 200 | 0.45 | 0.86 | +0.42 |
| 400 | 0.36 | 0.73 | +0.05 |

![droplet](figures/bins_droplet.png)

## Угол (шум 0.5 %): точность у 180° — O11

### roll (истинные {'theta': 120.0, 'delta': 0.03, 'phi': 0.4})

| θ, ° | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 100 | 0.35 | 0.91 | -0.07 |
| 120 | 0.35 | 0.81 | -0.07 |
| 140 | 0.34 | 0.73 | -0.07 |
| 160 | 0.44 | 1.16 | -0.10 |
| 170 | 0.97 | 1.51 | +0.29 |
| 175 | 2.16 | 5.00 | +1.37 |

![roll](figures/angle_roll.png)

### droplet (истинные {'theta': 130.0, 'delta': 0.03, 'phi': 0.05})

| θ, ° | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 100 | 0.33 | 0.65 | +0.30 |
| 120 | 0.39 | 0.75 | +0.35 |
| 140 | 0.54 | 0.98 | +0.48 |
| 160 | 3.51 | 17.73 | +3.21 |
| 170 | 5.79 | 10.00 | +4.02 |
| 175 | 5.43 | 11.62 | -2.18 |

![droplet](figures/angle_droplet.png)

### worm (истинные {'theta': 130.0, 'delta': 0.03, 'phi': 0.1})

| θ, ° | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| 100 | 0.24 | 0.56 | +0.17 |
| 120 | 0.33 | 0.73 | +0.33 |
| 140 | 0.46 | 0.96 | +0.45 |
| 160 | 0.80 | 1.66 | +0.80 |
| 170 | 2.07 | 3.54 | +2.07 |
| 175 | 3.99 | 4.99 | +3.93 |

![worm](figures/angle_worm.png)

## Центрирование при ошибке положения стенок 5 % H — O3

### roll (истинные {'theta': 120.0, 'delta': 0.03, 'phi': 0.4})

| центрирование по | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| поре | 3.03 | 5.00 | +2.28 |
| поре + shift | 0.44 | 1.76 | -0.41 |
| жидкости, δ известна | 1.31 | 2.19 | +1.31 |

![roll](figures/centering_roll.png)

### droplet (истинные {'theta': 130.0, 'delta': 0.03, 'phi': 0.05})

| центрирование по | среднее \|Δθ\|, ° | максимум \|Δθ\|, ° | смещение, ° |
|---|---|---|---|
| поре | 0.58 | 1.15 | +0.57 |
| поре + shift | 0.58 | 1.15 | +0.57 |
| жидкости, δ известна | 0.42 | 0.91 | +0.42 |

![droplet](figures/centering_droplet.png)
````

- [ ] **Шаг 3: Выводы**

`validation/README.md`:

```markdown
# Валидация точности метода

`run_accuracy.py` строит синтетические профили по модели, добавляет гауссов шум в долях ρ_bulk на бин и фитирует
ту же форму. Так измеряется ошибка самого метода: фита, дискретизации и шума. Ошибка выбора формы сюда не входит,
поэтому это нижняя граница ошибки на реальных данных. Таблицы и рисунки — `REPORT.md` и `figures/`. Отчёт
пересобирается запуском скрипта: `uv run python validation/run_accuracy.py`, около минуты, нужен `panda-md[plot]`.

Условия: щель 8 × 16 нм, ρ_bulk = 30 нм⁻³, 10 зёрен шума на точку. Настройки фита по умолчанию: свободны θ и δ,
φ берётся из ∫ρ. Истинные параметры: валик θ = 120°, δ = 0.03, φ = 0.4; капля θ = 130°, φ = 0.05; червь θ = 130°,
φ = 0.1.

## Выводы (16.10.2026)

1. **Шум.** Ошибка θ растёт примерно линейно с шумом. При шуме 0.5 % ρ_bulk на бин (200 бинов) средняя |Δθ|
   составляет 0.35° у валика, 0.45° у капли и 0.30° у червя; максимальная — до 0.9°. Критерий ТЗ «|Δθ| < 1°
   в среднем» выполняется до шума около 1 %. При шуме 2 % у валика и червя отдельные прогоны дают ошибку 3–4°.
2. **Число бинов.** Шум пересчитан как √(бинов / 200), то есть статистика траектории одна и та же. Валику хватает
   100 бинов. Капле нужно не меньше 200: при 50 бинах положение края плохо разрешено, и появляется смещение −1.2°.
   Рекомендация: 200 бинов.
3. **Угол — O11.** До 140° средняя |Δθ| не превышает 0.6°. Ближе к 180° профиль слабо зависит от θ, и ошибка
   резко растёт:
   - капля: 3.5° при 160° (отдельные прогоны до 17.7°), 5–6° при 170–175°;
   - червь: 2° при 170°, 4° при 175°;
   - валик держится дольше: 1° при 170°, 2° при 175°.
   Часть фитов упирается в верхнюю границу 180° — `FitResult` помечает это флагом «у границы». Для статьи:
   углы выше примерно 160° (у капли) и 170° (у валика) метод даёт с ошибкой в единицы градусов, и разброс по
   блокам (`fit_blocks`) нужно приводить вместе со значением.
4. **Центрирование — O3.** Положение стенок смещено на 5 % ширины щели:
   - валик, центрирование по поре: 3.0° в среднем, до 5°;
   - валик, по поре со свободным `shift`: 0.4°;
   - валик, по жидкости при известной δ: 1.3°;
   - у капли разница между способами меньше 0.6°: смещение стенок поглощает δ.
   Рекомендация (как в `tests/README.md`): по умолчанию центрировать по поре; для форм, касающихся обеих стенок,
   при неточных стенках добавлять `shift` к свободным параметрам.
5. **Систематика при малом шуме.** У капли и червя есть смещение +0.5…+0.6° даже при шуме 0.25 %. Его источник —
   дискретизация края профиля: δ определяется с точностью до бина, а θ с δ связан (O10). Смещение меньше
   критерия ТЗ, но в статье его стоит назвать.
```

Сверить каждое число выводов с таблицами `REPORT.md`.

- [ ] **Шаг 4: Проверка и коммит**

Run: `make check` — без ошибок, `482 passed, 28 skipped`.

```bash
git add pyproject.toml validation
git commit -m "Валидация точности: Δθ от шума, числа бинов, угла (O11) и центрирования (O3) — рисунки, отчёт и выводы" \
  -m "При шуме 0.5 % средняя |Δθ| 0.3–0.45°; 200 бинов достаточно; у 160–180° ошибка растёт до единиц градусов (капля — быстрее всего); при ошибке стенок 5 % H валику нужен свободный shift (3.0° → 0.4°)." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 4: Документация стадии

**Files:**
- Modify: `tests/README.md`, `docs/superpowers/specs/2026-10-01-panda-architecture-design.md`, `ROADMAP.md`, `README.md`

- [ ] **Шаг 1: Описание тестов**

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
| трёхмерные формы | `unit/test_shapes_3d.py` | математика | `contains` щели: капля — сферический сегмент на слое δ, червь — цилиндрический сегмент (прямая геометрия, совпадение > 99.95 % точек); для всех шести форм при k = 1 и 0.4 доля точек облака по слоям = профиль, всего — φ (контракт §9); оси симметрии объявлены у валика (горизонтальная), пончика и перфорации (вертикальные, у перфорации — центр отверстия); положение оси через границу ящика; у форм треугольника и квадрата `contains` нет (`NotDerivedError`) | 400 000 случайных точек ячейки | доли по слоям ± 0.02, φ ± 4σ выборки | — |
| облака точек | `synthetic/test_point_clouds.py` | физическая модель | `synthetic_trajectory`: частицы равномерно в форме (ρ_bulk), ось формы блуждает по ячейке, подложка — решётка, щель через границу ящика → `from_trajectory` → `density_profile` → фит (валик, капля, пончик); 2D-карта: с осевым центрированием валик симметричен и в сердцевине ρ_bulk, без него размазан; у перфорации провал плотности в x = 0; форма без оси — ошибка; сохранение карты | щель 4 × 8 нм, k = 0.5, 100 кадров | |Δθ| < 2° (§9), φ ± 3 % | — |
| точность и графики | `unit/test_accuracy_and_plots.py` | физическая модель | `angle_errors` без шума — θ восстановлен; исследование: больше шума — больше ошибка, таблица markdown; шум пересчитывается как √(бинов / 200); все графики (`plot_fit`, `plot_density_map`, `plot_blocks`, `plot_stability`, `plot_accuracy`) строятся из объектов PANDA; без matplotlib — понятная ошибка; `import panda_md` не тянет matplotlib | валик, синтетическая траектория | — | 0.5° |
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

**После итогового ревью.** Плато статистической неэффективности по умолчанию считается только по длинам блока, при которых блоков не меньше 20 (раньше при длинах до n/2 разброс плато для AR(1) был 1–33 при истинном 9; теперь среднее по 10 рядам — в пределах 20 %). Перебор форм пропускает только ошибки, зависящие от формы; ошибка профиля (профиль шире поры, нет ρ_bulk) пробрасывается. Разрыв во времени склеенной траектории длиннее блока даёт пустые блоки — они отбрасываются с предупреждением; перекрытие времени — ошибка с подсказкой gmx trjcat -settime. Молекулы в .tpr — номера молекул GROMACS: связные фрагменты не годятся (у воды TIP4P 23 614 «фрагментов» вместо 11 807 молекул — виртуальный сайт без связей).

## Что выяснилось на стадии 6 — трёхмерные формы (16.10.2026)

**Форма по профилю.** Профиль щели — доля площади сечения ячейки, занятая жидкостью, поэтому трёхмерная форма восстанавливается по нему без новых формул: круг (капля, пончик), отверстие (перфорация), полоса (валик, червь), всё сечение (слой). Прямая геометрия капли (шар) и червя (цилиндр) совпадает с этим построением.

**Облака точек.** Капля при φ = 0.05 в ячейке 4 × 8 нм — около 200 частиц на кадр: за 20 кадров ошибка θ до 3.4° (шум выборки), за 100 — до 1.5°. Сквозной критерий §9 (|Δθ| < 2°) — на 100 кадрах.

**2D-карта воды в щели графита** (каждый 50-й кадр после 2 нс, осевое центрирование по валику): симметрия относительно оси — корреляция с отражением 0.995, в сердцевине 33–35 нм⁻³ (ρ_bulk задачи 1 — 33.13). Валик в этой траектории почти не блуждает: без центрирования карта почти та же.

**Карта усредняет по короткой стороне.** Круглое отверстие перфорации даёт провал плотности в x = 0, а не ноль; для форм с вертикальной осью форму сечения по карте не прочесть.

**После итогового ревью стадии 6.** Ось формы ищется по сглаженной гистограмме частиц в плоскости стенок (гаусс σ = L/16, экстремум жидкости или пустоты, уточнение средним с весами): простое круговое среднее теряло маленькое отверстие перфорации (при φ = 0.92 ошибка до 2.5 нм в ящике 8 нм) и почти полный валик. Теперь на 10 траекториях ошибка < 0.15 нм. Валик, лёгший вдоль длинной стороны (k = 1), — предупреждение. Карта предупреждает о меняющемся ящике, отражается вместе с профилем (флаг flipped), интеграл карты = N; для вертикальной оси есть полоса вокруг оси (width_window_nm). На воде в щели графита карта по-прежнему симметрична (0.992), в центре валика 32.7 нм⁻³.

## Что выяснилось на стадии 7 — точность метода (16.10.2026)

Полные таблицы и рисунки — `validation/REPORT.md`, выводы — `validation/README.md`. Коротко: при шуме 0.5 % средняя |Δθ| 0.3–0.45° (валик, капля, червь); 200 бинов достаточно; у 160–180° ошибка растёт до единиц градусов (у капли быстрее всего) — O11 подтверждён количественно; при ошибке положения стенок 5 % H валику нужен свободный `shift` (3.0° → 0.4°).
````

- [ ] **Шаг 2: Спецификация** (Edit)

После строки D26 добавить:

```markdown
| D27 | Графики — пакет `panda_md.plotting` (extra `plot`, matplotlib импортируется только там), функции от готовых объектов (`plot_fit(result)` вместо метода `FitResult.plot()` — ядро не импортирует `plotting`). Точность метода — `analysis.accuracy` (ошибка θ по зёрнам шума при заданных условиях) и скрипт `validation/run_accuracy.py` → `validation/REPORT.md` (стадия 7) |
```

Строку O11 заменить на:

```markdown
| O11 | ~~θ у 180° плохо определяется при шуме~~ — оценено: до 140° средняя \|Δθ\| < 0.6°, у 160–180° — единицы градусов (капля до 17.7° в отдельных прогонах), часть фитов упирается в 180° (`validation/REPORT.md`) | закрыт стадией 7; в статье — приводить разброс по блокам |
```

- [ ] **Шаг 3: ROADMAP и README**

`ROADMAP.md`, «Фаза 6»: строку `- **6в, графики и валидация:** …` заменить на:

```markdown
- **6в, графики и валидация — выполнено** (план `docs/superpowers/plans/2026-10-16-stage7-validation.md`): `panda_md.plotting` (extra `plot`), `analysis.accuracy`, `validation/run_accuracy.py` → `validation/REPORT.md`: Δθ от шума, числа бинов, θ (O11 закрыт) и центрирования (O3).
```

`README.md`: после примера «From an MD trajectory» добавить абзац:

```markdown
Plots (`pip install 'panda-md[plot]'`): `from panda_md.plotting import plot_fit, plot_density_map` — and the
accuracy of the method on synthetic data is in `validation/REPORT.md`.
```

- [ ] **Шаг 4: Полная проверка и коммит**

Run: `make check`, `make test-env`, `make wheel-check`; Python 3.10: `UV_PROJECT_ENVIRONMENT=.venv-py310 uv sync --python 3.10 --all-extras` и `UV_PROJECT_ENVIRONMENT=.venv-py310 uv run --no-sync pytest -q`. Всё зелёное, `482 passed, 28 skipped`.

```bash
git add tests/README.md docs ROADMAP.md README.md
git commit -m "Стадия 7 (фаза 6в) закрыта: описание тестов, D27, O11 закрыт, ROADMAP, README" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
