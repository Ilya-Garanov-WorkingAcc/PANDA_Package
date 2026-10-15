# Стадия 3 — треугольная и квадратная поры: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Формы треугольной (14) и квадратной (7) пор из PANDA_Polygon по правилам проекта, совпадение с источником до 1e-12, диаграмма устойчивости по капиллярному функционалу, совпадающая с PANDA_Polygon, синтетика для форм с профилем.

**Architecture:** Общая база `PolygonGeometry`/`PolygonPore` (сторона a, длина L, ширина слоя нормировки ly) для обеих пор. Формы треугольника — на базе `TriangleShape` (система поры, z = 0 — центроид), квадрата — на базе `SquareShape`. Профиль и площади у формы необязательны: форма без профиля участвует только в диаграмме устойчивости, фит такой формы — ранняя ошибка `NotDerivedError`. Диаграмма устойчивости — `panda_md.analysis.stability`, общий код для всех пор: в каждой точке (l, φ) — допустимая форма с наименьшим S_LV − cos θ · S_SL.

**Tech Stack:** Python ≥ 3.10, numpy, scipy, pydantic v2, pytest, uv, black, ruff, mypy (strict + плагин pydantic).

**Spec:** `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (§3.2, §6 «устойчивость», O8, O9). Разбор источника — `docs/architecture/05_polygon_pores.md`. Источник формул — https://github.com/Vandyshge/PANDA_Polygon, `docs/model/{triangle,square}.py`, коммит `38ff491` (локальная копия `PANDA_Polygon/`, в git не входит).

**Решения пользователя (15.10.2026).** Переписать формы по правилам проекта (pydantic, `Enum`, типы, говорящие имена), а не копировать модуль источника. Девять форм квадрата с дефектами источника пока не переносить: filament_w4c0, w4c1, w4c2, w4c3, w3c2, w2c0_opposite, w2c1 (условие существования падает на `del l`), perforation и capsule_w4c4 (∫ρ ≠ φ).

**Проверено до написания плана.** Весь код ниже собран и прогнан по задачам в чистом клоне ветки `stage1-core-prototype` (коммит `f299fab`): для каждой задачи тесты задачи падают до реализации с указанной ошибкой, после — `black --check`, `ruff check`, `mypy` и `pytest` проходят (итого 359 тестов, 34 пропуска). Эталонные JSON генератор воспроизводит побайтно. Код переносить дословно.

## Global Constraints

- Все ограничения стадий 1–2 в силе: Python ≥ 3.10 (`class X(str, Enum)`, без `StrEnum`/`Self`); объекты-данные — pydantic, без датаклассов (D14); фиксированные варианты — `Enum`, типы у всех полей, говорящие имена (D17, §2.1 спецификации); black 120, ruff, mypy strict; комментарии и сообщения — по-русски.
- Разработка локальная, в ветке `stage1-core-prototype`; в GitHub не отправлять, пока пользователь не скажет.
- Многострочный текст и markdown — только через Write/Edit: в этой среде Bash-heredoc теряет пустые строки и строки, начинающиеся с `#`. Сообщения коммитов — отдельными `-m` на абзац, последний — `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Проверка перед коммитом: `make check`.
- Формулы PANDA_Polygon переносятся без изменений. Отступления — только там, где источник нефизичен, каждое с тестом и записью в `05_polygon_pores.md` §4: droplet_w2c1 треугольника не существует при S_LV < 0; граница droplet_w2c1 квадрата при θ = 3π/4 строгая.
- Генератор эталона читает `PANDA_Polygon/` (симлинк или копия в корне репозитория); тесты читают только JSON.

## Review Focus

1. Профиль треугольника в системе поры несимметричен: стенки при z = −H/3 и z = 2H/3 (H = √3/2). Ожидание — интегрирование, «вне поры» и сетка эталона берутся из `geometry.coordinate_bounds()`, а не из ±1/2. Тест — `test_profile_is_finite_bounded_and_zero_outside_pore[triangle/*]` и `test_profile_integrates_to_volume_fraction[triangle/*]` — задача 1.
2. θ ровно на границе режима формы (120° у треугольника, 135° у квадрата): источник использует строгие неравенства; ожидание — та же форма не существует на границе. Тест — `test_square_corner_droplet_boundary_is_strict` — задача 3; для треугольника — регрессия условий существования на сетке с θ = 125° и 110° — задача 6.
3. Форма без профиля в фите: ожидание — `NotDerivedError` до приведения профиля и оптимизации, с перечнем форм поры, у которых профиль есть. Тест — `test_fit_of_shape_without_profile_fails_before_optimizing` — задача 4.
4. Формула площади, дающая отрицательное S_LV там, где условие существования истинно (droplet_w2c1 треугольника при 112.6° < θ < 120°): ожидание — форма там не существует, диаграмма не падает на проверке `SurfaceTerms`. Тест — `test_triangle_corner_droplet_does_not_exist_where_its_free_surface_is_negative` — задача 2; полное сканирование θ ∈ [1°, 180°] с шагом 0.5° других таких форм не нашло.
5. Диаграмма устойчивости при равенстве энергий двух форм: ожидание — побеждает первая в списке кандидатов (как `argmin` в PANDA_Polygon); эталон фиксирует порядок форм. Тест — `test_polygon_stability_diagram_matches_panda_polygon` (передаёт порядок эталона) — задача 6.

---

### Задача 1: Общая база многоугольных пор, треугольная пора и её нити

**Files:**
- Create: `src/panda_md/pores/polygon.py`
- Modify: `src/panda_md/pores/square/pore.py` (на базе `PolygonPore`)
- Create: `src/panda_md/pores/triangle/pore.py`, `src/panda_md/pores/triangle/shape.py`, `src/panda_md/pores/triangle/filament.py`, `src/panda_md/pores/triangle/__init__.py`
- Modify: `src/panda_md/__init__.py` (экспорт `TrianglePore`)
- Modify: `tests/contract/test_shapes.py` (треугольник; сетка по стенкам поры; формы без профиля), `tests/unit/test_model.py` (набор форм треугольника)

**Interfaces:**
- Consumes: `Pore`, `PoreGeometry`, `InterfaceShape`, `SurfaceTerms`, `pores.common.circular_segment_factor` (стадии 1–2).
- Produces: `PolygonGeometry(pore_length_ratio, layer_width_ratio)` с `ClassVar section_area`, `profile_integral() = section_area / ly`, `layer_area() = l · ly`; `PolygonPore(side_nm, length_nm, layer_width_nm)` с `geometry_class`, `geometry()`, `characteristic_size_nm() = a`, `volume_nm3()`; `SquareGeometry`, `SquarePore` на этой базе (поведение прежнее). `TrianglePore` (`"triangle"`), `TriangleGeometry.coordinate_bounds() = (−H/3, 2H/3)`; `pores.triangle.pore`: `SQRT3`, `TRIANGLE_SECTION_AREA`, `TRIANGLE_HEIGHT`, `accessible_side(δ) = 1 − 2√3 δ`, `accessible_height`, `accessible_bottom`, `accessible_top`; `pores.triangle.shape`: `TriangleShape` (`side(values)`, `supplementary_angle(values)`), `corner_factor`, `THIRD_PI`, диапазоны `NON_WETTING_FULL`, `NON_WETTING_ABOVE_TWO_THIRDS_PI`, `NON_WETTING_BELOW_TWO_THIRDS_PI`. Восемь нитей: filament_w0c0, filament_w0c0_straight (без θ), filament_w1c0 (оба режима), w2c0, w2c1, w3c0, w3c1, w3c2.

- [ ] **Шаг 1: Тесты**

Заменить `tests/contract/test_shapes.py` целиком:

`tests/contract/test_shapes.py`:

```python
"""Контракт формы — проверяется автоматически для КАЖДОЙ зарегистрированной формы каждой поры.

Новая форма, подключённая декоратором @Pore.register, попадает сюда без правки тестов. Щель проверяется
и с квадратным, и с прямоугольным основанием (k = 0.4). Проверки профиля пропускаются у форм, для которых
профиль не выведен (такие формы участвуют только в диаграмме устойчивости).
"""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import SlitPore, SquarePore, TrianglePore
from panda_md.core.enums import ParameterName, Regime
from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.core.pore import Pore
from panda_md.core.shape import InterfaceShape, regime_of

PORES: tuple[Pore, ...] = (
    SlitPore(width_nm=8.0, length_nm=16.0),
    SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.4),  # прямоугольное основание (RectCell)
    SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
    TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
)
MODELS = [model for pore in PORES for model in pore.available_interfaces()]
INTEGRATION_POINTS = 200_001
OUTSIDE_OFFSETS = (0.01, 0.4)  # точки вне поры: чуть за стенкой и далеко за ней
INTEGRAL_TOLERANCE = 1e-4  # ступенька на краю профиля даёт ошибку трапеций ~ шаг сетки
CENTER_OF_MASS_TOLERANCE = 1e-4
SAMPLE_LIMIT = 12
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = np.linspace(0.005, 0.995, 400)  # мелко: droplet_w3c0 треугольника существует в окне ±1.5 % по φ


def has_contact_angle(model: InterfaceModel) -> bool:
    return ParameterName.CONTACT_ANGLE in model.parameters.names


def regime_angles(model: InterfaceModel) -> list[float]:
    """Три угла внутри каждого объявленного режима (градусы)."""
    angles: list[float] = []
    for lower, upper in model.shape.regimes.values():
        angles.extend(np.linspace(lower, upper, 5)[1:-1].tolist())
    return angles


def feasible_samples(model: InterfaceModel) -> list[dict[str, float]]:
    """Допустимые точки (θ, δ, φ) — не больше SAMPLE_LIMIT, равномерно по сетке; у формы без θ — (δ, φ)."""
    grid: list[dict[str, float]]
    if has_contact_angle(model):
        grid = [
            {"theta": theta, "delta": delta, "phi": float(phi)}
            for theta, delta, phi in itertools.product(regime_angles(model), WALL_LAYERS, VOLUME_FRACTIONS)
        ]
    else:
        grid = [{"delta": delta, "phi": float(phi)} for delta, phi in itertools.product(WALL_LAYERS, VOLUME_FRACTIONS)]
    samples = [values for values in grid if model.is_feasible(**values)]
    step = max(1, len(samples) // SAMPLE_LIMIT)
    return samples[::step]


def integration_grid(model: InterfaceModel) -> np.ndarray:
    lower, upper = model.pore.geometry().coordinate_bounds()
    return np.linspace(lower, upper, INTEGRATION_POINTS)


def outside_points(model: InterfaceModel) -> np.ndarray:
    lower, upper = model.pore.geometry().coordinate_bounds()
    return np.array(
        [lower - OUTSIDE_OFFSETS[1], lower - OUTSIDE_OFFSETS[0], upper + OUTSIDE_OFFSETS[0], upper + OUTSIDE_OFFSETS[1]]
    )


def require_profile(model: InterfaceModel) -> None:
    if not model.shape.has_profile():
        pytest.skip(f"{model_id(model)}: профиль не выведен")


def model_id(model: InterfaceModel) -> str:
    aspect_ratio = getattr(model.pore, "base_aspect_ratio", 1.0)
    suffix = "" if aspect_ratio == 1.0 else f"(k={aspect_ratio})"
    return f"{model.pore.name}{suffix}/{model.shape.name}"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_declares_contract_attributes(model: InterfaceModel) -> None:
    shape = model.shape
    assert isinstance(shape, InterfaceShape)
    assert shape.name in type(model.pore).own_registry()
    if has_contact_angle(model):
        assert shape.regimes, "форма с θ должна объявить хотя бы один режим смачивания"
    else:
        assert shape.regimes == {}, "у формы без θ режимов смачивания нет"
    for regime, (lower, upper) in shape.regimes.items():
        assert isinstance(regime, Regime)
        assert 0.0 <= lower < upper <= 180.0
    assert model.default_optimize, "у формы должен быть хотя бы один свободный параметр по умолчанию"
    assert set(model.default_optimize) <= set(model.parameters.names)


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_exists_somewhere_in_each_regime(model: InterfaceModel) -> None:
    samples = feasible_samples(model)
    assert samples, "ни одной допустимой точки на сетке — проверь feasible"
    if has_contact_angle(model):
        covered = {regime_of(sample["theta"]) for sample in samples}
        assert covered == set(model.shape.regimes), "в каком-то режиме нет допустимых точек — проверь feasible"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_profile_integrates_to_volume_fraction(model: InterfaceModel) -> None:
    require_profile(model)
    grid = integration_grid(model)
    integral_per_phi = model.pore.geometry().profile_integral()
    for values in feasible_samples(model):
        density = model.profile(grid, **values)
        integral = trapezoid(density, grid)
        assert integral == pytest.approx(values["phi"] * integral_per_phi, rel=INTEGRAL_TOLERANCE), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_profile_is_finite_bounded_and_zero_outside_pore(model: InterfaceModel) -> None:
    require_profile(model)
    grid = integration_grid(model)
    outside = outside_points(model)
    for values in feasible_samples(model):
        density = model.profile(grid, **values)
        assert np.all(np.isfinite(density)), values
        assert np.all(density >= 0.0), values
        assert np.all(density <= 1.0 + 1e-12), values
        assert np.all(model.profile(outside, **values) == 0.0), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_center_of_mass_matches_numerical_value(model: InterfaceModel) -> None:
    require_profile(model)
    geometry = model.pore.geometry()
    for values in feasible_samples(model):
        internal = model.internal_values(values)
        declared = model.shape.center_of_mass(geometry, internal)
        numerical = InterfaceShape.center_of_mass(model.shape, geometry, internal)
        assert declared == pytest.approx(numerical, abs=CENTER_OF_MASS_TOLERANCE), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_surface_terms_are_non_negative_where_derived(model: InterfaceModel) -> None:
    for values in feasible_samples(model):
        try:
            terms = model.surface_terms(**values)
        except NotDerivedError:
            pytest.skip(f"{model_id(model)}: площади не выведены")
        assert terms.liquid_liquid_area > 0.0
        assert terms.liquid_solid_area >= 0.0
        assert math.isfinite(model.capillary_energy(**values))
```

Заменить `tests/unit/test_model.py` целиком (набор форм треугольника — пока только нити):

`tests/unit/test_model.py`:

```python
"""InterfaceModel — форма в конкретной поре; значения параметров в единицах пользователя."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.errors import MissingParameterValueError, NotDerivedError, UnknownParameterError
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def test_model_lists_shape_parameters_and_shift() -> None:
    roll = SLIT.interface("roll")
    assert roll.parameters.names == ("theta", "delta", "phi", "shift")
    assert roll.default_optimize == ("theta", "delta")


def test_internal_values_convert_degrees_and_fill_defaults() -> None:
    values = SLIT.interface("roll").internal_values({"theta": 180.0, "phi": 0.4})
    assert values == {"theta": pytest.approx(math.pi), "delta": 0.0, "phi": 0.4, "shift": 0.0}


def test_missing_volume_fraction_is_reported() -> None:
    with pytest.raises(MissingParameterValueError, match="phi"):
        SLIT.interface("roll").profile([0.0], theta=120.0)


def test_unknown_parameter_is_reported() -> None:
    with pytest.raises(UnknownParameterError, match="angle"):
        SLIT.interface("roll").profile([0.0], theta=120.0, phi=0.4, angle=1.0)


def test_roll_profile_matches_closed_form_at_pore_center() -> None:
    theta, delta, phi = math.radians(120.0), 0.03, 0.4
    length_ratio = 2.0
    accessible = 1.0 - 2.0 * delta
    spot_width = (
        phi * length_ratio / accessible
        + accessible * (math.pi - 2 * theta) / (4 * math.cos(theta) ** 2)
        - 0.5 * accessible * math.tan(theta)
    )
    expected = (spot_width + accessible / abs(math.cos(theta)) + accessible * math.tan(theta)) / length_ratio
    profile = SLIT.interface("roll").profile(np.array([0.0, 0.49, 0.6]), theta=120.0, delta=delta, phi=phi)
    assert profile[0] == pytest.approx(expected, rel=1e-12)
    assert profile[1] == 0.0  # внутри слоя δ у стенки
    assert profile[2] == 0.0  # за стенкой


def test_feasibility_follows_volume_fraction_limits() -> None:
    roll = SLIT.interface("roll")
    lower, upper = roll.phi_limits(theta=120.0, delta=0.03, phi=0.5)
    assert 0.0 < lower < upper < 1.0
    assert roll.is_feasible(theta=120.0, delta=0.03, phi=0.5 * (lower + upper))
    assert not roll.is_feasible(theta=120.0, delta=0.03, phi=upper + 0.01)


def test_surface_terms_not_derived_for_slit_doughnut() -> None:
    with pytest.raises(NotDerivedError):
        SLIT.interface("doughnut").surface_terms(theta=120.0, phi=0.1)


def test_each_pore_exposes_its_shapes() -> None:
    assert set(SquarePore.shape_names()) == {"filament_w1c0", "droplet_w1c0"}
    assert set(TrianglePore.shape_names()) == {
        "filament_w0c0",
        "filament_w0c0_straight",
        "filament_w1c0",
        "filament_w2c0",
        "filament_w2c1",
        "filament_w3c0",
        "filament_w3c1",
        "filament_w3c2",
    }
    assert set(SlitPore.shape_names()) == {"droplet", "doughnut", "worm", "roll", "perforation", "layer"}


def test_square_droplet_energy_is_capillary_functional() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    droplet = square.interface("droplet_w1c0")
    values = {"theta": 140.0, "delta": 0.02, "phi": 0.05}
    terms = droplet.surface_terms(**values)
    expected = terms.liquid_liquid_area - math.cos(math.radians(140.0)) * terms.liquid_solid_area
    assert droplet.capillary_energy(**values) == pytest.approx(expected, rel=1e-12)


def test_square_pore_rejects_layer_narrower_than_pore() -> None:
    with pytest.raises(ValidationError, match="меньше стороны"):
        SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=5.0)


def test_layer_energy_is_two_flat_interfaces() -> None:
    rectangle = SlitPore(width_nm=1.0, length_nm=3.0, base_aspect_ratio=0.5)
    assert rectangle.interface("layer").capillary_energy(phi=0.5) == pytest.approx(2 * 0.5 * 9.0)
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: 2 ошибки сборки — `ImportError: cannot import name 'TrianglePore' from 'panda_md'` (`tests/contract/test_shapes.py`) и `ModuleNotFoundError: No module named 'panda_md.pores.triangle'` (`tests/unit/test_model.py`).

- [ ] **Шаг 3: Реализация**

`src/panda_md/pores/polygon.py`:

```python
"""Общее для пор-каналов с многоугольным сечением (квадрат, треугольник).

Соглашения PANDA_Polygon (`docs/model/*.py`, коммит 38ff491): сторона сечения a = 1, длина канала l = L / a,
профиль — поперёк одной стенки, плотность нормирована на слой ящика шириной ly (в долях a), φ — относительно
исходного сечения (без слоя δ).
"""

from abc import ABC
from typing import ClassVar

from pydantic import Field, model_validator

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry


class PolygonGeometry(PoreGeometry, ABC):
    """Безразмерная геометрия канала: длина l и ширина слоя нормировки ly (обе — в долях стороны a)."""

    section_area: ClassVar[float]  # площадь исходного сечения при a = 1

    pore_length_ratio: float = Field(gt=0.0)  # l = L / a
    layer_width_ratio: float = Field(gt=0.0)  # ly — ширина слоя нормировки профиля в долях a

    def profile_integral(self) -> float:
        return self.section_area / self.layer_width_ratio

    def layer_area(self) -> float:
        """l · ly — площадь слоя ящика, на которую делится сечение капли или капсулы."""
        return self.pore_length_ratio * self.layer_width_ratio


class PolygonPore(Pore, ABC):
    """Канал многоугольного сечения со стороной a и длиной L; профиль нормирован на слой ящика шириной ly."""

    geometry_class: ClassVar[type[PolygonGeometry]]

    side_nm: float = Field(gt=0.0)  # a — сторона сечения
    length_nm: float = Field(gt=0.0)  # L — длина канала (период вдоль оси)
    layer_width_nm: float = Field(gt=0.0)  # ширина слоя, на которую нормирован профиль (ширина ящика)

    @model_validator(mode="after")
    def _check_layer_covers_pore(self) -> "PolygonPore":
        if self.layer_width_nm < self.side_nm:
            raise ValueError(f"ширина слоя {self.layer_width_nm} нм меньше стороны поры {self.side_nm} нм")
        return self

    def geometry(self) -> PolygonGeometry:
        return self.geometry_class(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )

    def characteristic_size_nm(self) -> float:
        return self.side_nm

    def volume_nm3(self) -> float:
        return self.geometry_class.section_area * self.side_nm**2 * self.length_nm
```

Заменить `src/panda_md/pores/square/pore.py` целиком:

`src/panda_md/pores/square/pore.py`:

```python
"""Квадратная пора: канал квадратного сечения со стороной a и длиной L вдоль оси.

Соглашения — как в PANDA_Polygon (`docs/model/square.py`, коммит 38ff491): a = 1, z = 0 в центре сечения,
профиль поперёк одной стенки, нормировка на слой шириной ly (в долях a), φ — относительно исходного сечения.
"""

from typing import ClassVar

from panda_md.pores.polygon import PolygonGeometry, PolygonPore

SQUARE_HALF_SIDE = 0.5  # стенки квадрата в безразмерной координате: z = ±0.5
SQUARE_SECTION_AREA = 1.0  # площадь исходного сечения при a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2δ — сторона квадрата, доступная жидкости."""
    return 1.0 - 2.0 * wall_layer


class SquareGeometry(PolygonGeometry):
    section_area: ClassVar[float] = SQUARE_SECTION_AREA

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SQUARE_HALF_SIDE, SQUARE_HALF_SIDE)


class SquarePore(PolygonPore):
    name: ClassVar[str] = "square"
    geometry_class: ClassVar[type[PolygonGeometry]] = SquareGeometry

    def geometry(self) -> SquareGeometry:
        return SquareGeometry(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )
```

`src/panda_md/pores/triangle/pore.py`:

```python
"""Треугольная пора: канал с сечением — равносторонним треугольником со стороной a.

Соглашения — как в PANDA_Polygon (`docs/model/triangle.py`, коммит 38ff491): a = 1, профиль — по высоте,
перпендикулярно основанию; z = 0 — центроид (пересечение биссектрис), основание при z = −H/3, вершина при z = 2H/3,
H = √3 a / 2. Слой δ сдвигает все три стенки внутрь: доступная сторона a′ = 1 − 2√3 δ.
"""

import math
from typing import ClassVar

from panda_md.pores.polygon import PolygonGeometry, PolygonPore

SQRT3 = math.sqrt(3.0)
TRIANGLE_SECTION_AREA = SQRT3 / 4.0  # площадь исходного сечения при a = 1
TRIANGLE_HEIGHT = SQRT3 / 2.0  # высота исходного треугольника при a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2√3 δ — сторона треугольника, доступная жидкости."""
    return 1.0 - 2.0 * SQRT3 * wall_layer


def accessible_height(wall_layer: float) -> float:
    """H′ = √3 a′ / 2."""
    return 0.5 * SQRT3 * accessible_side(wall_layer)


def accessible_bottom(wall_layer: float) -> float:
    """Основание доступного треугольника: −H′ / 3."""
    return -accessible_height(wall_layer) / 3.0


def accessible_top(wall_layer: float) -> float:
    """Вершина доступного треугольника: 2 H′ / 3."""
    return 2.0 * accessible_height(wall_layer) / 3.0


class TriangleGeometry(PolygonGeometry):
    section_area: ClassVar[float] = TRIANGLE_SECTION_AREA

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-TRIANGLE_HEIGHT / 3.0, 2.0 * TRIANGLE_HEIGHT / 3.0)


class TrianglePore(PolygonPore):
    name: ClassVar[str] = "triangle"
    geometry_class: ClassVar[type[PolygonGeometry]] = TriangleGeometry

    def geometry(self) -> TriangleGeometry:
        return TriangleGeometry(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )
```

`src/panda_md/pores/triangle/shape.py`:

```python
"""Общее для форм треугольной поры: система отсчёта поры и функции угла, повторяющиеся в формулах."""

import math
from typing import ClassVar

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.pores.triangle.pore import TriangleGeometry, accessible_side

THIRD_PI = math.pi / 3.0
NON_WETTING_ABOVE_TWO_THIRDS_PI = (120.0, 180.0)  # формулы выведены при θ > 2π/3
NON_WETTING_FULL = (90.0, 180.0)
NON_WETTING_BELOW_TWO_THIRDS_PI = (90.0, 120.0)  # формулы выведены при π/2 < θ < 2π/3


class TriangleShape(InterfaceShape[TriangleGeometry]):
    """Форма треугольной поры: формулы записаны в системе поры (0 — центроид сечения)."""

    frame: ClassVar[Frame] = Frame.PORE

    @staticmethod
    def side(values: ParameterValues) -> float:
        """a′ — доступная сторона треугольника."""
        return accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])

    @staticmethod
    def supplementary_angle(values: ParameterValues) -> float:
        """θ′ = π − θ — угол через смачивающую жидкость (так записаны формулы PANDA_Polygon)."""
        return math.pi - values[ParameterName.CONTACT_ANGLE]


def corner_factor(supplementary_angle: float) -> float:
    """θ′ − π/3 − sin(2θ′ − π/3) + √3/2 — площадь скруглённого пустого угла, делённая на R²."""
    return supplementary_angle - THIRD_PI - math.sin(2.0 * supplementary_angle - THIRD_PI) + math.sqrt(3.0) / 2.0
```

`src/panda_md/pores/triangle/filament.py`:

```python
"""Нити (filament) в треугольной поре: однородны вдоль оси канала, поэтому профиль от l не зависит.

Формулы — PANDA_Polygon `docs/model/triangle.py` (коммит 38ff491), переносятся дословно. Имя формы
`filament_w<N>c<M>`: касается N стенок, занимает M углов. Профиль плотности в источнике выведен только у
filament_w0c0 и filament_w3c0; остальные нити участвуют в диаграмме устойчивости, но не в фите.
"""

import math
from abc import abstractmethod
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_factor
from panda_md.pores.triangle.pore import SQRT3, TRIANGLE_SECTION_AREA, TriangleGeometry, TrianglePore, accessible_height
from panda_md.pores.triangle.shape import (
    NON_WETTING_ABOVE_TWO_THIRDS_PI,
    NON_WETTING_FULL,
    THIRD_PI,
    TriangleShape,
    corner_factor,
)

ANGLE_SHAPE_PARAMETERS = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
NO_ANGLE_PARAMETERS = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)


class TriangleFilament(TriangleShape):
    """Нить: площади поверхностей — периметры сечения, умноженные на длину канала l."""

    @abstractmethod
    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """(P_LV, P_SL) — длины границ жидкость–жидкость и жидкость–стенка в сечении."""

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        liquid_liquid, liquid_solid = self.perimeters(geometry, values)
        length = geometry.pore_length_ratio
        return SurfaceTerms(liquid_liquid_area=liquid_liquid * length, liquid_solid_area=liquid_solid * length)


@TrianglePore.register
class TriangleFilamentNoWall(TriangleFilament):
    """filament_w0c0: круглый цилиндр по оси канала, стенок не касается — контактного угла нет."""

    name: ClassVar[str] = "filament_w0c0"
    parameters: ClassVar[tuple[Parameter, ...]] = NO_ANGLE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    def cylinder_radius(self, values: ParameterValues) -> float:
        """π r² = φ √3 / 4 — площадь круга равна доле φ сечения."""
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * SQRT3 / (4.0 * math.pi))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """0 < φ ≤ π a′² / (3√3) — круг вписан в доступный треугольник."""
        side = self.side(values)
        if side <= 0.0:
            return False
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return 0.0 < volume_fraction <= math.pi * side**2 / (3.0 * SQRT3)

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        return 2.0 * math.pi * self.cylinder_radius(values), 0.0

    def center_of_mass(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return 0.0  # ось цилиндра — в центроиде

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(r² − z²) / ly при |z| ≤ r."""
        radius = self.cylinder_radius(values)
        chord = 2.0 * np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        return np.where(np.abs(coordinate) <= radius, chord / geometry.layer_width_ratio, 0.0)


@TrianglePore.register
class TriangleFilamentStraight(TriangleFilament):
    """filament_w0c0_straight: прямой шнур по оси канала, стенок не касается — контактного угла нет."""

    name: ClassVar[str] = "filament_w0c0_straight"
    parameters: ClassVar[tuple[Parameter, ...]] = NO_ANGLE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """0 < φ и √φ ≤ a′."""
        side = self.side(values)
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return side > 0.0 and volume_fraction > 0.0 and math.sqrt(volume_fraction) <= side

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 3√φ (area_filament_w0c0_straight_triangle), S_SL = 0."""
        return 3.0 * math.sqrt(values[ParameterName.VOLUME_FRACTION]), 0.0


@TrianglePore.register
class TriangleFilamentOneWall(TriangleFilament):
    """filament_w1c0: цилиндрический сегмент на основании, углы свободны."""

    name: ClassVar[str] = "filament_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {
        Regime.WETTING: (0.0, 90.0),
        Regime.NON_WETTING: NON_WETTING_FULL,
    }

    def arc_radius(self, values: ParameterValues) -> float:
        """R² (θ − sin θ cos θ) = φ √3 / 4."""
        segment = circular_segment_factor(values[ParameterName.CONTACT_ANGLE])
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * SQRT3 / (4.0 * segment))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """Хорда помещается на основании и дуга не касается боковых стенок (cond_filament_w1c0_triangle)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if circular_segment_factor(contact_angle) <= 0.0 or side <= 0.0:
            return False
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        radius = self.arc_radius(values)
        fits_base = 2.0 * radius * math.sin(contact_angle) <= side
        # расстояние до боковой стенки как квадратичная функция высоты z над основанием: max < 0 — касания нет
        quadratic = -4.0 / 3.0
        linear = side / SQRT3 - 2.0 * radius * math.cos(contact_angle)
        constant = (radius * math.sin(contact_angle)) ** 2 - 0.25 * side**2
        top = radius * (1.0 - math.cos(contact_angle))
        largest = max(constant, quadratic * top**2 + linear * top + constant)
        vertex = -linear / (2.0 * quadratic)
        if 0.0 <= vertex <= top:
            largest = max(largest, quadratic * vertex**2 + linear * vertex + constant)
        return bool(fits_base and largest < 0.0)

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 2Rθ, P_SL = 2R sin θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.arc_radius(values)
        return 2.0 * radius * contact_angle, 2.0 * radius * math.sin(contact_angle)


@TrianglePore.register
class TriangleFilamentTwoWalls(TriangleFilament):
    """filament_w2c0: жидкость в угле между двумя стенками без заполнения угла (θ > 2π/3)."""

    name: ClassVar[str] = "filament_w2c0"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    def arc_radius(self, values: ParameterValues) -> float:
        """R² (π − 2θ′ + sin 2θ′) = φ √3 / 4."""
        supplementary = self.supplementary_angle(values)
        factor = math.pi - 2.0 * supplementary + math.sin(2.0 * supplementary)
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * TRIANGLE_SECTION_AREA / factor)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        supplementary = self.supplementary_angle(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= 2.0 * THIRD_PI:
            return False
        if math.pi - 2.0 * supplementary + math.sin(2.0 * supplementary) <= 0.0:
            return False
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        if not 0.0 < volume_fraction < side**2:
            return False
        radius = self.arc_radius(values)
        chord_fits = 2.0 * radius * math.sin(THIRD_PI + supplementary) <= side
        stays_below_top = radius * (2.0 * math.cos(supplementary) + 1.0) <= SQRT3 / 2.0 * side
        return chord_fits and stays_below_top

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 2R(π − 2θ′), P_SL = 4R sin θ′."""
        supplementary = self.supplementary_angle(values)
        radius = self.arc_radius(values)
        return 2.0 * radius * (math.pi - 2.0 * supplementary), 4.0 * radius * math.sin(supplementary)


@TrianglePore.register
class TriangleFilamentTwoWallsOneCorner(TriangleFilament):
    """filament_w2c1: жидкость заполняет угол между двумя стенками (θ > π/2)."""

    name: ClassVar[str] = "filament_w2c1"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_FULL}

    @staticmethod
    def area_factor(supplementary: float) -> float:
        """2π/3 − θ′ + √3/2 + sin(π/3 + 2θ′) — площадь сечения, делённая на R²."""
        return 2.0 * THIRD_PI - supplementary + SQRT3 / 2.0 + math.sin(THIRD_PI + 2.0 * supplementary)

    def arc_radius(self, values: ParameterValues) -> float:
        factor = self.area_factor(self.supplementary_angle(values))
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * TRIANGLE_SECTION_AREA / factor)

    def contact_length(self, values: ParameterValues) -> float:
        """x = 2R sin(π/3 + θ′) — длина пятна на каждой из двух стенок."""
        return 2.0 * self.arc_radius(values) * math.sin(THIRD_PI + self.supplementary_angle(values))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """Как в источнике: проверяется только нижняя граница θ > π/2; φ < 1 (cond_filament_w2c1_triangle)."""
        side = self.side(values)
        supplementary = self.supplementary_angle(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= math.pi / 2.0:
            return False
        if self.area_factor(supplementary) <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        radius = self.arc_radius(values)
        fits_wall = self.contact_length(values) <= side
        stays_below_top = radius * (2.0 * math.cos(supplementary) + 1.0) <= SQRT3 / 2.0 * side
        return fits_wall and stays_below_top

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = R(4π/3 − 2θ′), P_SL = 2x."""
        radius = self.arc_radius(values)
        return radius * (4.0 * THIRD_PI - 2.0 * self.supplementary_angle(values)), 2.0 * self.contact_length(values)


class TriangleFilamentEmptyCorners(TriangleFilament):
    """Нити, занимающие середину сечения при пустых скруглённых углах (filament_w3c0, w3c1, w3c2)."""

    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}
    empty_corner_count: ClassVar[int]  # число пустых скруглённых углов
    corner_area_scale: ClassVar[float]  # √3/12, √3/8 или √3/4 — делитель площади пустых углов в источнике
    contact_limit_fraction: ClassVar[float]  # x ≤ (доля) · a′

    def arc_radius(self, values: ParameterValues) -> float:
        """R² · corner_factor(θ′) = (a′² − φ) · corner_area_scale."""
        empty_area = max(0.0, self.side(values) ** 2 - values[ParameterName.VOLUME_FRACTION])
        return math.sqrt(empty_area * self.corner_area_scale / corner_factor(self.supplementary_angle(values)))

    def corner_contact(self, values: ParameterValues) -> float:
        """x = 2R sin(π/3 − θ′) — расстояние от вершины до точки контакта."""
        return 2.0 * self.arc_radius(values) * math.sin(THIRD_PI - self.supplementary_angle(values))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= 2.0 * THIRD_PI:
            return False
        if corner_factor(self.supplementary_angle(values)) <= 0.0:
            return False
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        if not 0.0 < volume_fraction < side**2 or self.arc_radius(values) <= 0.0:
            return False
        return self.corner_contact(values) <= self.contact_limit_fraction * side

    def perimeters(self, geometry: TriangleGeometry, values: ParameterValues) -> tuple[float, float]:
        """P_LV = 2N R(π/3 − θ′), P_SL = 3a′ − 2N x (N — число пустых углов)."""
        corners = self.empty_corner_count
        radius = self.arc_radius(values)
        liquid_liquid = 2.0 * corners * radius * (THIRD_PI - self.supplementary_angle(values))
        return liquid_liquid, 3.0 * self.side(values) - 2.0 * corners * self.corner_contact(values)


@TrianglePore.register
class TriangleFilamentThreeWalls(TriangleFilamentEmptyCorners):
    """filament_w3c0: середина сечения занята, все три угла пусты."""

    name: ClassVar[str] = "filament_w3c0"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    empty_corner_count: ClassVar[int] = 3
    corner_area_scale: ClassVar[float] = SQRT3 / 12.0
    contact_limit_fraction: ClassVar[float] = 0.5

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """Три участка: нижние скруглённые углы, полная ширина треугольника, верхний скруглённый угол."""
        supplementary = self.supplementary_angle(values)
        side = self.side(values)
        height = accessible_height(values[ParameterName.WALL_LAYER_THICKNESS])
        radius = self.arc_radius(values)
        bottom, top = -height / 3.0, 2.0 * height / 3.0
        lower_center = bottom + radius * math.cos(supplementary)  # центры окружностей нижних углов
        upper_center = top - 2.0 * radius * math.cos(supplementary)  # центр окружности верхнего угла
        vertex_to_contact = 2.0 * radius * math.sin(THIRD_PI - supplementary)
        lower_width = side - 2.0 * SQRT3 * radius * math.cos(supplementary)
        lower_end = bottom + vertex_to_contact * SQRT3 / 2.0
        upper_start = top - vertex_to_contact * SQRT3 / 2.0
        upper_end = upper_center + radius

        width = np.zeros_like(coordinate, dtype=np.float64)
        in_lower = (coordinate >= bottom) & (coordinate <= lower_end)
        width[in_lower] = lower_width + 2.0 * np.sqrt(
            np.maximum(0.0, radius**2 - (coordinate[in_lower] - lower_center) ** 2)
        )
        in_middle = (coordinate > lower_end) & (coordinate <= upper_start)
        width[in_middle] = side - 2.0 / SQRT3 * (coordinate[in_middle] - bottom)
        in_upper = (coordinate > upper_start) & (coordinate <= upper_end)
        width[in_upper] = 2.0 * np.sqrt(np.maximum(0.0, radius**2 - (coordinate[in_upper] - upper_center) ** 2))
        return width / geometry.layer_width_ratio


@TrianglePore.register
class TriangleFilamentThreeWallsTwoCorners(TriangleFilamentEmptyCorners):
    """filament_w3c2: касается трёх стенок, два угла заняты, один пуст (формулы источника)."""

    name: ClassVar[str] = "filament_w3c2"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    empty_corner_count: ClassVar[int] = 2
    corner_area_scale: ClassVar[float] = SQRT3 / 8.0
    contact_limit_fraction: ClassVar[float] = 0.5


@TrianglePore.register
class TriangleFilamentThreeWallsOneCorner(TriangleFilamentEmptyCorners):
    """filament_w3c1: касается трёх стенок, один угол занят (формулы источника)."""

    name: ClassVar[str] = "filament_w3c1"
    parameters: ClassVar[tuple[Parameter, ...]] = ANGLE_SHAPE_PARAMETERS
    empty_corner_count: ClassVar[int] = 1
    corner_area_scale: ClassVar[float] = SQRT3 / 4.0
    contact_limit_fraction: ClassVar[float] = 1.0
```

`src/panda_md/pores/triangle/__init__.py`:

```python
"""Треугольная пора и её формы. Импорт модулей форм регистрирует их в TrianglePore."""

from panda_md.pores.triangle import filament
from panda_md.pores.triangle.pore import TriangleGeometry, TrianglePore

__all__ = ["TriangleGeometry", "TrianglePore", "filament"]
```

Заменить `src/panda_md/__init__.py` целиком:

`src/panda_md/__init__.py`:

```python
"""PANDA: Predicting Angle from Nanoscale Density Analysis."""

from importlib.metadata import PackageNotFoundError, version

from panda_md.core.enums import Frame, LossFunction, Normalization, OptimizerMethod, ParameterName, Regime, Unit
from panda_md.core.errors import PandaError, PandaWarning
from panda_md.core.model import InterfaceModel
from panda_md.core.parameters import Parameter
from panda_md.core.profile import DensityProfile
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = [
    "DensityProfile",
    "FitResult",
    "FitSpec",
    "Frame",
    "InterfaceModel",
    "LossFunction",
    "Normalization",
    "OptimizerMethod",
    "PandaError",
    "PandaWarning",
    "Parameter",
    "ParameterName",
    "Regime",
    "SlitPore",
    "SquarePore",
    "TrianglePore",
    "Unit",
    "__version__",
]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `270 passed, 22 skipped` (пропуски — проверки профиля у шести нитей треугольника без профиля и площади пончика и перфорации щели).

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests/contract/test_shapes.py tests/unit/test_model.py
git commit -m "Треугольная пора: общая база многоугольных пор, восемь нитей из PANDA_Polygon" \
  -m "PolygonGeometry/PolygonPore — общее для квадрата и треугольника; контракт берёт сетку и точки вне поры из coordinate_bounds и пропускает проверки профиля у форм без профиля." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 2: Капли и капсулы треугольника

**Files:**
- Modify: `src/panda_md/pores/common.py` (`circular_segment_area`)
- Create: `src/panda_md/pores/triangle/droplet.py`, `src/panda_md/pores/triangle/capsule.py`
- Modify: `src/panda_md/pores/triangle/__init__.py`, `tests/unit/test_model.py`
- Create: `tests/unit/test_polygon_shapes.py`

**Interfaces:**
- Consumes: `TriangleShape`, `TriangleGeometry`, функции `pores.triangle.pore` (задача 1); `spherical_cap_centroid_offset` (стадия 1).
- Produces: `circular_segment_area(radius: FloatArray, distance: FloatArray | float) -> FloatArray` — площадь сегмента круга, отсечённого хордой на расстоянии d (нужна квадрату в задаче 3). Капли: droplet_w1c0 (профиль, центр масс), w2c0, w2c1 (не существует при S_LV ≤ 0), w3c0 (φ в окне ±1.5 % от объёма, заданного сечением; `FIXED_VOLUME_TOLERANCE`). Капсулы: capsule_w3c0 (профиль; одна длина вставки `insertion_length`), capsule_w3c3 (только площади, перенесено как в источнике).

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_model.py` целиком (набор форм треугольника полный):

`tests/unit/test_model.py`:

```python
"""InterfaceModel — форма в конкретной поре; значения параметров в единицах пользователя."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.errors import MissingParameterValueError, NotDerivedError, UnknownParameterError
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def test_model_lists_shape_parameters_and_shift() -> None:
    roll = SLIT.interface("roll")
    assert roll.parameters.names == ("theta", "delta", "phi", "shift")
    assert roll.default_optimize == ("theta", "delta")


def test_internal_values_convert_degrees_and_fill_defaults() -> None:
    values = SLIT.interface("roll").internal_values({"theta": 180.0, "phi": 0.4})
    assert values == {"theta": pytest.approx(math.pi), "delta": 0.0, "phi": 0.4, "shift": 0.0}


def test_missing_volume_fraction_is_reported() -> None:
    with pytest.raises(MissingParameterValueError, match="phi"):
        SLIT.interface("roll").profile([0.0], theta=120.0)


def test_unknown_parameter_is_reported() -> None:
    with pytest.raises(UnknownParameterError, match="angle"):
        SLIT.interface("roll").profile([0.0], theta=120.0, phi=0.4, angle=1.0)


def test_roll_profile_matches_closed_form_at_pore_center() -> None:
    theta, delta, phi = math.radians(120.0), 0.03, 0.4
    length_ratio = 2.0
    accessible = 1.0 - 2.0 * delta
    spot_width = (
        phi * length_ratio / accessible
        + accessible * (math.pi - 2 * theta) / (4 * math.cos(theta) ** 2)
        - 0.5 * accessible * math.tan(theta)
    )
    expected = (spot_width + accessible / abs(math.cos(theta)) + accessible * math.tan(theta)) / length_ratio
    profile = SLIT.interface("roll").profile(np.array([0.0, 0.49, 0.6]), theta=120.0, delta=delta, phi=phi)
    assert profile[0] == pytest.approx(expected, rel=1e-12)
    assert profile[1] == 0.0  # внутри слоя δ у стенки
    assert profile[2] == 0.0  # за стенкой


def test_feasibility_follows_volume_fraction_limits() -> None:
    roll = SLIT.interface("roll")
    lower, upper = roll.phi_limits(theta=120.0, delta=0.03, phi=0.5)
    assert 0.0 < lower < upper < 1.0
    assert roll.is_feasible(theta=120.0, delta=0.03, phi=0.5 * (lower + upper))
    assert not roll.is_feasible(theta=120.0, delta=0.03, phi=upper + 0.01)


def test_surface_terms_not_derived_for_slit_doughnut() -> None:
    with pytest.raises(NotDerivedError):
        SLIT.interface("doughnut").surface_terms(theta=120.0, phi=0.1)


def test_each_pore_exposes_its_shapes() -> None:
    assert set(SquarePore.shape_names()) == {"filament_w1c0", "droplet_w1c0"}
    assert set(TrianglePore.shape_names()) == {
        "filament_w0c0",
        "filament_w0c0_straight",
        "filament_w1c0",
        "filament_w2c0",
        "filament_w2c1",
        "filament_w3c0",
        "filament_w3c1",
        "filament_w3c2",
        "droplet_w1c0",
        "droplet_w2c0",
        "droplet_w2c1",
        "droplet_w3c0",
        "capsule_w3c0",
        "capsule_w3c3",
    }
    assert set(SlitPore.shape_names()) == {"droplet", "doughnut", "worm", "roll", "perforation", "layer"}


def test_square_droplet_energy_is_capillary_functional() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    droplet = square.interface("droplet_w1c0")
    values = {"theta": 140.0, "delta": 0.02, "phi": 0.05}
    terms = droplet.surface_terms(**values)
    expected = terms.liquid_liquid_area - math.cos(math.radians(140.0)) * terms.liquid_solid_area
    assert droplet.capillary_energy(**values) == pytest.approx(expected, rel=1e-12)


def test_square_pore_rejects_layer_narrower_than_pore() -> None:
    with pytest.raises(ValidationError, match="меньше стороны"):
        SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=5.0)


def test_layer_energy_is_two_flat_interfaces() -> None:
    rectangle = SlitPore(width_nm=1.0, length_nm=3.0, base_aspect_ratio=0.5)
    assert rectangle.interface("layer").capillary_energy(phi=0.5) == pytest.approx(2 * 0.5 * 9.0)
```

`tests/unit/test_polygon_shapes.py`:

```python
"""Отступления от PANDA_Polygon там, где формулы источника дают нефизичный результат (05_polygon_pores.md §4)."""

import math

from panda_md import TrianglePore

TRIANGLE = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0)
NEGATIVE_AREA_ANGLE_DEG = 2.0 * math.degrees(math.atan(1.5))  # 1 + cos θ = ⅔ sin θ: S_LV droplet_w2c1 меняет знак


def test_triangle_corner_droplet_does_not_exist_where_its_free_surface_is_negative() -> None:
    droplet = TRIANGLE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=100.0, delta=0.0, phi=0.02)
    assert droplet.surface_terms(theta=100.0, delta=0.0, phi=0.02).liquid_liquid_area > 0.0
    for theta in (NEGATIVE_AREA_ANGLE_DEG + 0.1, 115.0, 119.0):
        assert not droplet.is_feasible(theta=theta, delta=0.0, phi=0.02), theta
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: `2 failed` — `test_each_pore_exposes_its_shapes` (нет капель и капсул треугольника) и `test_triangle_corner_droplet_does_not_exist_where_its_free_surface_is_negative` (`UnknownShapeError: в поре triangle нет формы droplet_w2c1`).

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/pores/common.py` целиком:

`src/panda_md/pores/common.py`:

```python
"""Геометрия, общая для форм разных пор: сферический и круговой сегменты."""

import math

import numpy as np

from panda_md.core.types import FloatArray


def spherical_cap_volume_factor(contact_angle: float) -> float:
    """2 − 3 cos θ + cos³ θ: объём сферического сегмента = π R³ / 3 · (этот множитель)."""
    cosine = math.cos(contact_angle)
    return 2.0 - 3.0 * cosine + cosine**3


def spherical_cap_centroid_offset(radius: float, contact_angle: float) -> float:
    """Расстояние от центра сферы до центра масс сегмента: 3 R cos⁴(θ/2) / (2 + cos θ)."""
    return 3.0 * radius * math.cos(0.5 * contact_angle) ** 4 / (2.0 + math.cos(contact_angle))


def circular_segment_factor(contact_angle: float) -> float:
    """θ − sin θ cos θ: площадь кругового сегмента с углом θ = R² · (этот множитель)."""
    return contact_angle - math.sin(contact_angle) * math.cos(contact_angle)


def circular_segment_centroid_offset(radius: float, contact_angle: float) -> float:
    """Расстояние от центра окружности до центра масс сегмента: 4 R sin³θ / (3 (2θ − sin 2θ))."""
    return 4.0 * radius * math.sin(contact_angle) ** 3 / (3.0 * (2.0 * contact_angle - math.sin(2.0 * contact_angle)))


def circular_segment_area(radius: FloatArray, distance: FloatArray | float) -> FloatArray:
    """Площадь сегмента круга радиуса r, отсечённого хордой на расстоянии d от центра (при r ≤ d — ноль)."""
    ratio = np.clip(np.divide(distance, radius, out=np.ones_like(radius), where=radius > 0), -1.0, 1.0)
    return radius**2 * np.arccos(ratio) - distance * np.sqrt(np.maximum(0.0, radius**2 - distance**2))
```

`src/panda_md/pores/triangle/droplet.py`:

```python
"""Капли (droplet) в треугольной поре. Формулы — PANDA_Polygon `docs/model/triangle.py` (коммит 38ff491).

Профиль плотности в источнике выведен только у droplet_w1c0; остальные капли — для диаграммы устойчивости.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import spherical_cap_centroid_offset
from panda_md.pores.triangle.pore import SQRT3, TRIANGLE_SECTION_AREA, TriangleGeometry, TrianglePore, accessible_bottom
from panda_md.pores.triangle.shape import (
    NON_WETTING_ABOVE_TWO_THIRDS_PI,
    NON_WETTING_BELOW_TWO_THIRDS_PI,
    NON_WETTING_FULL,
    THIRD_PI,
    TriangleShape,
)

DROPLET_PARAMETERS = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
FIXED_VOLUME_TOLERANCE = 0.015  # droplet_w3c0: объём задан геометрией, φ допускается в пределах 1.5 % (источник)


def liquid_volume(geometry: TriangleGeometry, values: ParameterValues) -> float:
    """V = φ · (√3/4) · l — объём жидкости в канале длины l."""
    return values[ParameterName.VOLUME_FRACTION] * TRIANGLE_SECTION_AREA * geometry.pore_length_ratio


def accessible_height_of(side: float) -> float:
    return SQRT3 * side / 2.0


@TrianglePore.register
class TriangleDropletOneWall(TriangleShape):
    """droplet_w1c0: сферический сегмент на основании, боковых стенок не касается."""

    name: ClassVar[str] = "droplet_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_FULL}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """π/3 (2 − 3 cos θ + cos³ θ): объём сегмента = R³ · (этот множитель)."""
        cosine = math.cos(contact_angle)
        return math.pi / 3.0 * (2.0 - 3.0 * cosine + cosine**3)

    def sphere_radius(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return float(np.cbrt(liquid_volume(geometry, values) / self.volume_factor(values[ParameterName.CONTACT_ANGLE])))

    def sphere_center(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        bottom = accessible_bottom(values[ParameterName.WALL_LAYER_THICKNESS])
        return bottom - self.sphere_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """Сфера короче канала, пятно на основании, сегмент не касается боковых стенок (cond_droplet_w1c0)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or contact_angle < math.pi / 2.0 or self.volume_factor(contact_angle) <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        radius = self.sphere_radius(geometry, values)
        fits_length = 2.0 * radius <= geometry.pore_length_ratio
        fits_base = 2.0 * radius * math.sin(contact_angle) <= side
        # расстояние до боковой стенки как квадратичная функция высоты над основанием: max < 0 — касания нет
        quadratic = 4.0 / 3.0
        linear = side / SQRT3 - 2.0 * radius * math.cos(contact_angle)
        constant = -0.25 * side**2 + (radius * math.sin(contact_angle)) ** 2
        top = radius * (1.0 - math.cos(contact_angle))
        largest = max(constant, -quadratic * top**2 + linear * top + constant)
        vertex = linear / (2.0 * quadratic)
        if 0.0 <= vertex <= top:
            largest = max(largest, -quadratic * vertex**2 + linear * vertex + constant)
        return bool(fits_length and fits_base and largest < 0.0)

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π R² (1 − cos θ), S_SL = π R² sin² θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=2.0 * math.pi * radius**2 * (1.0 - math.cos(contact_angle)),
            liquid_solid_area=math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )

    def center_of_mass(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        radius = self.sphere_radius(geometry, values)
        return self.sphere_center(geometry, values) + spherical_cap_centroid_offset(
            radius, values[ParameterName.CONTACT_ANGLE]
        )

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / (l · ly) между основанием и верхушкой капли."""
        radius = self.sphere_radius(geometry, values)
        center = self.sphere_center(geometry, values)
        bottom = accessible_bottom(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = (coordinate >= bottom) & (coordinate <= center + radius)
        section_area = math.pi * np.maximum(0.0, radius**2 - (coordinate - center) ** 2)
        return np.where(inside, section_area / geometry.layer_area(), 0.0)


@TrianglePore.register
class TriangleDropletTwoWalls(TriangleShape):
    """droplet_w2c0: капля в угле между двумя стенками без заполнения угла (θ > 2π/3)."""

    name: ClassVar[str] = "droplet_w2c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """2π/3 (−cos θ)(3 − cos² θ)."""
        cosine = math.cos(contact_angle)
        return 2.0 * math.pi / 3.0 * (-cosine) * (3.0 - cosine**2)

    def sphere_radius(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return float(np.cbrt(liquid_volume(geometry, values) / self.volume_factor(values[ParameterName.CONTACT_ANGLE])))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or contact_angle <= 2.0 * THIRD_PI or self.volume_factor(contact_angle) <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        radius = self.sphere_radius(geometry, values)
        fits_length = 2.0 * radius <= geometry.pore_length_ratio
        misses_third_wall = radius * (1.0 - 2.0 * math.cos(contact_angle)) <= accessible_height_of(side)
        fits_width = radius * (math.sin(contact_angle) + SQRT3 * math.cos(contact_angle)) <= side
        return fits_length and misses_third_wall and fits_width

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = −4π R² cos θ, S_SL = 2π R² sin² θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=-4.0 * math.pi * radius**2 * math.cos(contact_angle),
            liquid_solid_area=2.0 * math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )


@TrianglePore.register
class TriangleDropletTwoWallsOneCorner(TriangleShape):
    """droplet_w2c1: капля заполняет угол между двумя стенками (π/2 < θ < 2π/3).

    Отступление от источника: при θ > 2 arctg(3/2) ≈ 112.6° формула S_LV = 2π R² (1 + cos θ − ⅔ sin θ)
    отрицательна, а cond_droplet_w2c1_triangle форму допускает; здесь форма там не существует
    (05_polygon_pores.md §4, вопрос автору).
    """

    name: ClassVar[str] = "droplet_w2c1"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_BELOW_TWO_THIRDS_PI}

    @staticmethod
    def area_factors(contact_angle: float) -> tuple[float, float]:
        """(S_LV, S_SL) / R²: 2π (1 + cos θ − 2 sin θ / 3) и 2π sin² θ."""
        liquid_liquid = 2.0 * math.pi * (1.0 + math.cos(contact_angle) - 2.0 * math.sin(contact_angle) / 3.0)
        return liquid_liquid, 2.0 * math.pi * math.sin(contact_angle) ** 2

    def volume_factor(self, contact_angle: float) -> float:
        """(S_LV − cos θ · S_SL) / (3 R²)."""
        liquid_liquid, liquid_solid = self.area_factors(contact_angle)
        return (liquid_liquid - liquid_solid * math.cos(contact_angle)) / 3.0

    def sphere_radius(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        return float(np.cbrt(liquid_volume(geometry, values) / self.volume_factor(values[ParameterName.CONTACT_ANGLE])))

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or not math.pi / 2.0 < contact_angle < 2.0 * THIRD_PI:
            return False
        if self.volume_factor(contact_angle) <= 0.0 or self.area_factors(contact_angle)[0] <= 0.0:
            return False
        if not 0.0 < values[ParameterName.VOLUME_FRACTION] < side**2:
            return False
        radius = self.sphere_radius(geometry, values)
        fits_length = 2.0 * radius <= geometry.pore_length_ratio
        misses_third_wall = radius * (1.0 - 2.0 * math.cos(contact_angle)) <= accessible_height_of(side)
        fits_width = radius * (math.sin(contact_angle) - SQRT3 * math.cos(contact_angle)) <= side
        return fits_length and misses_third_wall and fits_width

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        liquid_liquid, liquid_solid = self.area_factors(values[ParameterName.CONTACT_ANGLE])
        radius_squared = self.sphere_radius(geometry, values) ** 2
        return SurfaceTerms(
            liquid_liquid_area=liquid_liquid * radius_squared, liquid_solid_area=liquid_solid * radius_squared
        )


@TrianglePore.register
class TriangleDropletThreeWalls(TriangleShape):
    """droplet_w3c0: капля касается трёх стенок; её размер задан сечением, φ — с допуском 1.5 % (источник)."""

    name: ClassVar[str] = "droplet_w3c0"
    parameters: ClassVar[tuple[Parameter, ...]] = DROPLET_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    def sphere_radius(self, values: ParameterValues) -> float:
        """R = −a′ / (2√3 cos θ) — сфера, касающаяся трёх стенок под углом θ."""
        return -self.side(values) / (2.0 * SQRT3 * math.cos(values[ParameterName.CONTACT_ANGLE]))

    def droplet_volume(self, values: ParameterValues) -> float:
        cosine = math.cos(values[ParameterName.CONTACT_ANGLE])
        return math.pi * self.sphere_radius(values) ** 3 / 3.0 * (3.0 * cosine**3 - 9.0 * cosine - 2.0)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        side = self.side(values)
        if side <= 0.0 or values[ParameterName.CONTACT_ANGLE] <= 2.0 * THIRD_PI:
            return False
        radius = self.sphere_radius(values)
        if radius <= 0.0 or not 0.0 < values[ParameterName.VOLUME_FRACTION] < 1.0:
            return False
        fixed_volume = self.droplet_volume(values)
        mismatch = abs(liquid_volume(geometry, values) - fixed_volume) / max(abs(fixed_volume), 1e-12)
        return 2.0 * radius <= geometry.pore_length_ratio and mismatch < FIXED_VOLUME_TOLERANCE

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = −2π R² (1 + 3 cos θ), S_SL = 3π (R sin θ)²."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(values)
        return SurfaceTerms(
            liquid_liquid_area=-2.0 * math.pi * radius**2 * (1.0 + 3.0 * math.cos(contact_angle)),
            liquid_solid_area=3.0 * math.pi * (radius * math.sin(contact_angle)) ** 2,
        )
```

`src/panda_md/pores/triangle/capsule.py`:

```python
"""Капсулы (capsule) в треугольной поре: перемычка через сечение с двумя менисками на торцах.

Формулы — PANDA_Polygon `docs/model/triangle.py` (коммит 38ff491). Профиль выведен только у capsule_w3c0.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_area
from panda_md.pores.triangle.pore import (
    SQRT3,
    TRIANGLE_SECTION_AREA,
    TriangleGeometry,
    TrianglePore,
    accessible_bottom,
    accessible_top,
)
from panda_md.pores.triangle.shape import (
    NON_WETTING_ABOVE_TWO_THIRDS_PI,
    NON_WETTING_BELOW_TWO_THIRDS_PI,
    THIRD_PI,
    TriangleShape,
)

CAPSULE_PARAMETERS = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)


def meniscus_radius(side: float, contact_angle: float) -> float:
    """r = −a′ / (2√3 cos θ) — мениск, опирающийся на три стенки под углом θ."""
    return -side / (2.0 * SQRT3 * math.cos(contact_angle))


@TrianglePore.register
class TriangleCapsuleThreeWalls(TriangleShape):
    """capsule_w3c0: вставка касается трёх стенок, углы пусты (θ > 2π/3)."""

    name: ClassVar[str] = "capsule_w3c0"
    parameters: ClassVar[tuple[Parameter, ...]] = CAPSULE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_TWO_THIRDS_PI}

    @staticmethod
    def section_factor(contact_angle: float) -> float:
        """A / a′² = (3θ − 2π − 3 sin θ cos θ) / (12 cos² θ) — сечение вставки (три сегмента срезаны стенками)."""
        cosine = math.cos(contact_angle)
        return (3.0 * contact_angle - 2.0 * math.pi - 3.0 * math.sin(contact_angle) * cosine) / (12.0 * cosine**2)

    def insertion_length(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        """l̃ = (√3/4 φ l − V_торцов) / A, V_торцов = π a′³ (2 + 9 cos θ − 3 cos³ θ) / (72√3 cos³ θ).

        В источнике l̃ записана трижды разными выражениями (условие, площади, профиль); они тождественны.
        """
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        cosine = math.cos(contact_angle)
        ends_volume = math.pi * (2.0 + 9.0 * cosine - 3.0 * cosine**3) / (72.0 * SQRT3 * cosine**3) * side**3
        liquid_volume = TRIANGLE_SECTION_AREA * values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
        return (liquid_volume - ends_volume) / (self.section_factor(contact_angle) * side**2)

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        """cond_capsule_w3c0_triangle: 2π/3 < θ ≤ π, l̃ ≥ 0, l̃ + 2r ≤ l."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or not 2.0 * THIRD_PI < contact_angle <= math.pi:
            return False
        if self.section_factor(contact_angle) <= 0.0 or values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        insertion = self.insertion_length(geometry, values)
        radius = meniscus_radius(side, contact_angle)
        return insertion >= 0.0 and insertion + 2.0 * radius <= geometry.pore_length_ratio

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π r (3h − r) + 2 r (π − 3 arccos(h/r)) l̃, S_SL = 6 l̃ √(r² − h²) + 3π (r² − h²).

        h = a′ / (2√3) — расстояние от центроида до доступной стенки.
        """
        side = self.side(values)
        wall_distance = side / (2.0 * SQRT3)
        radius = meniscus_radius(side, values[ParameterName.CONTACT_ANGLE])
        insertion = self.insertion_length(geometry, values)
        half_chord_squared = radius**2 - wall_distance**2
        liquid_liquid = 2.0 * math.pi * radius * (3.0 * wall_distance - radius)
        liquid_liquid += 2.0 * radius * (math.pi - 3.0 * math.acos(wall_distance / radius)) * insertion
        liquid_solid = 6.0 * insertion * math.sqrt(half_chord_squared) + 3.0 * math.pi * half_chord_squared
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=liquid_solid)

    def profile(self, coordinate: FloatArray, geometry: TriangleGeometry, values: ParameterValues) -> FloatArray:
        """Сечение на высоте z — «стадион» 2 l̃ r_z + π r_z², у которого наклонные стенки срезают сегменты."""
        wall_layer = values[ParameterName.WALL_LAYER_THICKNESS]
        side = self.side(values)
        radius = meniscus_radius(side, values[ParameterName.CONTACT_ANGLE])
        insertion = self.insertion_length(geometry, values)
        top = accessible_top(wall_layer)
        inside = (coordinate >= accessible_bottom(wall_layer)) & (coordinate <= radius)
        section_radius = np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        half_width = (top - coordinate) / SQRT3
        unclipped = 2.0 * insertion * section_radius + np.pi * section_radius**2
        segment = circular_segment_area(section_radius, half_width)
        clipped = 2.0 * insertion * half_width + np.pi * section_radius**2 - 2.0 * segment
        section = np.where(section_radius <= half_width, unclipped, clipped)
        return np.where(inside, section / geometry.layer_area(), 0.0)


@TrianglePore.register
class TriangleCapsuleThreeWallsThreeCorners(TriangleShape):
    """capsule_w3c3: вставка заполняет всё сечение, мениски на торцах (π/2 < θ < 2π/3).

    Перенесено как в источнике. При δ > 0 источник неточен: длина вставки и площадь контакта 3 L_ins записаны
    для a′ = 1 (сечение вставки √3/4 вместо √3 a′²/4) — вопрос автору (05_polygon_pores.md §4).
    """

    name: ClassVar[str] = "capsule_w3c3"
    parameters: ClassVar[tuple[Parameter, ...]] = CAPSULE_PARAMETERS
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_BELOW_TWO_THIRDS_PI}

    @staticmethod
    def _ends(contact_angle: float) -> tuple[float, float]:
        """(1 − 4 cos² θ, аргумент arccos) — общие для объёма и площадей торцов."""
        cosine = math.cos(contact_angle)
        argument = (1.0 - 6.0 * cosine**2) / (2.0 - 6.0 * cosine**2)
        return 1.0 - 4.0 * cosine**2, min(1.0, max(-1.0, argument))

    def insertion_length(self, geometry: TriangleGeometry, values: ParameterValues) -> float:
        """L_ins = φ l − 4 V_торцов / √3 (как в источнике)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        cosine = math.cos(contact_angle)
        root_argument, arccos_argument = self._ends(contact_angle)
        ends_volume = -(3.0 * math.acos(arccos_argument) - math.pi) / (36.0 * SQRT3 * cosine**3)
        ends_volume += cosine / (3.0 * math.sqrt(root_argument))
        ends_volume *= self.side(values) ** 3
        return values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio - 4.0 * ends_volume / SQRT3

    def feasible(self, geometry: TriangleGeometry, values: ParameterValues) -> bool:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        side = self.side(values)
        if side <= 0.0 or not math.pi / 2.0 < contact_angle < 2.0 * THIRD_PI:
            return False
        if self._ends(contact_angle)[0] <= 0.0 or values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        insertion = self.insertion_length(geometry, values)
        radius = meniscus_radius(side, contact_angle)
        return radius > 0.0 and insertion >= 0.0 and insertion + 2.0 * radius <= geometry.pore_length_ratio

    def surface_terms(self, geometry: TriangleGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV — свободные поверхности торцов; S_SL — контакт на торцах + 3 L_ins по стенкам вставки."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        cosine = math.cos(contact_angle)
        tangent = math.tan(contact_angle)
        root_argument, arccos_argument = self._ends(contact_angle)
        side_squared = self.side(values) ** 2
        arcsin_argument = min(1.0, max(-1.0, SQRT3 / tangent))
        contact_ends = (
            -math.sqrt(3.0 * root_argument) / cosine - tangent**2 * math.asin(arcsin_argument)
        ) * side_squared
        free_ends = (3.0 * math.acos(arccos_argument) - math.pi) / (6.0 * cosine**2) * side_squared
        return SurfaceTerms(
            liquid_liquid_area=free_ends,
            liquid_solid_area=contact_ends + 3.0 * self.insertion_length(geometry, values),
        )
```

Заменить `src/panda_md/pores/triangle/__init__.py` целиком:

`src/panda_md/pores/triangle/__init__.py`:

```python
"""Треугольная пора и её формы. Импорт модулей форм регистрирует их в TrianglePore."""

from panda_md.pores.triangle import capsule, droplet, filament
from panda_md.pores.triangle.pore import TriangleGeometry, TrianglePore

__all__ = ["TriangleGeometry", "TrianglePore", "capsule", "droplet", "filament"]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `295 passed, 34 skipped`; контракт находит допустимые точки droplet_w3c0 на сетке 400 значений φ; ∫ρ = φ у capsule_w3c0 выполняется.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/pores tests/unit/test_model.py tests/unit/test_polygon_shapes.py
git commit -m "Треугольная пора: капли и капсулы из PANDA_Polygon" \
  -m "droplet_w2c1 не существует там, где формула источника даёт S_LV < 0 (θ > 2 arctg(3/2)); у capsule_w3c0 три выражения длины вставки источника тождественны — оставлено одно." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 3: Новые формы квадрата

**Files:**
- Create: `src/panda_md/pores/square/shape.py`, `src/panda_md/pores/square/capsule.py`
- Modify: `src/panda_md/pores/square/filament.py` (база `SquareShape`; filament_w0c0, filament_w0c0_straight), `src/panda_md/pores/square/droplet.py` (база `SquareShape`; `SquareCornerDroplet`, droplet_w2c0, droplet_w2c1), `src/panda_md/pores/square/__init__.py`
- Modify: `tests/unit/test_model.py`, `tests/unit/test_polygon_shapes.py`

**Interfaces:**
- Consumes: `PolygonPore`, `SquareGeometry` (задача 1); `circular_segment_area` (задача 2).
- Produces: `SquareShape` (`side(values)`, `floor(values) = −a′/2`), `THREE_QUARTERS_PI`, диапазоны `NON_WETTING_FULL`, `NON_WETTING_BELOW_THREE_QUARTERS_PI`, `NON_WETTING_ABOVE_THREE_QUARTERS_PI`. Пять новых форм с профилем: filament_w0c0, filament_w0c0_straight (без θ), droplet_w2c0, droplet_w2c1 (граница θ < 3π/4 строгая), capsule_w4c0. Существующие filament_w1c0 и droplet_w1c0 переходят на `SquareShape` без изменения формул.

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_model.py` целиком:

`tests/unit/test_model.py`:

```python
"""InterfaceModel — форма в конкретной поре; значения параметров в единицах пользователя."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.errors import MissingParameterValueError, NotDerivedError, UnknownParameterError
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def test_model_lists_shape_parameters_and_shift() -> None:
    roll = SLIT.interface("roll")
    assert roll.parameters.names == ("theta", "delta", "phi", "shift")
    assert roll.default_optimize == ("theta", "delta")


def test_internal_values_convert_degrees_and_fill_defaults() -> None:
    values = SLIT.interface("roll").internal_values({"theta": 180.0, "phi": 0.4})
    assert values == {"theta": pytest.approx(math.pi), "delta": 0.0, "phi": 0.4, "shift": 0.0}


def test_missing_volume_fraction_is_reported() -> None:
    with pytest.raises(MissingParameterValueError, match="phi"):
        SLIT.interface("roll").profile([0.0], theta=120.0)


def test_unknown_parameter_is_reported() -> None:
    with pytest.raises(UnknownParameterError, match="angle"):
        SLIT.interface("roll").profile([0.0], theta=120.0, phi=0.4, angle=1.0)


def test_roll_profile_matches_closed_form_at_pore_center() -> None:
    theta, delta, phi = math.radians(120.0), 0.03, 0.4
    length_ratio = 2.0
    accessible = 1.0 - 2.0 * delta
    spot_width = (
        phi * length_ratio / accessible
        + accessible * (math.pi - 2 * theta) / (4 * math.cos(theta) ** 2)
        - 0.5 * accessible * math.tan(theta)
    )
    expected = (spot_width + accessible / abs(math.cos(theta)) + accessible * math.tan(theta)) / length_ratio
    profile = SLIT.interface("roll").profile(np.array([0.0, 0.49, 0.6]), theta=120.0, delta=delta, phi=phi)
    assert profile[0] == pytest.approx(expected, rel=1e-12)
    assert profile[1] == 0.0  # внутри слоя δ у стенки
    assert profile[2] == 0.0  # за стенкой


def test_feasibility_follows_volume_fraction_limits() -> None:
    roll = SLIT.interface("roll")
    lower, upper = roll.phi_limits(theta=120.0, delta=0.03, phi=0.5)
    assert 0.0 < lower < upper < 1.0
    assert roll.is_feasible(theta=120.0, delta=0.03, phi=0.5 * (lower + upper))
    assert not roll.is_feasible(theta=120.0, delta=0.03, phi=upper + 0.01)


def test_surface_terms_not_derived_for_slit_doughnut() -> None:
    with pytest.raises(NotDerivedError):
        SLIT.interface("doughnut").surface_terms(theta=120.0, phi=0.1)


def test_each_pore_exposes_its_shapes() -> None:
    assert set(SquarePore.shape_names()) == {
        "filament_w0c0",
        "filament_w0c0_straight",
        "filament_w1c0",
        "droplet_w1c0",
        "droplet_w2c0",
        "droplet_w2c1",
        "capsule_w4c0",
    }
    assert set(TrianglePore.shape_names()) == {
        "filament_w0c0",
        "filament_w0c0_straight",
        "filament_w1c0",
        "filament_w2c0",
        "filament_w2c1",
        "filament_w3c0",
        "filament_w3c1",
        "filament_w3c2",
        "droplet_w1c0",
        "droplet_w2c0",
        "droplet_w2c1",
        "droplet_w3c0",
        "capsule_w3c0",
        "capsule_w3c3",
    }
    assert set(SlitPore.shape_names()) == {"droplet", "doughnut", "worm", "roll", "perforation", "layer"}


def test_square_droplet_energy_is_capillary_functional() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    droplet = square.interface("droplet_w1c0")
    values = {"theta": 140.0, "delta": 0.02, "phi": 0.05}
    terms = droplet.surface_terms(**values)
    expected = terms.liquid_liquid_area - math.cos(math.radians(140.0)) * terms.liquid_solid_area
    assert droplet.capillary_energy(**values) == pytest.approx(expected, rel=1e-12)


def test_square_pore_rejects_layer_narrower_than_pore() -> None:
    with pytest.raises(ValidationError, match="меньше стороны"):
        SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=5.0)


def test_layer_energy_is_two_flat_interfaces() -> None:
    rectangle = SlitPore(width_nm=1.0, length_nm=3.0, base_aspect_ratio=0.5)
    assert rectangle.interface("layer").capillary_energy(phi=0.5) == pytest.approx(2 * 0.5 * 9.0)
```

Заменить `tests/unit/test_polygon_shapes.py` целиком:

`tests/unit/test_polygon_shapes.py`:

```python
"""Отступления от PANDA_Polygon там, где формулы источника дают нефизичный результат (05_polygon_pores.md §4)."""

import math

import pytest

from panda_md import SquarePore, TrianglePore

TRIANGLE = TrianglePore(side_nm=1.0, length_nm=2.0, layer_width_nm=1.0)
SQUARE = SquarePore(side_nm=1.0, length_nm=4.0, layer_width_nm=1.0)
NEGATIVE_AREA_ANGLE_DEG = 2.0 * math.degrees(math.atan(1.5))  # 1 + cos θ = ⅔ sin θ: S_LV droplet_w2c1 меняет знак


def test_triangle_corner_droplet_does_not_exist_where_its_free_surface_is_negative() -> None:
    droplet = TRIANGLE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=100.0, delta=0.0, phi=0.02)
    assert droplet.surface_terms(theta=100.0, delta=0.0, phi=0.02).liquid_liquid_area > 0.0
    for theta in (NEGATIVE_AREA_ANGLE_DEG + 0.1, 115.0, 119.0):
        assert not droplet.is_feasible(theta=theta, delta=0.0, phi=0.02), theta


@pytest.mark.parametrize("theta", [135.0, 134.999])
def test_square_corner_droplet_boundary_is_strict(theta: float) -> None:
    """Источник допускает droplet_w2c1 при θ = 3π/4, но площадь там бесконечна; здесь граница строгая."""
    droplet = SQUARE.interface("droplet_w2c1")
    assert droplet.is_feasible(theta=theta, delta=0.0, phi=0.02) == (theta < 135.0)
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: `3 failed` — `test_each_pore_exposes_its_shapes` (у квадрата две формы) и оба `test_square_corner_droplet_boundary_is_strict` (`UnknownShapeError: в поре square нет формы droplet_w2c1`).

- [ ] **Шаг 3: Реализация**

`src/panda_md/pores/square/shape.py`:

```python
"""Общее для форм квадратной поры: система отсчёта поры и доступная сторона."""

import math
from typing import ClassVar

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.pores.square.pore import SquareGeometry, accessible_side

THREE_QUARTERS_PI = 0.75 * math.pi
NON_WETTING_FULL = (90.0, 180.0)
NON_WETTING_BELOW_THREE_QUARTERS_PI = (90.0, 135.0)  # формулы выведены при π/2 < θ < 3π/4
NON_WETTING_ABOVE_THREE_QUARTERS_PI = (135.0, 180.0)  # формулы выведены при θ ≥ 3π/4


class SquareShape(InterfaceShape[SquareGeometry]):
    """Форма квадратной поры: формулы записаны в системе поры (0 — центр сечения)."""

    frame: ClassVar[Frame] = Frame.PORE

    @staticmethod
    def side(values: ParameterValues) -> float:
        """a′ — доступная сторона квадрата."""
        return accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])

    @classmethod
    def floor(cls, values: ParameterValues) -> float:
        """Нижняя доступная стенка: −a′ / 2."""
        return -0.5 * cls.side(values)
```

Заменить `src/panda_md/pores/square/filament.py` целиком:

`src/panda_md/pores/square/filament.py`:

```python
"""Нити (filament) в квадратной поре. Формулы — PANDA_Polygon `square.py` (коммит 38ff491)."""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_centroid_offset, circular_segment_factor
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side
from panda_md.pores.square.shape import SquareShape


@SquarePore.register
class SquareFilamentOneWall(SquareShape):
    """filament_w1c0: цилиндрический сегмент вдоль оси поры на нижней стенке, углы свободны."""

    name: ClassVar[str] = "filament_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {
        Regime.WETTING: (0.0, 90.0),
        Regime.NON_WETTING: (90.0, 180.0),
    }

    def arc_radius(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """R = √(φ / (θ − sin θ cos θ)): площадь сегмента равна φ · a²."""
        return math.sqrt(
            values[ParameterName.VOLUME_FRACTION] / circular_segment_factor(values[ParameterName.CONTACT_ANGLE])
        )

    def arc_center(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        return floor - self.arc_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_filament_w1c0_square: φ > 0 и хорда 2R меньше доступной стороны a′."""
        side = accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        if side <= 0.0 or circular_segment_factor(values[ParameterName.CONTACT_ANGLE]) <= 0.0:
            return False
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        return 2.0 * self.arc_radius(geometry, values) < side

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """Центр масс сегмента: 4 R sin³θ / (3 (2θ − sin 2θ)) от центра окружности."""
        radius = self.arc_radius(geometry, values)
        offset = circular_segment_centroid_offset(radius, values[ParameterName.CONTACT_ANGLE])
        return self.arc_center(geometry, values) + offset

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2 R θ l, S_SL = 2 R sin θ l (area_filament_w1c0_square)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.arc_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=2.0 * radius * contact_angle * geometry.pore_length_ratio,
            liquid_solid_area=2.0 * radius * math.sin(contact_angle) * geometry.pore_length_ratio,
        )

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(R² − (z − z_c)²) / ly (rho_filament_w1c0_square)."""
        radius = self.arc_radius(geometry, values)
        center = self.arc_center(geometry, values)
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = (coordinate >= floor) & (coordinate <= center + radius)
        chord = 2.0 * np.sqrt(np.maximum(radius**2 - (coordinate - center) ** 2, 0.0))
        return np.where(inside, chord / geometry.layer_width_ratio, 0.0)


@SquarePore.register
class SquareFilamentNoWall(SquareShape):
    """filament_w0c0: круглый цилиндр по оси канала, стенок не касается — контактного угла нет."""

    name: ClassVar[str] = "filament_w0c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    @staticmethod
    def cylinder_radius(values: ParameterValues) -> float:
        """π r² = φ — площадь круга равна доле φ сечения."""
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] / math.pi)

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """0 < φ < π a′² / 4 — круг вписан в доступный квадрат (cond_filament_w0c0_square)."""
        side = self.side(values)
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return side > 0.0 and 0.0 < volume_fraction < math.pi * side**2 / 4.0

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π r l, S_SL = 0."""
        liquid_liquid = 2.0 * math.pi * self.cylinder_radius(values) * geometry.pore_length_ratio
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=0.0)

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        return 0.0  # ось цилиндра — в центре сечения

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(r² − z²) / ly при |z| ≤ r."""
        radius = self.cylinder_radius(values)
        chord = 2.0 * np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        return np.where(np.abs(coordinate) <= radius, chord / geometry.layer_width_ratio, 0.0)


@SquarePore.register
class SquareFilamentStraight(SquareShape):
    """filament_w0c0_straight: шнур квадратного сечения b × b по оси канала, стенок не касается."""

    name: ClassVar[str] = "filament_w0c0_straight"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    @staticmethod
    def cord_side(values: ParameterValues) -> float:
        """b² = φ."""
        return math.sqrt(values[ParameterName.VOLUME_FRACTION])

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """0 < φ < a′² (cond_filament_w0c0_straight_square)."""
        side = self.side(values)
        volume_fraction = values[ParameterName.VOLUME_FRACTION]
        return side > 0.0 and 0.0 < volume_fraction < side**2

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 4 b l, S_SL = 0."""
        liquid_liquid = 4.0 * self.cord_side(values) * geometry.pore_length_ratio
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=0.0)

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        return 0.0  # шнур симметричен относительно центра сечения

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = b / ly при |z| ≤ b / 2."""
        cord_side = self.cord_side(values)
        return np.where(np.abs(coordinate) <= cord_side / 2.0, cord_side / geometry.layer_width_ratio, 0.0)
```

Заменить `src/panda_md/pores/square/droplet.py` целиком:

`src/panda_md/pores/square/droplet.py`:

```python
"""Капли (droplet) в квадратной поре. Формулы — PANDA_Polygon `square.py` (коммит 38ff491)."""

import math
from abc import abstractmethod
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import REGIME_BOUNDARY_DEG, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_area, spherical_cap_centroid_offset, spherical_cap_volume_factor
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side
from panda_md.pores.square.shape import (
    NON_WETTING_ABOVE_THREE_QUARTERS_PI,
    NON_WETTING_BELOW_THREE_QUARTERS_PI,
    THREE_QUARTERS_PI,
    SquareShape,
)


@SquarePore.register
class SquareDropletOneWall(SquareShape):
    """droplet_w1c0: сферический сегмент на нижней стенке, не касается других стенок и углов."""

    name: ClassVar[str] = "droplet_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def sphere_radius(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """R из π R³ (2 − 3 cos θ + cos³ θ) / 3 = φ l (r_droplet_w1c0_square)."""
        volume = values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
        return float(
            np.cbrt(3.0 * volume / (math.pi * spherical_cap_volume_factor(values[ParameterName.CONTACT_ANGLE])))
        )

    def sphere_center(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        return floor - self.sphere_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_droplet_w1c0_square: θ ≥ 90°, 2R < l, 2R ≤ a′."""
        side = accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        if side <= 0.0 or math.degrees(values[ParameterName.CONTACT_ANGLE]) < REGIME_BOUNDARY_DEG:
            return False
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        diameter = 2.0 * self.sphere_radius(geometry, values)
        return diameter < geometry.pore_length_ratio and diameter <= side

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        radius = self.sphere_radius(geometry, values)
        return self.sphere_center(geometry, values) + spherical_cap_centroid_offset(
            radius, values[ParameterName.CONTACT_ANGLE]
        )

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2π R² (1 − cos θ), S_SL = π R² sin²θ (area_droplet_w1c0_square)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=2.0 * math.pi * radius**2 * (1.0 - math.cos(contact_angle)),
            liquid_solid_area=math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / (l · ly) (rho_droplet_w1c0_square)."""
        radius = self.sphere_radius(geometry, values)
        center = self.sphere_center(geometry, values)
        floor = -0.5 * accessible_side(values[ParameterName.WALL_LAYER_THICKNESS])
        inside = (coordinate >= floor) & (coordinate <= center + radius)
        section_area = math.pi * np.maximum(radius**2 - (coordinate - center) ** 2, 0.0)
        layer_area = geometry.pore_length_ratio * geometry.layer_width_ratio
        return np.where(inside, section_area / layer_area, 0.0)


class SquareCornerDroplet(SquareShape):
    """Капля в двугранном угле между нижней и боковой стенками: общий профиль droplet_w2c0 и droplet_w2c1.

    Сечение на высоте z — круг радиуса r_z, у которого боковая стенка срезает сегмент, пока
    z лежит между точками контакта L1 и L2 (rho_droplet_w2c0_square, rho_droplet_w2c1_square).
    """

    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)

    @staticmethod
    @abstractmethod
    def volume_factor(contact_angle: float) -> float:
        """F(θ): объём капли = 2 R³ F(θ) / 3."""

    def sphere_radius(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """R = (3 φ l / (2 F(θ)))^(1/3)."""
        volume = values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
        return float(np.cbrt(3.0 * volume / (2.0 * self.volume_factor(values[ParameterName.CONTACT_ANGLE]))))

    def fits(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """φ > 0, 2R ≤ l и R (1 − cos θ) ≤ a′ — общие условия обеих капель в углу."""
        if values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        radius = self.sphere_radius(geometry, values)
        cap_height = radius * (1.0 - math.cos(values[ParameterName.CONTACT_ANGLE]))
        return 2.0 * radius <= geometry.pore_length_ratio and cap_height <= self.side(values)

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        floor = self.floor(values)
        wall_distance = -radius * math.cos(contact_angle)  # от центра сферы до каждой из двух стенок
        center = floor + wall_distance
        cut_start = floor - math.sqrt(2.0) * radius * math.sin(contact_angle + math.pi / 4.0)  # L1_square
        cut_end = floor + math.sqrt(2.0) * radius * math.sin(contact_angle - math.pi / 4.0)  # L2_square
        inside = (coordinate >= floor) & (coordinate <= center + radius)
        section_radius_squared = np.maximum(0.0, radius**2 - (coordinate - center) ** 2)
        section = np.pi * section_radius_squared
        cut = circular_segment_area(np.sqrt(section_radius_squared), wall_distance)
        section = np.where((coordinate > cut_start) & (coordinate < cut_end), section - cut, section)
        return np.where(inside, section / geometry.layer_area(), 0.0)


@SquarePore.register
class SquareDropletTwoWalls(SquareCornerDroplet):
    """droplet_w2c0: капля касается двух соседних стенок, угол между ними не заполнен (θ > 3π/4)."""

    name: ClassVar[str] = "droplet_w2c0"
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_THREE_QUARTERS_PI}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """F = π (−cos θ)(3 − cos² θ) (r_droplet_w2c0_square)."""
        cosine = math.cos(contact_angle)
        return math.pi * (-cosine) * (3.0 - cosine**2)

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_droplet_w2c0_square: θ > 3π/4, 2R ≤ l, R (1 − cos θ) ≤ a′."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        if self.side(values) <= 0.0 or not THREE_QUARTERS_PI < contact_angle <= math.pi:
            return False
        return self.volume_factor(contact_angle) > 0.0 and self.fits(geometry, values)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = −4π R² cos θ, S_SL = 2π R² sin² θ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry, values)
        return SurfaceTerms(
            liquid_liquid_area=-4.0 * math.pi * radius**2 * math.cos(contact_angle),
            liquid_solid_area=2.0 * math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )


@SquarePore.register
class SquareDropletTwoWallsOneCorner(SquareCornerDroplet):
    """droplet_w2c1: капля заполняет угол между двумя соседними стенками (π/2 < θ < 3π/4).

    В источнике условие допускает θ = 3π/4, но площадь там не определена (бесконечна); здесь граница строгая.
    """

    name: ClassVar[str] = "droplet_w2c1"
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_BELOW_THREE_QUARTERS_PI}

    @staticmethod
    def volume_factor(contact_angle: float) -> float:
        """F = arccos(ctg² θ) − cos θ (2 + sin² θ) arccos(ctg θ) + cos² θ √(−cos 2θ)."""
        cosine = math.cos(contact_angle)
        cotangent = cosine / math.sin(contact_angle)
        return (
            math.acos(cotangent**2)
            - cosine * (2.0 + math.sin(contact_angle) ** 2) * math.acos(cotangent)
            + cosine**2 * math.sqrt(-math.cos(2.0 * contact_angle))
        )

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_droplet_w2c1_square: π/2 < θ < 3π/4, 2R ≤ l, R (1 − cos θ) ≤ a′."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        if self.side(values) <= 0.0 or not math.pi / 2.0 < contact_angle < THREE_QUARTERS_PI:
            return False
        return self.volume_factor(contact_angle) > 0.0 and self.fits(geometry, values)

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 2R² (arccos ctg² θ − 2 cos θ arccos ctg θ), S_SL = 2R² (sin² θ arccos ctg θ − cos θ √(−cos 2θ))."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        cosine = math.cos(contact_angle)
        cotangent = cosine / math.sin(contact_angle)
        radius_squared = self.sphere_radius(geometry, values) ** 2
        liquid_liquid = 2.0 * radius_squared * (math.acos(cotangent**2) - 2.0 * cosine * math.acos(cotangent))
        liquid_solid = (
            2.0
            * radius_squared
            * (math.sin(contact_angle) ** 2 * math.acos(cotangent) - cosine * math.sqrt(-math.cos(2.0 * contact_angle)))
        )
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=liquid_solid)
```

`src/panda_md/pores/square/capsule.py`:

```python
"""Капсулы (capsule) в квадратной поре. Формулы — PANDA_Polygon `square.py` (коммит 38ff491).

capsule_w4c4 не переносится: в источнике ∫ρ dz ≠ φ (docs/architecture/05_polygon_pores.md §4).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_area
from panda_md.pores.square.pore import SquareGeometry, SquarePore
from panda_md.pores.square.shape import NON_WETTING_ABOVE_THREE_QUARTERS_PI, THREE_QUARTERS_PI, SquareShape


@SquarePore.register
class SquareCapsuleFourWalls(SquareShape):
    """capsule_w4c0: перемычка касается четырёх стенок, углы пусты; на торцах — мениски (3π/4 ≤ θ ≤ π)."""

    name: ClassVar[str] = "capsule_w4c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: NON_WETTING_ABOVE_THREE_QUARTERS_PI}

    def meniscus_radius(self, values: ParameterValues) -> float:
        """r = a′ / (2 cos θ′), θ′ = π − θ."""
        return self.side(values) / (2.0 * math.cos(math.pi - values[ParameterName.CONTACT_ANGLE]))

    def section_area(self, values: ParameterValues) -> float:
        """S_сеч = r² (π − 4θ′ + 2 sin 2θ′) — сечение вставки (S_cross_capsule_w4c0_square)."""
        supplementary_angle = math.pi - values[ParameterName.CONTACT_ANGLE]
        return self.meniscus_radius(values) ** 2 * (
            math.pi - 4.0 * supplementary_angle + 2.0 * math.sin(2.0 * supplementary_angle)
        )

    def insertion_length(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        """l̃ = (φ l + 4π r³ / 3 − 2π a′ r² + π a′³ / 6) / S_сеч (l_tilde_capsule_w4c0_square)."""
        side = self.side(values)
        radius = self.meniscus_radius(values)
        numerator = (
            values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio
            + 4.0 / 3.0 * math.pi * radius**3
            - 2.0 * math.pi * side * radius**2
            + math.pi / 6.0 * side**3
        )
        return numerator / self.section_area(values)

    def feasible(self, geometry: SquareGeometry, values: ParameterValues) -> bool:
        """cond_capsule_w4c0_square: 3π/4 ≤ θ ≤ π, l̃ ≥ 0, 2r + l̃ ≤ l."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        if self.side(values) <= 0.0 or not THREE_QUARTERS_PI <= contact_angle <= math.pi:
            return False
        if self.section_area(values) <= 0.0 or values[ParameterName.VOLUME_FRACTION] <= 0.0:
            return False
        insertion = self.insertion_length(geometry, values)
        return insertion >= 0.0 and 2.0 * self.meniscus_radius(values) + insertion <= geometry.pore_length_ratio

    def surface_terms(self, geometry: SquareGeometry, values: ParameterValues) -> SurfaceTerms:
        """S_LV = 4π r (a′ − r) + 2 r (π − 4θ′) l̃, S_SL = 4π r² − π a′² + 8 r sin θ′ · l̃."""
        side = self.side(values)
        radius = self.meniscus_radius(values)
        supplementary_angle = math.pi - values[ParameterName.CONTACT_ANGLE]
        insertion = self.insertion_length(geometry, values)
        liquid_liquid = 4.0 * math.pi * radius * (side - radius)
        liquid_liquid += insertion * 2.0 * radius * (math.pi - 4.0 * supplementary_angle)
        liquid_solid = 4.0 * math.pi * radius**2 - math.pi * side**2
        liquid_solid += insertion * 8.0 * radius * math.sin(supplementary_angle)
        return SurfaceTerms(liquid_liquid_area=liquid_liquid, liquid_solid_area=liquid_solid)

    def center_of_mass(self, geometry: SquareGeometry, values: ParameterValues) -> float:
        return 0.0  # капсула симметрична относительно центра сечения

    def profile(self, coordinate: FloatArray, geometry: SquareGeometry, values: ParameterValues) -> FloatArray:
        """Сечение на высоте z — «стадион» 2 l̃ r_z + π r_z², у которого боковые стенки срезают сегменты."""
        floor = self.floor(values)
        half_side = -floor
        radius = self.meniscus_radius(values)
        insertion = self.insertion_length(geometry, values)
        inside = (coordinate >= floor) & (coordinate <= half_side)
        section_radius = np.sqrt(np.maximum(0.0, radius**2 - coordinate**2))
        unclipped = 2.0 * insertion * section_radius + np.pi * section_radius**2
        clipped = (
            2.0 * insertion * half_side
            + np.pi * section_radius**2
            - 2.0 * circular_segment_area(section_radius, half_side)
        )
        section = np.where(section_radius <= half_side, unclipped, clipped)
        return np.where(inside, section / geometry.layer_area(), 0.0)
```

Заменить `src/panda_md/pores/square/__init__.py` целиком:

`src/panda_md/pores/square/__init__.py`:

```python
"""Квадратная пора и её формы. Импорт модулей форм регистрирует их в SquarePore."""

from panda_md.pores.square import capsule, droplet, filament
from panda_md.pores.square.pore import SquareGeometry, SquarePore

__all__ = ["SquareGeometry", "SquarePore", "capsule", "droplet", "filament"]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `327 passed, 34 skipped`; старая регрессия квадрата (`test_square_matches_panda_polygon`) по-прежнему зелёная — перенос на `SquareShape` ничего не изменил.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/pores/square tests/unit/test_model.py tests/unit/test_polygon_shapes.py
git commit -m "Квадратная пора: нити без θ, капли в углу, капсула w4c0 из PANDA_Polygon" \
  -m "Девять форм квадрата с дефектами источника не перенесены (05_polygon_pores.md §4); граница droplet_w2c1 при θ = 3π/4 строгая — там площадь источника бесконечна." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 4: Фит формы без профиля — ранняя ошибка

**Files:**
- Modify: `src/panda_md/core/fitting.py` (проверка в начале `fit`)
- Modify: `tests/unit/test_fitting_details.py`

**Interfaces:**
- Consumes: `InterfaceShape.has_profile()` (стадия 1), `Pore.available_interfaces()`.
- Produces: `fit(model, profile, spec)` поднимает `NotDerivedError("фит формы <имя> невозможен: профиль плотности не выведен ...; формы поры <пора> с профилем: ...")` до приведения профиля.

- [ ] **Шаг 1: Тест**

Заменить `tests/unit/test_fitting_details.py` целиком:

`tests/unit/test_fitting_details.py`:

```python
"""Общий фит: барьер, счётчик вычислений по режимам, сообщения об ошибках, формы без θ и без φ."""

from collections.abc import Iterator
from typing import ClassVar

import numpy as np
import pytest

from panda_md import DensityProfile, LossFunction, Normalization, Regime, SlitPore, SquarePore, TrianglePore
from panda_md.core.enums import Frame
from panda_md.core.errors import (
    InfeasibleParametersError,
    InvalidProfileError,
    NotDerivedError,
    UnsupportedRegimeError,
)
from panda_md.core.fitting import _Objective
from panda_md.core.parameters import CONTACT_ANGLE, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import FloatArray
from panda_md.pores.slit import SlitGeometry
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
SLIT = SlitPore(width_nm=8.0, length_nm=16.0)
SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)


def test_barrier_is_above_any_feasible_loss_and_grows_with_violation() -> None:
    roll = SLIT.interface("roll")
    truth = {"theta": 120.0, "delta": 0.03, "phi": 0.4}
    profile = synthetic_profile(roll, truth, density_bulk_per_nm3=DENSITY_BULK)
    reduced = SLIT.reduce(profile, Normalization.BULK, DENSITY_BULK)
    free_spec = roll.parameters.resolve(["phi"], {"theta": 120.0, "delta": 0.03}, {}, {}, {})
    objective = _Objective(roll, free_spec, reduced, np.ones(reduced.coordinate.shape, dtype=bool), LossFunction.L1)
    _, upper = roll.phi_limits(**truth)
    feasible_loss = objective(np.array([0.5 * upper]))
    slightly_outside = objective(np.array([upper + 0.01]))
    far_outside = objective(np.array([upper + 0.05]))
    assert feasible_loss < objective.barrier_level < slightly_outside < far_outside


def test_evaluation_count_belongs_to_its_regime() -> None:
    filament = SQUARE.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 60.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    both = filament.fit(profile, fixed={"phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK)
    wetting_only = filament.fit(profile, fixed={"phi": 0.05}, regime=Regime.WETTING, density_bulk_per_nm3=DENSITY_BULK)
    assert both.regime is Regime.WETTING
    assert both.evaluation_count == wetting_only.evaluation_count
    assert both.alternatives[0].evaluation_count > 0


def test_infeasible_start_error_lists_ranges() -> None:
    roll = SLIT.interface("roll")
    profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK)
    with pytest.raises(InfeasibleParametersError, match=r"theta ∈ \[90\.5, 180\].*phi = 0\.999"):
        roll.fit(profile, fixed={"phi": 0.999}, density_bulk_per_nm3=DENSITY_BULK)


def test_requested_regime_for_shape_without_contact_angle_is_an_error() -> None:
    layer = SLIT.interface("layer")
    profile = synthetic_profile(layer, {"delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK)
    with pytest.raises(UnsupportedRegimeError, match="нет контактного угла"):
        layer.fit(profile, regime=Regime.WETTING, density_bulk_per_nm3=DENSITY_BULK)


def test_layer_volume_fraction_has_no_systematic_bias() -> None:
    """φ слоя — ступенька: различима с точностью до бина (0.005), но без систематического сдвига.

    Без старта из ∫ρ/ρ_bulk Nelder–Mead вставал на нижний край плато: среднее −0.006 (ревью стадии 2).
    """
    layer = SLIT.interface("layer")
    errors = []
    for volume_fraction in np.linspace(0.05, 0.9, 35):
        profile = synthetic_profile(
            layer, {"delta": 0.03, "phi": float(volume_fraction)}, density_bulk_per_nm3=DENSITY_BULK
        )
        errors.append(layer.fit(profile, density_bulk_per_nm3=DENSITY_BULK).values["phi"] - volume_fraction)
    assert max(abs(error) for error in errors) <= 0.005 + 1e-9
    assert abs(float(np.mean(errors))) < 0.0025


class ContactAngleOnly(InterfaceShape[SlitGeometry]):
    """Форма без параметра φ — только чтобы проверить отказ нормировки mean."""

    name: ClassVar[str] = "test_contact_angle_only"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE,)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE
    default_optimize: ClassVar[tuple[str, ...]] = ("theta",)

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        return True

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        return np.full_like(coordinate, 0.5)


@pytest.fixture
def contact_angle_only() -> Iterator[None]:
    SlitPore.register(ContactAngleOnly)
    yield
    del SlitPore.own_registry()["test_contact_angle_only"]


@pytest.mark.usefixtures("contact_angle_only")
def test_mean_normalization_needs_volume_fraction_parameter() -> None:
    profile = DensityProfile(
        coordinate_nm=np.linspace(-3.98, 3.98, 200),
        density_per_nm3=np.full(200, 10.0),
        particle_count=1000.0,
        pore_volume_nm3=SLIT.volume_nm3(),
    )
    with pytest.raises(InvalidProfileError, match="φ"):
        SLIT.interface("test_contact_angle_only").fit(profile)


def test_fit_of_shape_without_profile_fails_before_optimizing() -> None:
    triangle = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    profile = synthetic_profile(
        triangle.interface("droplet_w1c0"),
        {"theta": 140.0, "delta": 0.02, "phi": 0.1},
        density_bulk_per_nm3=DENSITY_BULK,
    )
    corner_droplet = triangle.interface("droplet_w2c0")
    with pytest.raises(NotDerivedError, match=r"фит формы droplet_w2c0 невозможен.*droplet_w1c0"):
        corner_droplet.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    with pytest.raises(NotDerivedError, match=r"droplet_w2c0.*профиль"):
        synthetic_profile(
            corner_droplet, {"theta": 140.0, "delta": 0.02, "phi": 0.1}, density_bulk_per_nm3=DENSITY_BULK
        )
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Run: `uv run pytest -q tests/unit/test_fitting_details.py`
Expected: `1 failed` — `Regex pattern did not match`, фактическое сообщение `droplet_w2c0: профиль плотности не выведен` (ошибка возникает только в проверке «параметр влияет», после поиска стартов).

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/core/fitting.py` целиком:

`src/panda_md/core/fitting.py`:

```python
"""Общий фит: один алгоритм для любой формы, любой поры и любого набора свободных параметров."""

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import Bounds, minimize

from panda_md.core.enums import Frame, LossFunction, Normalization, OptimizerMethod, ParameterName, Regime
from panda_md.core.errors import (
    EmptyWindowError,
    InfeasibleParametersError,
    InvalidProfileError,
    NotDerivedError,
    PandaError,
    ProfileMismatchError,
    UnidentifiableParameterError,
    UnsupportedRegimeError,
)
from panda_md.core.parameters import FreeSpec, ParameterValues
from panda_md.core.profile import DensityProfile, ReducedProfile
from panda_md.core.result import FitResult
from panda_md.core.shape import REGIME_BOUNDARY_DEG, REGIME_GAP_DEG, PoreGeometry
from panda_md.core.spec import FitSpec
from panda_md.core.types import BoolArray, FloatArray

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel


MIN_WINDOW_POINTS = 10  # меньше точек в окне фита — ошибка EmptyWindowError
# шаги проверки «параметр влияет», доли ширины его границ: крупные нужны параметрам, двигающим лишь край
# профиля (φ у слоя), — малый сдвиг края не выходит за бин
IDENTIFIABILITY_STEP_FRACTIONS = (1e-3, 1e-2, 5e-2)
IDENTIFIABILITY_TOLERANCE = 1e-9  # относительное изменение профиля, ниже которого параметр «не влияет»
BOUND_PROXIMITY_FRACTION = 1e-3  # ближе этой доли ширины границ — параметр «у границы»
BARRIER_LEVEL_FACTOR = 10.0  # уровень барьера: во столько раз выше ошибки нулевой модели (+1)
BARRIER_SLOPE = 100.0  # наклон барьера по величине выхода φ за границы существования
COVERAGE_TOLERANCE = 0.05  # данные должны доходить до стенок с точностью 5 % размера поры
OUTSIDE_MASS_TOLERANCE = 0.05  # допустимая доля ∫ρ за пределами поры (система поры)

CONTACT_ANGLE_NAME = ParameterName.CONTACT_ANGLE.value
WALL_LAYER_NAME = ParameterName.WALL_LAYER_THICKNESS.value
VOLUME_FRACTION_NAME = ParameterName.VOLUME_FRACTION.value
PROFILE_SHIFT_NAME = ParameterName.PROFILE_SHIFT.value


class _Objective:
    """Целевая функция фита: модель в системе отсчёта данных, сравнение с данными, барьер."""

    model: "InterfaceModel"
    geometry: PoreGeometry
    free_spec: FreeSpec
    coordinate: FloatArray
    data: FloatArray
    frame: Frame
    normalization: Normalization
    loss_function: LossFunction
    barrier_level: float
    evaluation_count: int

    def __init__(
        self,
        model: "InterfaceModel",
        free_spec: FreeSpec,
        reduced: ReducedProfile,
        window_mask: BoolArray,
        loss_function: LossFunction,
    ) -> None:
        self.model = model
        self.geometry = model.pore.geometry()
        self.free_spec = free_spec
        self.coordinate = reduced.coordinate[window_mask]
        self.data = reduced.density[window_mask]
        self.frame = reduced.frame
        self.normalization = reduced.normalization
        self.loss_function = loss_function
        self.barrier_level = BARRIER_LEVEL_FACTOR * (self.loss(np.zeros_like(self.data)) + 1.0)
        self.evaluation_count = 0

    def loss(self, curve: FloatArray) -> float:
        residual = curve - self.data
        if self.loss_function is LossFunction.L1:
            return float(np.mean(np.abs(residual)))
        return float(np.mean(residual**2))

    def prediction(self, values: ParameterValues) -> FloatArray:
        """Модельная безразмерная плотность на координатах данных."""
        shape = self.model.shape
        position = self.coordinate - values.get(PROFILE_SHIFT_NAME, 0.0)
        if self.frame is not shape.frame:
            if self.frame is Frame.LIQUID and shape.frame is Frame.PORE:
                position = position + shape.center_of_mass(self.geometry, values)
            else:
                raise PandaError(
                    f"данные в системе {self.frame.value}, формулы формы {shape.name} — в системе "
                    f"{shape.frame.value}: такой пересчёт не поддержан"
                )
        curve = shape.profile(position, self.geometry, values)
        if self.normalization is Normalization.MEAN:
            curve = curve / values[VOLUME_FRACTION_NAME]
        return curve

    def phi_violation(self, values: ParameterValues) -> float:
        shape = self.model.shape
        if not shape.has_phi_limits() or VOLUME_FRACTION_NAME not in values:
            return 0.0
        with np.errstate(all="ignore"):
            lower, upper = shape.phi_limits(self.geometry, values)
        volume_fraction = values[VOLUME_FRACTION_NAME]
        violation = max(lower - volume_fraction, volume_fraction - upper, 0.0)
        return violation if math.isfinite(violation) else 0.0

    def is_feasible(self, values: ParameterValues) -> bool:
        with np.errstate(all="ignore"):
            return bool(self.model.shape.feasible(self.geometry, values))

    def __call__(self, vector: FloatArray) -> float:
        self.evaluation_count += 1
        values = self.free_spec.unpack(vector)
        if not self.is_feasible(values):
            return self.barrier_level + BARRIER_SLOPE * (1.0 + self.phi_violation(values))
        return self.loss(self.prediction(values))


class _LocalOptimum:
    """Лучшая точка локальной минимизации (общий вид для любого метода scipy)."""

    position: FloatArray
    loss: float
    success: bool
    message: str

    def __init__(self, position: FloatArray, loss: float, success: bool, message: str) -> None:
        self.position = position
        self.loss = loss
        self.success = success
        self.message = message


class _RegimeOutcome:
    regime: Regime | None
    free_spec: FreeSpec
    optimum: _LocalOptimum
    start_count: int
    evaluation_count: int  # вычислений целевой функции в этом режиме

    def __init__(
        self,
        regime: Regime | None,
        free_spec: FreeSpec,
        optimum: _LocalOptimum,
        start_count: int,
        evaluation_count: int,
    ) -> None:
        self.regime = regime
        self.free_spec = free_spec
        self.optimum = optimum
        self.start_count = start_count
        self.evaluation_count = evaluation_count


def _resolve_normalization(spec: FitSpec) -> Normalization:
    if spec.normalization is not Normalization.AUTO:
        return spec.normalization
    return Normalization.BULK if spec.density_bulk_per_nm3 is not None else Normalization.MEAN


def _window_mask(reduced: ReducedProfile, geometry: PoreGeometry, window: tuple[float, float] | None) -> BoolArray:
    if window is not None:
        lower, upper = window
    elif reduced.frame is Frame.PORE:
        lower, upper = geometry.coordinate_bounds()
    else:
        lower, upper = -math.inf, math.inf
    mask = (reduced.coordinate >= lower) & (reduced.coordinate <= upper)
    point_count = int(mask.sum())
    if point_count < MIN_WINDOW_POINTS:
        raise EmptyWindowError(
            f"в окне фита [{lower:.3g}, {upper:.3g}] {point_count} точек (нужно не меньше {MIN_WINDOW_POINTS}); "
            f"координата данных в долях размера поры: [{float(np.min(reduced.coordinate)):.3g}, "
            f"{float(np.max(reduced.coordinate)):.3g}] — проверь размер поры и систему отсчёта профиля"
        )
    return mask


def _check_profile_matches_pore(reduced: ReducedProfile, geometry: PoreGeometry) -> None:
    """Профиль должен соответствовать поре по размеру: иначе фит молча вернёт неверный угол."""
    coordinate, density = reduced.coordinate, reduced.density
    pore_lower, pore_upper = geometry.coordinate_bounds()
    data_range = f"[{float(np.min(coordinate)):.3g}, {float(np.max(coordinate)):.3g}]"
    if reduced.frame is Frame.PORE:
        if np.min(coordinate) > pore_lower + COVERAGE_TOLERANCE or np.max(coordinate) < pore_upper - COVERAGE_TOLERANCE:
            raise ProfileMismatchError(
                f"данные не покрывают пору: координата в долях размера поры {data_range}, стенки — "
                f"[{pore_lower:.3g}, {pore_upper:.3g}]; проверь размер поры и единицы профиля"
            )
        outside = (coordinate < pore_lower) | (coordinate > pore_upper)
        outside_fraction = float(np.sum(density[outside]) / np.sum(density))
        if outside_fraction > OUTSIDE_MASS_TOLERANCE:
            raise ProfileMismatchError(
                f"{outside_fraction:.0%} плотности за пределами поры (координата {data_range}); проверь размер поры, "
                f"единицы профиля и центрирование"
            )
        return
    occupied = coordinate[density > 0.0]
    occupied_width = float(np.max(occupied) - np.min(occupied))
    if occupied_width > (pore_upper - pore_lower) * (1.0 + COVERAGE_TOLERANCE):
        raise ProfileMismatchError(
            f"профиль шире поры: жидкость занимает {occupied_width:.3g} в долях размера поры при ширине поры "
            f"{pore_upper - pore_lower:.3g}; проверь размер поры и единицы профиля"
        )


def _regime_intervals(
    model: "InterfaceModel", free_spec: FreeSpec, requested: Regime | None
) -> list[tuple[Regime | None, FreeSpec]]:
    shape = model.shape
    if CONTACT_ANGLE_NAME not in model.parameters.names:
        if requested is not None:
            raise UnsupportedRegimeError(
                f"у формы {shape.name} нет контактного угла — режим смачивания {requested.value} не задаётся"
            )
        return [(None, free_spec)]
    if CONTACT_ANGLE_NAME not in free_spec.free_names:
        regime = shape.require_supported_contact_angle(math.degrees(free_spec.fixed_values[CONTACT_ANGLE_NAME]))
        if requested is not None and requested is not regime:
            raise UnsupportedRegimeError(
                f"форма {shape.name}: зафиксированный θ относится к режиму {regime.value}, а запрошен {requested.value}"
            )
        return [(regime, free_spec)]

    index = free_spec.free_names.index(CONTACT_ANGLE_NAME)
    lower_deg = math.degrees(free_spec.lower_bounds[index])
    upper_deg = math.degrees(free_spec.upper_bounds[index])
    intervals: list[tuple[Regime | None, FreeSpec]] = []
    for regime, (regime_lower, regime_upper) in shape.regimes.items():
        if requested is not None and regime is not requested:
            continue
        if regime_lower == REGIME_BOUNDARY_DEG:
            regime_lower += REGIME_GAP_DEG
        if regime_upper == REGIME_BOUNDARY_DEG:
            regime_upper -= REGIME_GAP_DEG
        lower = max(lower_deg, regime_lower)
        upper = min(upper_deg, regime_upper)
        if lower < upper:
            intervals.append(
                (regime, free_spec.with_free_bounds(CONTACT_ANGLE_NAME, math.radians(lower), math.radians(upper)))
            )
    if not intervals:
        declared = [regime.value for regime in shape.regimes]
        raise UnsupportedRegimeError(
            f"форма {shape.name}: границы θ [{lower_deg:.1f}°, {upper_deg:.1f}°] не пересекаются с объявленными "
            f"режимами {declared}"
        )
    return intervals


def _start_points(free_spec: FreeSpec, starts_per_parameter: int) -> list[FloatArray]:
    """Центры ячеек равномерной сетки по свободным параметрам + стартовая точка пользователя."""
    axes = [
        lower + (np.arange(starts_per_parameter) + 0.5) / starts_per_parameter * (upper - lower)
        for lower, upper in zip(free_spec.lower_bounds, free_spec.upper_bounds, strict=True)
    ]
    points = [np.array(point, dtype=np.float64) for point in itertools.product(*axes)]
    if free_spec.initial_values:
        midpoint = 0.5 * (free_spec.lower_bounds + free_spec.upper_bounds)
        user_start = np.array(
            [
                free_spec.initial_values.get(name, float(midpoint[index]))
                for index, name in enumerate(free_spec.free_names)
            ],
            dtype=np.float64,
        )
        points.insert(0, np.clip(user_start, free_spec.lower_bounds, free_spec.upper_bounds))
    return points


def _check_identifiable(objective: _Objective, free_spec: FreeSpec, start: FloatArray) -> None:
    reference = objective.prediction(free_spec.unpack(start))
    scale = max(1.0, float(np.max(np.abs(reference))))
    for index, name in enumerate(free_spec.free_names):
        width = free_spec.upper_bounds[index] - free_spec.lower_bounds[index]
        changes_profile = False
        for step, direction in itertools.product(
            (fraction * width for fraction in IDENTIFIABILITY_STEP_FRACTIONS), (1.0, -1.0)
        ):
            trial = start.copy()
            trial[index] = np.clip(
                start[index] + direction * step, free_spec.lower_bounds[index], free_spec.upper_bounds[index]
            )
            values = free_spec.unpack(trial)
            if not objective.is_feasible(values):
                continue
            change = float(np.max(np.abs(objective.prediction(values) - reference)))
            if change > IDENTIFIABILITY_TOLERANCE * scale:
                changes_profile = True
                break
        if not changes_profile:
            raise UnidentifiableParameterError(
                f"параметр {name} не влияет на профиль формы {objective.model.shape.name} "
                f"в системе отсчёта {objective.frame.value}: зафиксируй его"
            )


def _fit_in_regime(
    objective: _Objective, regime: Regime | None, free_spec: FreeSpec, spec: FitSpec, method: OptimizerMethod
) -> _RegimeOutcome:
    objective.free_spec = free_spec
    evaluations_before = objective.evaluation_count
    if not free_spec.free_names:
        empty = np.zeros(0, dtype=np.float64)
        optimum = _LocalOptimum(empty, objective(empty), True, "нет свободных параметров")
        return _RegimeOutcome(regime, free_spec, optimum, 1, objective.evaluation_count - evaluations_before)

    starts = [
        point
        for point in _start_points(free_spec, spec.starts_per_parameter)
        if objective.is_feasible(free_spec.unpack(point))
    ]
    if not starts:
        raise InfeasibleParametersError(
            f"форма {objective.model.shape.name}, режим {regime.value if regime else '—'}: ни одна стартовая точка "
            f"не попадает в область существования при {_describe_ranges(free_spec)}; "
            f"проверь фиксированные значения и границы"
        )
    _check_identifiable(objective, free_spec, starts[0])
    box: Bounds = Bounds(free_spec.lower_bounds, free_spec.upper_bounds)
    best: _LocalOptimum | None = None
    for start in starts:
        local = minimize(objective, start, method=method.value, bounds=box)
        if best is None or float(local.fun) < best.loss:
            best = _LocalOptimum(
                np.asarray(local.x, dtype=np.float64), float(local.fun), bool(local.success), str(local.message)
            )
    assert best is not None  # стартов не меньше одного
    return _RegimeOutcome(regime, free_spec, best, len(starts), objective.evaluation_count - evaluations_before)


def _describe_ranges(free_spec: FreeSpec) -> str:
    """Границы свободных и значения фиксированных параметров в единицах пользователя — для сообщений об ошибках."""
    parts = []
    for index, name in enumerate(free_spec.free_names):
        parameter = free_spec.parameters[name]
        lower = parameter.to_user(float(free_spec.lower_bounds[index]))
        upper = parameter.to_user(float(free_spec.upper_bounds[index]))
        parts.append(f"{name} ∈ [{lower:.4g}, {upper:.4g}]")
    for name, value in free_spec.fixed_values.items():
        parts.append(f"{name} = {free_spec.parameters[name].to_user(value):.4g}")
    return ", ".join(parts)


def _boundary_flags(outcome: _RegimeOutcome) -> tuple[tuple[str, ...], bool]:
    free_spec = outcome.free_spec
    at_bounds: list[str] = []
    at_regime_boundary = False
    regime_edges = (
        math.radians(REGIME_BOUNDARY_DEG - REGIME_GAP_DEG),
        math.radians(REGIME_BOUNDARY_DEG + REGIME_GAP_DEG),
    )
    for index, name in enumerate(free_spec.free_names):
        lower, upper = free_spec.lower_bounds[index], free_spec.upper_bounds[index]
        tolerance = BOUND_PROXIMITY_FRACTION * (upper - lower)
        value = outcome.optimum.position[index]
        for edge in (lower, upper):
            if abs(value - edge) <= tolerance:
                if name == CONTACT_ANGLE_NAME and any(math.isclose(edge, regime_edge) for regime_edge in regime_edges):
                    at_regime_boundary = True
                elif name not in at_bounds:
                    at_bounds.append(name)
    return tuple(at_bounds), at_regime_boundary


def _build_result(
    objective: _Objective,
    outcome: _RegimeOutcome,
    reduced: ReducedProfile,
    spec: FitSpec,
    volume_fraction_from_data: float | None,
    alternatives: tuple[FitResult, ...],
) -> FitResult:
    model = objective.model
    free_spec = outcome.free_spec
    values = free_spec.unpack(outcome.optimum.position)
    user_values = free_spec.to_user(values)
    user_values.update(
        {name: value for name, value in spec.fixed.items() if name in user_values}
    )  # как задал пользователь
    at_bounds, at_regime_boundary = _boundary_flags(outcome)
    wall_layer_nm = (
        user_values[WALL_LAYER_NAME] * model.pore.characteristic_size_nm() if WALL_LAYER_NAME in user_values else None
    )
    density_bulk = (
        reduced.density_scale_per_nm3 / values[VOLUME_FRACTION_NAME]
        if reduced.normalization is Normalization.MEAN and VOLUME_FRACTION_NAME in values
        else None
    )
    feasible = objective.is_feasible(values)
    curve = objective.prediction(values) if feasible else np.zeros_like(objective.data)
    return FitResult(
        pore_name=model.pore.name,
        shape_name=model.shape.name,
        regime=outcome.regime,
        frame=reduced.frame,
        normalization=reduced.normalization,
        values=user_values,
        free_parameters=free_spec.free_names,
        fixed_parameters=tuple(free_spec.fixed_values),
        loss=outcome.optimum.loss,
        success=outcome.optimum.success,
        message=outcome.optimum.message,
        start_count=outcome.start_count,
        evaluation_count=outcome.evaluation_count,
        feasible=feasible,
        at_bounds=at_bounds,
        at_regime_boundary=at_regime_boundary,
        wall_layer_thickness_nm=wall_layer_nm,
        density_bulk_derived_per_nm3=density_bulk,
        volume_fraction_from_data=volume_fraction_from_data,
        coordinate=objective.coordinate,
        data=objective.data,
        model_curve=curve,
        alternatives=alternatives,
        spec=spec,
    )


def _within_fit_bounds(model: "InterfaceModel", spec: FitSpec, name: str, user_value: float) -> bool:
    lower, upper = model.parameters[name].bounds
    if name in spec.bounds:
        lower, upper = max(lower, spec.bounds[name][0]), min(upper, spec.bounds[name][1])
    return lower <= user_value <= upper


def fit(model: "InterfaceModel", profile: DensityProfile, spec: FitSpec) -> FitResult:
    """Подобрать свободные параметры модели под профиль. Алгоритм — спецификация §4."""
    if not model.shape.has_profile():
        fittable = [other.shape.name for other in model.pore.available_interfaces() if other.shape.has_profile()]
        raise NotDerivedError(
            f"фит формы {model.shape.name} невозможен: профиль плотности не выведен (форма участвует только "
            f"в диаграмме устойчивости); формы поры {model.pore.name} с профилем: {', '.join(fittable)}"
        )
    geometry = model.pore.geometry()
    normalization = _resolve_normalization(spec)
    reduced = model.pore.reduce(profile, normalization, spec.density_bulk_per_nm3)
    window_mask = _window_mask(reduced, geometry, spec.window)
    _check_profile_matches_pore(reduced, geometry)
    optimize = spec.optimize if spec.optimize is not None else model.default_optimize
    has_volume_fraction = VOLUME_FRACTION_NAME in model.parameters.names
    if normalization is Normalization.MEAN and not has_volume_fraction:
        raise InvalidProfileError(
            f"нормировка mean делит модель на φ, а у формы {model.shape.name} параметра φ нет; "
            f"задай density_bulk_per_nm3"
        )

    values_from_data: dict[str, float] = {}
    initial = dict(spec.initial)
    volume_fraction_from_data: float | None = None
    if normalization is Normalization.BULK and has_volume_fraction and VOLUME_FRACTION_NAME not in spec.fixed:
        volume_fraction_from_data = float(trapezoid(reduced.density, reduced.coordinate) / geometry.profile_integral())
        if VOLUME_FRACTION_NAME not in optimize:
            values_from_data[VOLUME_FRACTION_NAME] = volume_fraction_from_data
        elif VOLUME_FRACTION_NAME not in initial and _within_fit_bounds(
            model, spec, VOLUME_FRACTION_NAME, volume_fraction_from_data
        ):
            # ∫ρ/ρ_bulk — точная φ для любой формы; старт из неё не даёт сползти на край плато (ступенчатый
            # профиль слоя делает ошибку кусочно-постоянной по φ)
            initial[VOLUME_FRACTION_NAME] = volume_fraction_from_data

    free_spec = model.parameters.resolve(optimize, spec.fixed, spec.bounds, initial, values_from_data)
    objective = _Objective(model, free_spec, reduced, window_mask, spec.loss)
    method = spec.optimizer

    outcomes: list[_RegimeOutcome] = []
    failures: list[InfeasibleParametersError] = []
    for regime, regime_spec in _regime_intervals(model, free_spec, spec.regime):
        try:
            outcomes.append(_fit_in_regime(objective, regime, regime_spec, spec, method))
        except InfeasibleParametersError as error:
            failures.append(error)
    if not outcomes:
        raise failures[0]
    outcomes.sort(key=lambda outcome: outcome.optimum.loss)

    alternatives = tuple(
        _build_result(objective, outcome, reduced, spec, volume_fraction_from_data, ()) for outcome in outcomes[1:]
    )
    return _build_result(objective, outcomes[0], reduced, spec, volume_fraction_from_data, alternatives)
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `328 passed, 34 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/core/fitting.py tests/unit/test_fitting_details.py
git commit -m "Фит: форма без профиля — NotDerivedError до оптимизации со списком форм с профилем" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 5: Диаграмма устойчивости

**Files:**
- Modify: `src/panda_md/core/shape.py` (`has_surface_terms`)
- Create: `src/panda_md/analysis/__init__.py`, `src/panda_md/analysis/stability.py`
- Create: `tests/unit/test_stability.py`

**Interfaces:**
- Consumes: `InterfaceModel.is_feasible`, `InterfaceModel.capillary_energy` (стадии 1–2), `Pore.own_registry`, поле поры `length_nm`.
- Produces: `InterfaceShape.has_surface_terms() -> bool`; `StabilityDiagram` (pydantic, frozen: `pore_name`, `contact_angle_deg`, `wall_layer`, `volume_fractions`, `length_ratios`, `shape_names`, `stable_shapes[i][j]` — имя формы или `None`); `stability_diagram(pore, *, contact_angle_deg, wall_layer, volume_fractions, length_ratios, shape_names=None) -> StabilityDiagram`. Кандидаты — формы с выведенными площадями; неизвестное или без площадей имя — `PandaError`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_stability.py`:

```python
"""Диаграмма устойчивости: в каждой точке (l, φ) — форма с наименьшим капиллярным функционалом."""

import math

import pytest

from panda_md import SlitPore, SquarePore, TrianglePore
from panda_md.analysis.stability import stability_diagram
from panda_md.core.errors import PandaError

SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)


def test_each_point_holds_the_feasible_shape_with_lowest_energy() -> None:
    diagram = stability_diagram(
        SQUARE, contact_angle_deg=150.0, wall_layer=0.02, volume_fractions=(0.05, 0.3, 0.6), length_ratios=(1.0, 4.0)
    )
    assert diagram.volume_fractions == (0.05, 0.3, 0.6)
    assert diagram.length_ratios == (1.0, 4.0)
    for length_index, length_ratio in enumerate(diagram.length_ratios):
        pore = SquarePore(side_nm=6.0, length_nm=length_ratio * 6.0, layer_width_nm=6.0)
        for phi_index, phi in enumerate(diagram.volume_fractions):
            energies: dict[str, float] = {}
            for model in pore.available_interfaces():
                values = {"theta": 150.0, "delta": 0.02, "phi": phi}
                values = {name: value for name, value in values.items() if name in model.parameters.names}
                if model.is_feasible(**values):
                    energies[model.shape.name] = model.capillary_energy(**values)
            expected = min(energies, key=energies.__getitem__) if energies else None
            assert diagram.stable_shapes[length_index][phi_index] == expected


def test_shapes_can_be_restricted_and_unknown_names_fail() -> None:
    diagram = stability_diagram(
        SQUARE,
        contact_angle_deg=150.0,
        wall_layer=0.0,
        volume_fractions=(0.1, 0.5),
        length_ratios=(2.0,),
        shape_names=("filament_w1c0", "droplet_w1c0"),
    )
    assert diagram.shape_names == ("filament_w1c0", "droplet_w1c0")
    assert {name for row in diagram.stable_shapes for name in row} <= {"filament_w1c0", "droplet_w1c0", None}
    with pytest.raises(PandaError, match="perforation"):
        stability_diagram(
            SQUARE,
            contact_angle_deg=150.0,
            wall_layer=0.0,
            volume_fractions=(0.1,),
            length_ratios=(2.0,),
            shape_names=("perforation",),
        )


def test_shapes_without_surface_terms_are_left_out() -> None:
    slit = SlitPore(width_nm=8.0, length_nm=16.0)
    diagram = stability_diagram(
        slit, contact_angle_deg=120.0, wall_layer=0.0, volume_fractions=(0.2,), length_ratios=(2.0,)
    )
    assert "doughnut" not in diagram.shape_names
    assert "perforation" not in diagram.shape_names
    assert "roll" in diagram.shape_names


def test_points_where_no_shape_exists_are_empty() -> None:
    triangle = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    unreachable_layer = 1.0 / (2.0 * math.sqrt(3.0))  # a′ = 0: доступного сечения нет
    diagram = stability_diagram(
        triangle, contact_angle_deg=150.0, wall_layer=unreachable_layer, volume_fractions=(0.1,), length_ratios=(2.0,)
    )
    assert diagram.stable_shapes == ((None,),)
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.analysis'`.

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/core/shape.py` целиком:

`src/panda_md/core/shape.py`:

```python
"""Контракт формы интерфейса: только формулы, без состояния, данных и оптимизатора."""

import math
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import numpy as np
from pydantic import Field
from scipy.integrate import trapezoid

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.errors import NotDerivedError, UnsupportedRegimeError
from panda_md.core.parameters import Parameter, ParameterValues
from panda_md.core.types import FloatArray, PandaModel

REGIME_BOUNDARY_DEG = 90.0  # граница смачивания и несмачивания
REGIME_GAP_DEG = 0.5  # θ ∈ (90° − зазор, 90° + зазор) не считается: в формулах tan θ и 1/cos θ
CENTER_OF_MASS_GRID_POINTS = 20001  # сетка для численного центра масс, если форма не дала формулу


class PoreGeometry(PandaModel, ABC):
    """Безразмерная геометрия поры, которую получают формулы формы."""

    @abstractmethod
    def coordinate_bounds(self) -> tuple[float, float]:
        """Безразмерные координаты стенок вдоль оси профиля в системе поры (при δ = 0)."""

    @abstractmethod
    def profile_integral(self) -> float:
        """∫ρ dz / φ для любой формы этой поры: площадь исходного сечения / ширина слоя нормировки."""


class SurfaceTerms(PandaModel):
    """Площади межфазных поверхностей формы (безразмерные)."""

    liquid_liquid_area: float = Field(ge=0.0)  # S_LV
    liquid_solid_area: float = Field(ge=0.0)  # S_SL

    def capillary_energy(self, contact_angle_rad: float) -> float:
        """Капиллярный функционал S_LV − cos θ · S_SL: по его минимуму выбирается устойчивая форма."""
        return self.liquid_liquid_area - math.cos(contact_angle_rad) * self.liquid_solid_area


def regime_of(contact_angle_deg: float) -> Regime:
    return Regime.WETTING if contact_angle_deg < REGIME_BOUNDARY_DEG else Regime.NON_WETTING


GeometryT = TypeVar("GeometryT", bound=PoreGeometry)


class InterfaceShape(ABC, Generic[GeometryT]):
    """Одна форма интерфейса в одном типе поры. Значения параметров — во внутренних единицах."""

    name: ClassVar[str]
    parameters: ClassVar[tuple[Parameter, ...]]
    regimes: ClassVar[dict[Regime, tuple[float, float]]]  # диапазон θ в градусах для каждого выведенного режима
    frame: ClassVar[Frame]  # система отсчёта, в которой записаны формулы
    default_optimize: ClassVar[tuple[str, ...]] = (
        ParameterName.CONTACT_ANGLE.value,
        ParameterName.WALL_LAYER_THICKNESS.value,
    )

    @abstractmethod
    def feasible(self, geometry: GeometryT, values: ParameterValues) -> bool:
        """Существует ли форма при этих параметрах."""

    def profile(self, coordinate: FloatArray, geometry: GeometryT, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk на безразмерной координате. Вызывается только для допустимых параметров."""
        raise NotDerivedError(f"{self.name}: профиль плотности не выведен")

    def phi_limits(self, geometry: GeometryT, values: ParameterValues) -> tuple[float, float]:
        """Допустимый интервал φ при остальных параметрах."""
        raise NotDerivedError(f"{self.name}: интервал φ не выведен")

    def surface_terms(self, geometry: GeometryT, values: ParameterValues) -> SurfaceTerms:
        raise NotDerivedError(f"{self.name}: площади поверхностей не выведены")

    def center_of_mass(self, geometry: GeometryT, values: ParameterValues) -> float:
        """Центр масс жидкости вдоль координаты профиля, в системе формул формы. По умолчанию — численно."""
        lower, upper = geometry.coordinate_bounds()
        grid = np.linspace(lower, upper, CENTER_OF_MASS_GRID_POINTS)
        density = self.profile(grid, geometry, values)
        return float(trapezoid(grid * density, grid) / trapezoid(density, grid))

    def has_phi_limits(self) -> bool:
        return type(self).phi_limits is not InterfaceShape.phi_limits

    def has_profile(self) -> bool:
        return type(self).profile is not InterfaceShape.profile

    def has_surface_terms(self) -> bool:
        return type(self).surface_terms is not InterfaceShape.surface_terms

    def require_supported_contact_angle(self, contact_angle_deg: float) -> Regime:
        """Режим угла, если для него у формы выведены формулы; иначе UnsupportedRegimeError."""
        if abs(contact_angle_deg - REGIME_BOUNDARY_DEG) < REGIME_GAP_DEG:
            raise UnsupportedRegimeError(
                f"θ = {contact_angle_deg:.2f}° внутри зазора ±{REGIME_GAP_DEG}° у 90°: формулы там не определены"
            )
        regime = regime_of(contact_angle_deg)
        if regime not in self.regimes:
            raise UnsupportedRegimeError(
                f"форма {self.name}: режим {regime.value} (θ = {contact_angle_deg:.4g}°) не объявлен — "
                f"формулы не выведены"
            )
        lower, upper = self.regimes[regime]
        if not lower <= contact_angle_deg <= upper:
            raise UnsupportedRegimeError(
                f"форма {self.name}: θ = {contact_angle_deg:.4g}° вне диапазона режима {regime.value} "
                f"[{lower}, {upper}]"
            )
        return regime
```

`src/panda_md/analysis/__init__.py`:

```python
"""Надстройки над моделью: диаграммы устойчивости (дальше — блоки, перебор форм, сравнение)."""
```

`src/panda_md/analysis/stability.py`:

```python
"""Диаграмма устойчивости: какая форма интерфейса выгоднее при данных θ, δ на сетке (l, φ).

Устойчивой считается допустимая форма с наименьшим капиллярным функционалом S_LV − cos θ · S_SL
(у формы без θ — S_LV). Так же строит диаграмму PANDA_Polygon (`docs/model-runtime.js`, computePhase).
"""

from collections.abc import Sequence

from pydantic import ConfigDict

from panda_md.core.enums import ParameterName
from panda_md.core.errors import PandaError
from panda_md.core.model import InterfaceModel
from panda_md.core.pore import Pore
from panda_md.core.types import PandaModel

LENGTH_FIELD = "length_nm"  # поле поры, которое меняется вдоль оси l диаграммы


class StabilityDiagram(PandaModel):
    """Результат: stable_shapes[i][j] — устойчивая форма при length_ratios[i], volume_fractions[j] (None — ни одной)."""

    model_config = ConfigDict(frozen=True)

    pore_name: str
    contact_angle_deg: float
    wall_layer: float
    volume_fractions: tuple[float, ...]
    length_ratios: tuple[float, ...]  # l = L / (характерный размер поры)
    shape_names: tuple[str, ...]  # формы-кандидаты: у них выведены площади поверхностей
    stable_shapes: tuple[tuple[str | None, ...], ...]


def stability_diagram(
    pore: Pore,
    *,
    contact_angle_deg: float,
    wall_layer: float,
    volume_fractions: Sequence[float],
    length_ratios: Sequence[float],
    shape_names: Sequence[str] | None = None,
) -> StabilityDiagram:
    """Построить диаграмму для поры `pore`; длина поры берётся из `length_ratios`, остальные размеры — из `pore`."""
    if LENGTH_FIELD not in type(pore).model_fields:
        raise PandaError(f"у поры {pore.name} нет поля {LENGTH_FIELD}: диаграмму по длине не построить")
    candidates = _candidate_names(pore, shape_names)
    size_nm = pore.characteristic_size_nm()
    rows: list[tuple[str | None, ...]] = []
    for length_ratio in length_ratios:
        sized_pore = pore.model_copy(update={LENGTH_FIELD: length_ratio * size_nm})
        models = [sized_pore.interface(name) for name in candidates]
        rows.append(
            tuple(
                _stable_shape(models, contact_angle_deg, wall_layer, volume_fraction)
                for volume_fraction in volume_fractions
            )
        )
    return StabilityDiagram(
        pore_name=pore.name,
        contact_angle_deg=contact_angle_deg,
        wall_layer=wall_layer,
        volume_fractions=tuple(volume_fractions),
        length_ratios=tuple(length_ratios),
        shape_names=candidates,
        stable_shapes=tuple(rows),
    )


def _candidate_names(pore: Pore, shape_names: Sequence[str] | None) -> tuple[str, ...]:
    registered = pore.own_registry()
    with_energy = tuple(name for name, shape_class in registered.items() if shape_class().has_surface_terms())
    if shape_names is None:
        return with_energy
    unusable = [name for name in shape_names if name not in with_energy]
    if unusable:
        raise PandaError(
            f"в поре {pore.name} нет форм с выведенными площадями поверхностей: {unusable}; "
            f"доступны: {list(with_energy)}"
        )
    return tuple(shape_names)


def _stable_shape(
    models: Sequence[InterfaceModel], contact_angle_deg: float, wall_layer: float, volume_fraction: float
) -> str | None:
    """Допустимая форма с наименьшей энергией; при равенстве — первая в списке кандидатов."""
    point = {
        ParameterName.CONTACT_ANGLE.value: contact_angle_deg,
        ParameterName.WALL_LAYER_THICKNESS.value: wall_layer,
        ParameterName.VOLUME_FRACTION.value: volume_fraction,
    }
    best_name: str | None = None
    best_energy = float("inf")
    for model in models:
        values = {name: value for name, value in point.items() if name in model.parameters.names}
        if not model.is_feasible(**values):
            continue
        energy = model.capillary_energy(**values)
        if energy < best_energy:
            best_name, best_energy = model.shape.name, energy
    return best_name
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `332 passed, 34 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/core/shape.py src/panda_md/analysis tests/unit/test_stability.py
git commit -m "Диаграмма устойчивости: в каждой точке (l, φ) — форма с наименьшим капиллярным функционалом" \
  -m "Общий код для всех пор; формы без выведенных площадей (пончик и перфорация щели) не участвуют." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 6: Регрессия на PANDA_Polygon — формы и диаграммы

**Files:**
- Create: `tests/regression/generate_polygon_reference.py`
- Delete: `tests/regression/generate_square_reference.py` (заменён общим генератором)
- Modify: `tests/regression/test_reference_formulas.py`
- Create: `tests/regression/data/triangle_reference.json`; Modify: `tests/regression/data/square_reference.json` (оба — вывод генератора)

**Interfaces:**
- Consumes: `TrianglePore`, `SquarePore`, `PolygonPore` (задачи 1–3), `stability_diagram` (задача 5).
- Produces: эталоны `{triangle,square}_reference.json` со структурой `{"source", "coordinate", "layer_width_ratio", "shapes": {имя: [случаи]}, "stability": {"volume_fractions", "length_ratios", "shape_order", "cases"}}`; случай формы — `length_ratio`, `delta`, `phi`, `theta` (если есть), `feasible`, при допустимости — `energy`, `area_sum`, `profile` (если выведен).

- [ ] **Шаг 1: Тест и генератор**

Заменить `tests/regression/test_reference_formulas.py` целиком:

`tests/regression/test_reference_formulas.py`:

```python
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
```

`tests/regression/generate_polygon_reference.py`:

```python
"""Эталоны треугольной и квадратной пор из PANDA_Polygon (`docs/model/{triangle,square}.py`, коммит 38ff491).

Запуск из корня репозитория (нужна локальная копия PANDA_Polygon/, см. ROADMAP «Фаза 4а»):
    uv run python tests/regression/generate_polygon_reference.py
Результат: tests/regression/data/{triangle,square}_reference.json — коммитятся, тесты читают только их.

Для каждой формы записываются условие существования, энергия S_LV − cos θ · S_SL и сумма площадей S_LV + S_SL;
профиль — только у форм, для которых он выведен в источнике. Девять форм квадрата с дефектами источника
(docs/architecture/05_polygon_pores.md §4) не переносятся и в эталон не входят.
"""

import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
from pydantic import BaseModel, ConfigDict

SOURCE_DIRECTORY = Path(__file__).parents[2] / "PANDA_Polygon" / "docs" / "model"
OUTPUT_DIRECTORY = Path(__file__).parent / "data"
SOURCE_COMMIT = "github.com/Vandyshge/PANDA_Polygon@38ff491"
BIN_COUNT = 200  # координата — центры бинов: ни одна точка не лежит на стенке
LAYER_WIDTH_RATIO = 1.0
LENGTH_RATIOS = (1.0, 2.0, 4.0, 8.0)
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9)
UNUSED_ANGLE_RAD = math.pi  # форме без θ источник всё равно принимает угол, но не использует его
FIXED_VOLUME_SCALES = (1.0, 1.01, 0.98)  # droplet_w3c0: φ ровно по объёму, внутри и вне допуска 1.5 %
STABILITY_POINTS = 30  # сетка диаграммы устойчивости: φ и l — как в computePhase (docs/model-runtime.js)
STABILITY_MAX_LENGTH_RATIO = 5.0
STABILITY_WALL_LAYERS = (0.0, 0.03)


class PolygonReferenceSetup(BaseModel):
    """Что и на какой сетке снимать с одного модуля источника."""

    model_config = ConfigDict(frozen=True)

    pore_name: str
    coordinate_bounds: tuple[float, float]
    angles_deg: tuple[float, ...]
    stability_angles_deg: tuple[float, ...]
    shapes: tuple[str, ...]
    shapes_without_angle: tuple[str, ...]


TRIANGLE_HEIGHT = math.sqrt(3.0) / 2.0
SETUPS = (
    PolygonReferenceSetup(
        pore_name="triangle",
        coordinate_bounds=(-TRIANGLE_HEIGHT / 3.0, 2.0 * TRIANGLE_HEIGHT / 3.0),
        angles_deg=(60.0, 100.0, 110.0, 125.0, 150.0, 175.0),
        # 112.6° < θ < 120° не берём: там у droplet_w2c1 источник даёт S_LV < 0 (05_polygon_pores.md §4)
        stability_angles_deg=(100.0, 110.0, 125.0, 150.0, 180.0),
        shapes=(
            "filament_w0c0",
            "filament_w0c0_straight",
            "filament_w1c0",
            "filament_w2c0",
            "filament_w2c1",
            "filament_w3c0",
            "filament_w3c1",
            "filament_w3c2",
            "droplet_w1c0",
            "droplet_w2c0",
            "droplet_w2c1",
            "droplet_w3c0",
            "capsule_w3c0",
            "capsule_w3c3",
        ),
        shapes_without_angle=("filament_w0c0", "filament_w0c0_straight"),
    ),
    PolygonReferenceSetup(
        pore_name="square",
        coordinate_bounds=(-0.5, 0.5),
        # 135° не берём: у droplet_w2c1 источник допускает θ = 3π/4, но площадь там бесконечна (05_polygon_pores.md)
        angles_deg=(60.0, 100.0, 120.0, 140.0, 150.0, 175.0),
        stability_angles_deg=(100.0, 120.0, 140.0, 160.0, 180.0),
        shapes=(
            "filament_w0c0",
            "filament_w0c0_straight",
            "filament_w1c0",
            "droplet_w1c0",
            "droplet_w2c0",
            "droplet_w2c1",
            "capsule_w4c0",
        ),
        shapes_without_angle=("filament_w0c0", "filament_w0c0_straight"),
    ),
)


def load_source(pore_name: str) -> ModuleType:
    path = SOURCE_DIRECTORY / f"{pore_name}.py"
    spec = importlib.util.spec_from_file_location(f"panda_polygon_{pore_name}", path)
    assert spec is not None and spec.loader is not None, f"нет файла {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bin_centers(bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    width = (upper - lower) / BIN_COUNT
    return np.linspace(lower + width / 2.0, upper - width / 2.0, BIN_COUNT)


def fixed_volume_fractions(length_ratio: float, angle: float, wall_layer: float) -> list[float]:
    """φ, при которых droplet_w3c0 треугольника существует: объём капли задан сечением (cond_droplet_w3c0)."""
    side = 1.0 - 2.0 * math.sqrt(3.0) * wall_layer
    cosine = math.cos(angle)
    if cosine >= 0.0:
        return []
    radius = -side / (2.0 * math.sqrt(3.0) * cosine)
    volume = math.pi * radius**3 / 3.0 * (3.0 * cosine**3 - 9.0 * cosine - 2.0)
    exact = volume / (math.sqrt(3.0) / 4.0 * length_ratio)
    return [exact * scale for scale in FIXED_VOLUME_SCALES if 0.0 < exact * scale < 1.0]


def shape_cases(source: ModuleType, setup: PolygonReferenceSetup, shape_name: str) -> list[dict[str, object]]:
    suffix = setup.pore_name.upper()
    condition = getattr(source, f"COND_{suffix}")[shape_name]
    area = getattr(source, f"AREA_{suffix}")[shape_name]
    density = getattr(source, f"RHO_{suffix}").get(shape_name)
    coordinate = bin_centers(setup.coordinate_bounds)
    without_angle = shape_name in setup.shapes_without_angle
    angles_deg: tuple[float | None, ...] = (None,) if without_angle else setup.angles_deg
    cases: list[dict[str, object]] = []
    for length_ratio, angle_deg, wall_layer in itertools.product(LENGTH_RATIOS, angles_deg, WALL_LAYERS):
        angle = UNUSED_ANGLE_RAD if angle_deg is None else math.radians(angle_deg)
        fractions = list(VOLUME_FRACTIONS)
        if shape_name == "droplet_w3c0":
            fractions += fixed_volume_fractions(length_ratio, angle, wall_layer)
        for phi in fractions:
            feasible = bool(condition(length_ratio, phi, angle, wall_layer))
            case: dict[str, object] = {"length_ratio": length_ratio, "delta": wall_layer, "phi": phi}
            if angle_deg is not None:
                case["theta"] = angle_deg
            case["feasible"] = feasible
            if feasible:
                case["energy"] = float(area(length_ratio, phi, angle, wall_layer, flag=False))
                case["area_sum"] = float(area(length_ratio, phi, angle, wall_layer, flag=True))
                if density is not None:
                    profile = density(coordinate, length_ratio, LAYER_WIDTH_RATIO, phi, angle, wall_layer)
                    case["profile"] = np.asarray(profile, dtype=float).tolist()
            cases.append(case)
    return cases


def stability_grids() -> tuple[np.ndarray, np.ndarray]:
    """(φ, l) — те же формулы сеток, что в computePhase."""
    return np.linspace(0.001, 1.0, STABILITY_POINTS), np.linspace(0.001, STABILITY_MAX_LENGTH_RATIO, STABILITY_POINTS)


def stability_cases(source: ModuleType, setup: PolygonReferenceSetup) -> list[dict[str, object]]:
    """Диаграммы устойчивости источника: argmin энергии по формам setup.shapes (порядок важен при равенстве)."""
    registry = getattr(source, f"AREA_{setup.pore_name.upper()}")
    volume_fractions, length_ratios = stability_grids()
    phi_mesh, length_mesh = np.meshgrid(volume_fractions, length_ratios)
    cases: list[dict[str, object]] = []
    for angle_deg, wall_layer in itertools.product(setup.stability_angles_deg, STABILITY_WALL_LAYERS):
        angle = math.radians(angle_deg)
        energies = []
        for shape_name in setup.shapes:
            energy = np.asarray(registry[shape_name](length_mesh, phi_mesh, angle, wall_layer, False), dtype=float)
            energy[~np.isfinite(energy)] = np.inf
            energies.append(energy)
        stacked = np.stack(energies)
        valid = np.isfinite(stacked).any(axis=0)
        winner = np.argmin(stacked, axis=0)
        stable = [
            [setup.shapes[winner[i, j]] if valid[i, j] else None for j in range(len(volume_fractions))]
            for i in range(len(length_ratios))
        ]
        cases.append({"theta": angle_deg, "delta": wall_layer, "stable_shapes": stable})
    return cases


def main() -> None:
    for setup in SETUPS:
        source = load_source(setup.pore_name)
        reference: dict[str, object] = {
            "source": f"{SOURCE_COMMIT} docs/model/{setup.pore_name}.py",
            "coordinate": bin_centers(setup.coordinate_bounds).tolist(),
            "layer_width_ratio": LAYER_WIDTH_RATIO,
            "shapes": {name: shape_cases(source, setup, name) for name in setup.shapes},
            "stability": {
                "volume_fractions": stability_grids()[0].tolist(),
                "length_ratios": stability_grids()[1].tolist(),
                "shape_order": list(setup.shapes),
                "cases": stability_cases(source, setup),
            },
        }
        output = OUTPUT_DIRECTORY / f"{setup.pore_name}_reference.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reference))
        print(f"записано в {output}: {len(setup.shapes)} форм")


if __name__ == "__main__":
    main()
```

```bash
git rm tests/regression/generate_square_reference.py
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `FileNotFoundError: ... tests/regression/data/triangle_reference.json`.

- [ ] **Шаг 3: Эталоны**

Run: `uv run python tests/regression/generate_polygon_reference.py`
Expected: `записано в .../triangle_reference.json: 14 форм` и `записано в .../square_reference.json: 7 форм` (предупреждения numpy об `invalid value` из источника — ожидаемы).

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `353 passed, 34 skipped`; 21 случай `test_polygon_matches_panda_polygon` и 2 `test_polygon_stability_diagram_matches_panda_polygon`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add tests/regression
git commit -m "Регрессия: 14 форм треугольника и 7 форм квадрата совпадают с PANDA_Polygon до 1e-12, диаграммы устойчивости — во всех точках" \
  -m "Эталон диаграмм — та же процедура, что computePhase в PANDA_Polygon; диапазон 112.6° < θ < 120° не используется (дефект droplet_w2c1 треугольника в источнике)." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 7: Синтетика многоугольных пор, описание тестов, документация стадии

Проверочная задача для синтетики (код написан в задачах 1–3).

**Files:**
- Modify: `tests/synthetic/test_recovery.py`, `tests/README.md`, `docs/architecture/05_polygon_pores.md`, `docs/superpowers/specs/2026-10-01-panda-architecture-design.md`, `ROADMAP.md`, `README.md`

- [ ] **Шаг 1: Синтетика**

Заменить `tests/synthetic/test_recovery.py` целиком (добавлены капля, нить w3c0 и капсула треугольника; капли в углу и капсула квадрата):

`tests/synthetic/test_recovery.py`:

```python
"""Синтетическая валидация: форма → профиль с шумом → фит → восстановленные параметры.

Пороги и найденные закономерности описаны в tests/README.md («Синтетика», «O2», «O3»).
"""

import numpy as np
import pytest

from panda_md import Frame, SlitPore, SquarePore, TrianglePore
from panda_md.core.model import InterfaceModel
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
NOISE_STD = 0.005  # шум 0.5 % ρ_bulk на бин, 200 бинов
SEEDS = range(5)
MAX_ANGLE_ERROR_DEG = 1.5  # в каждом прогоне
MEAN_ANGLE_ERROR_DEG = 1.0  # в среднем по прогонам (критерий ТЗ |Δθ| < 1°)

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)
SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
TRIANGLE = TrianglePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
RECTANGULAR_SLIT = SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.4)
CASES: list[tuple[InterfaceModel, dict[str, float]]] = [
    (SLIT.interface("roll"), {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
    (SLIT.interface("droplet"), {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
    (SQUARE.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    (SQUARE.interface("filament_w1c0"), {"theta": 130.0, "delta": 0.02, "phi": 0.1}),
    (SQUARE.interface("filament_w1c0"), {"theta": 60.0, "delta": 0.02, "phi": 0.05}),
    (SLIT.interface("doughnut"), {"theta": 130.0, "delta": 0.03, "phi": 0.3372}),
    (SLIT.interface("worm"), {"theta": 130.0, "delta": 0.03, "phi": 0.226}),
    (SLIT.interface("perforation"), {"theta": 130.0, "delta": 0.03, "phi": 0.6493}),
    (RECTANGULAR_SLIT.interface("droplet"), {"theta": 130.0, "delta": 0.03, "phi": 0.0561}),
    (RECTANGULAR_SLIT.interface("roll"), {"theta": 130.0, "delta": 0.03, "phi": 0.4848}),
    (TRIANGLE.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    (TRIANGLE.interface("filament_w3c0"), {"theta": 150.0, "delta": 0.02, "phi": 0.5}),
    (TRIANGLE.interface("capsule_w3c0"), {"theta": 150.0, "delta": 0.02, "phi": 0.4}),
    (SQUARE.interface("droplet_w2c0"), {"theta": 150.0, "delta": 0.02, "phi": 0.05}),
    (SQUARE.interface("droplet_w2c1"), {"theta": 115.0, "delta": 0.02, "phi": 0.1}),
    (SQUARE.interface("capsule_w4c0"), {"theta": 160.0, "delta": 0.02, "phi": 0.4}),
]


def case_id(case: tuple[InterfaceModel, dict[str, float]]) -> str:
    model, truth = case
    aspect_ratio = getattr(model.pore, "base_aspect_ratio", 1.0)
    suffix = "" if aspect_ratio == 1.0 else f"(k={aspect_ratio})"
    return f"{model.pore.name}{suffix}/{model.shape.name}/θ={truth['theta']:.0f}"


def angle_errors(model: InterfaceModel, truth: dict[str, float], density_bulk_per_nm3: float) -> list[float]:
    errors = []
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, density_bulk_per_nm3=density_bulk_per_nm3)
        errors.append(abs(result.values["theta"] - truth["theta"]))
    return errors


def assert_angle_recovered(errors: list[float]) -> None:
    assert max(errors) < MAX_ANGLE_ERROR_DEG, errors
    assert float(np.mean(errors)) < MEAN_ANGLE_ERROR_DEG, errors


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_angle_and_wall_layer_recovered_with_bulk_density(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    model, truth = case
    assert_angle_recovered(angle_errors(model, truth, DENSITY_BULK))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("optimize", "fixed"),
    [(["phi"], {"theta": 120.0, "delta": 0.03}), (["delta", "phi"], {"theta": 120.0}), (["theta", "delta", "phi"], {})],
    ids=["phi", "delta-phi", "theta-delta-phi"],
)
def test_task_modes_recover_volume_fraction(optimize: list[str], fixed: dict[str, float]) -> None:
    model, truth = CASES[0]
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, optimize=optimize, fixed=fixed, density_bulk_per_nm3=DENSITY_BULK)
        assert result.values["phi"] == pytest.approx(truth["phi"], abs=0.01)
        assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_mean_normalization_recovers_compact_shapes(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    """O2: без ρ_bulk φ и θ определяются вместе — для капель и нитей (у валика нет, см. README)."""
    model, truth = case
    for seed in SEEDS:
        profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, noise_std=NOISE_STD, seed=seed)
        result = model.fit(profile, optimize=["theta", "delta", "phi"])
        assert result.values["phi"] == pytest.approx(truth["phi"], abs=0.01)
        assert result.values["theta"] == pytest.approx(truth["theta"], abs=3.0)
        assert result.density_bulk_derived_per_nm3 == pytest.approx(DENSITY_BULK, rel=0.15)


def test_mean_normalization_with_known_phi_recovers_roll() -> None:
    """O2: валику в нормировке mean нужна известная φ (из построения системы)."""
    model, truth = CASES[0]
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK)
    result = model.fit(profile, fixed={"phi": truth["phi"]})
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.5)
    assert result.density_bulk_derived_per_nm3 == pytest.approx(DENSITY_BULK, rel=1e-3)


WALL_OFFSET = 0.05  # ошибка положения стенок: 5 % размера поры


@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_wall_offset_goes_into_wall_layer_for_one_wall_shapes(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    """O3: у форм на одной стенке сдвиг стенок поглощает δ, θ не меняется — центрирование по поре."""
    model, truth = case
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, offset=WALL_OFFSET)
    result = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.1)


def test_wall_offset_needs_free_shift_for_roll() -> None:
    """O3: валик касается обеих стенок — сдвиг поглощает только параметр shift."""
    model, truth = CASES[0]
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, offset=WALL_OFFSET)
    with_shift = model.fit(profile, optimize=["theta", "delta", "shift"], density_bulk_per_nm3=DENSITY_BULK)
    without_shift = model.fit(profile, density_bulk_per_nm3=DENSITY_BULK)
    assert with_shift.values["theta"] == pytest.approx(truth["theta"], abs=0.5)
    assert abs(without_shift.values["theta"] - truth["theta"]) > 2.0


@pytest.mark.parametrize("case", CASES[1:4], ids=case_id)
def test_liquid_frame_recovers_angle_with_known_wall_layer(case: tuple[InterfaceModel, dict[str, float]]) -> None:
    model, truth = case
    profile = synthetic_profile(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame=Frame.LIQUID)
    result = model.fit(profile, optimize=["theta"], fixed={"delta": truth["delta"]}, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=0.1)
```

Run: `uv run pytest -q tests/synthetic`
Expected: `30 passed`. Пороги не ослаблять: падение — расхождение с `tests/README.md`, разбирать.

- [ ] **Шаг 2: Описание тестов**

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
| регрессия | `regression/test_reference_formulas.py` | математика | перенос формул без изменений: шесть форм щели (k = 1), 14 форм треугольника, 7 форм квадрата; диаграммы устойчивости треугольника и квадрата | эталоны из старой PANDA (`alpha.py`, 93c6dbb) и PANDA_Polygon (`triangle.py`, `square.py`, 38ff491); генераторы `regression/generate_*.py`; диаграммы — 10 пар (θ, δ) на пору, сетка 30 × 30 по (l, φ), как в `computePhase` PANDA_Polygon | профиль (где выведен), интервал φ, условие существования, энергия и сумма площадей совпадают; раскраска диаграмм совпадает во всех точках | 1e-12 |
| фит | `unit/test_fitting.py` | оптимизация | один оптимизатор при 1, 2, 3 свободных параметрах; фиксированные не меняются; флаги «у границы» и «у 90°»; пустое окно, профиль не соответствует поре (пора шире данных, профиль в других единицах, профиль в системе жидкости шире поры), фиксированное значение или φ из данных вне границ, нет φ, нет допустимого старта, не объявленный режим, θ в зазоре у 90°, параметр без влияния — ошибки; деление по режимам смачивания | синтетика без шума, 200 бинов | восстановление в пределах дискретизации | θ — 0.5°, δ — 0.0025 (полбина), φ — 0.003 |
| детали фита | `unit/test_fitting_details.py` | оптимизация | барьер выше любой допустимой ошибки и растёт с выходом φ за предел; счётчик вычислений у каждого режима свой; ошибка «нет старта» перечисляет диапазоны; `regime=` у формы без θ — ошибка; φ слоя без систематического сдвига; нормировка mean у формы без φ — ошибка; фит формы без профиля — `NotDerivedError` до оптимизации со списком форм поры, у которых профиль есть | валик, нить квадрата, слой на 35 значениях φ, капли треугольника, форма из теста | точные значения; φ слоя в пределах бина, среднее смещение < 0.0025 | бин 0.005 |
| защита модели | `unit/test_model_guards.py` | математика | формулы вне объявленного режима (θ < 90° у щели, зазор у 90°) не вызываются — ошибка, `is_feasible` = False (раньше капля при θ = 0 падала с ZeroDivisionError); синтетика вне режима — ошибка; неизвестная форма — `UnknownShapeError` (и `KeyError`); энергия формы без θ, касающейся стенок, — ошибка; слой при φ = 1 − 2δ не существует | щель 8 × 16 нм, формы из теста | ожидаемые исключения | — |
| отступления многоугольников | `unit/test_polygon_shapes.py` | физическая модель | где формулы PANDA_Polygon нефизичны, форма не существует: droplet_w2c1 треугольника при θ > 2 arctg(3/2) ≈ 112.6° (там S_LV < 0); граница droplet_w2c1 квадрата строгая (при θ = 3π/4 площадь бесконечна) | треугольник и квадрат 1 нм | `is_feasible` = False | — |
| устойчивость | `unit/test_stability.py` | математика | в каждой точке (l, φ) — допустимая форма с наименьшим S_LV − cos θ · S_SL; ограничение списка форм; формы без площадей (пончик, перфорация щели) не участвуют; нет допустимых форм — пусто | квадрат, щель, треугольник при a′ = 0 | совпадение с перебором | точное равенство |
| неизменяемость | `unit/test_immutability.py` | данные | словари в `FitResult` и `FitSpec` нельзя изменить; одинаковые `FitSpec` равны и хешируются; JSON туда и обратно | фит валика | `TypeError` при записи; равенство | — |
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

**Перенос.** 14 форм треугольника и 7 форм квадрата (девять форм квадрата с дефектами источника не перенесены,
`docs/architecture/05_polygon_pores.md` §4) совпадают с PANDA_Polygon до 1e-12; диаграммы устойчивости совпадают
во всех точках. У capsule_w3c0 треугольника длина вставки в источнике записана тремя разными выражениями
(условие, площади, профиль) — они тождественны, оставлено одно.

**Новые дефекты источника** (вопросы автору, `05_polygon_pores.md` §4):
- droplet_w2c1 треугольника при 112.6° < θ < 120°: S_LV = 2π R² (1 + cos θ − ⅔ sin θ) < 0, а условие существования
  истинно. Здесь форма там не существует; в эталоне диаграмм этот диапазон θ не используется.
- capsule_w3c3 треугольника при δ > 0: длина вставки L = φ l − 4 V_торцов / √3 и площадь контакта 3 L записаны
  для a′ = 1 (сечение вставки √3/4 вместо √3 a′²/4). Перенесено как в источнике; при δ = 0 верно.

**Синтетика новых форм** (шум 0.5 %, 5 зёрен, свободны θ и δ): капля, нить w3c0 и капсула треугольника — |Δθ| до 1.4°
(среднее до 0.6°); капли в углу и капсула квадрата — до 1°. Капсулы восстанавливаются лучше всего (< 0.35°).

**Локальный минимум (открытый вопрос O11).** droplet_w2c0 квадрата при θ = 160°: в одном зерне из пяти фит
останавливается на θ = 168.9°, δ = 0.0075 — ошибка на 4 % выше, чем в истинной точке, но ни 9 стартов на параметр,
ни Powell оттуда не выводят: δ остаётся на стартовом значении, потому что L1-ошибка по δ ступенчатая (δ двигает
край профиля по бинам). При θ = 150° этого нет. Лечение — уточнение δ по сетке бинов после локального поиска
(фаза 5, стратегия фита).
````

- [ ] **Шаг 3: Разбор источника** (`docs/architecture/05_polygon_pores.md`, точечные замены через Edit)

В §4 после пункта 6 добавить:

```markdown
7. Треугольник, `droplet_w2c1`: при 2 arctg(3/2) ≈ 112.6° < θ < 120° площадь S_LV = 2π R² (1 + cos θ − ⅔ sin θ)
   отрицательна, а `cond_droplet_w2c1_triangle` форму допускает. В `panda-md` форма там не существует
   (`tests/unit/test_polygon_shapes.py`).
8. Треугольник, `capsule_w3c3` при δ > 0: длина вставки `L_ins = φ l − 4 V_ends / √3` и площадь контакта `3 L_ins`
   записаны для a′ = 1 (сечение вставки √3/4 вместо √3 a′²/4, периметр 3 вместо 3a′). Перенесено как есть.
9. Треугольник, `capsule_w3c0`: длина вставки записана трижды разными выражениями (`cond_*`, `area_*`,
   `d_capsule_w3c0_triangle`) — численно они тождественны; в `panda-md` оставлено одно.
10. Квадрат, `droplet_w2c1`: условие допускает θ = 3π/4, а площадь там бесконечна. В `panda-md` граница строгая.
```

В §5 две строки

```markdown
- Формы с найденными расхождениями переносятся, но помечаются «не проверено» и не попадают в `fit_all`, пока
  автор не подтвердит исправление; контрактные тесты для них — с маркером `xfail` и ссылкой на пункт выше.
```

заменить на:

```markdown
- Решение пользователя (15.10.2026): девять форм квадрата с дефектами (пункты 1–3) пока **не переносятся**;
  формы, где источник нефизичен лишь в части области (пункты 7, 10), переносятся с отступлением и тестом;
  пункт 8 перенесён как есть. Всё перечисленное — вопросы автору.
```

- [ ] **Шаг 4: Спецификация** (точечные замены через Edit)

В таблице решений после строки D19 добавить:

```markdown
| D20 | Многоугольные поры: общая база `PolygonGeometry`/`PolygonPore` (a, L, ly); профиль и площади у формы необязательны — форма без профиля только в диаграмме устойчивости, `fit` для неё — `NotDerivedError` до оптимизации; диаграмма устойчивости — `analysis.stability`, общий код для всех пор (стадия 3) |
```

§11, строку O8 заменить на:

```markdown
| O8 | Расхождения в PANDA_Polygon (`05_polygon_pores.md` §4): 7 условий существования квадрата падают; `perforation` при δ > 0 и `capsule_w4c4` не сходятся по ∫ρ = φ (эти 9 форм не перенесены); у 10 форм треугольника нет профиля; `droplet_w2c1` треугольника даёт S_LV < 0 при θ > 112.6°; `capsule_w3c3` неверна при δ > 0 | сообщить автору; после исправления — поднять коммит эталона и перенести недостающие формы |
```

После строки O10 добавить:

```markdown
| O11 | Локальный минимум фита по δ: L1-ошибка ступенчатая по δ, и Nelder–Mead не уходит со стартового δ (droplet_w2c0 квадрата при θ = 160°: 1 зерно из 5, θ 168.9° вместо 160°); больше стартов и Powell не помогают | фаза 5: уточнение δ по сетке бинов после локального поиска |
```

- [ ] **Шаг 5: ROADMAP и README**

`ROADMAP.md`, «Фаза 4а. Треугольная и квадратная поры»: первой строкой списка (перед строкой `- \`TrianglePore(a, L, ly)\`...`) добавить

```markdown
- **Статус: выполнено** — план `docs/superpowers/plans/2026-10-16-stage3-polygon-pores.md`; 359 тестов; 14 форм треугольника и 7 форм квадрата (9 форм с дефектами источника не перенесены, решение 15.10.2026); выводы — `tests/README.md` (стадия 3).
```

В том же разделе три строки

```markdown
- Расхождения в исходных моделях (сообщить автору, до исправления — «не проверено», тесты `xfail`):
  7 условий существования квадрата падают (`del l`); квадрат `perforation` при δ > 0 и `capsule_w4c4`
  не дают ∫ρ = φ; у 10 форм треугольника нет профиля.
```

заменить на:

```markdown
- Расхождения в исходных моделях — вопросы автору (`05_polygon_pores.md` §4): 9 форм квадрата не перенесены
  (7 условий падают на `del l`; `perforation` при δ > 0 и `capsule_w4c4` не дают ∫ρ = φ); у 10 форм треугольника
  нет профиля; `droplet_w2c1` треугольника при θ > 112.6° и граница `droplet_w2c1` квадрата — с отступлением.
```

`README.md`, блок «Status: pre-alpha», строка 8: фрагмент

```text
 and the square pore (`filament_w1c0`, `droplet_w1c0`).
```

(с пробелом в начале) заменить на

```text
, the triangular pore (14 shapes) and the square pore (7 shapes); stability diagrams by the capillary functional.
```

- [ ] **Шаг 6: Полная проверка и коммит**

Run: `make check`, `make test-env`, `make wheel-check`; на Python 3.10: `UV_PROJECT_ENVIRONMENT=.venv-py310 uv run --no-sync pytest -q`. Всё зелёное, `359 passed, 34 skipped`.

```bash
git add tests docs ROADMAP.md README.md
git commit -m "Стадия 3 закрыта: синтетика многоугольных пор, описание тестов, разбор источника, спецификация и ROADMAP" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
