# Стадия 2 — щель полностью: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Все шесть форм щели (droplet, doughnut, worm, roll, perforation, layer) с прямоугольным основанием ячейки L × kL, сверка со старой PANDA (k = 1) и с формулами RectCell (θ = 180°, δ = 0, любое k), площади поверхностей там, где они следуют из геометрии.

**Architecture:** Ядро почти не меняется: добавляются `default_optimize` в контракт формы (у слоя нет θ) и более надёжная проверка «параметр влияет». Параметр k — поле `SlitPore`/`SlitGeometry`. Осесимметричные формы (капля, пончик, перфорация) наследуют `AxisymmetricSlitShape`: формулы пишутся для квадратной ячейки, перенос на L × kL — общий код (ячейка L × kL ≡ квадратная kL × kL с пересчитанной φ). Червь и валик лежат вдоль короткой стороны, их профиль от k не зависит.

**Tech Stack:** Python ≥ 3.10, numpy, scipy, pydantic v2, pytest, uv, black, ruff, mypy (strict + плагин pydantic).

**Spec:** `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (§3.2–3.3, §6, O5, O7). Источники формул: старая PANDA `PANDA/panda/interface/alpha.py` (93c6dbb); Semenchuk et al., Colloids Surf. A 708 (2025) 135994; Shistko et al., Mol. Simul. (2025), DOI 10.1080/08927022.2025.2574643 (`roughness_2/25_MS_RectCellInterfaceShape.pdf`).

**Проверено до написания плана.** Весь код ниже собран и прогнан по задачам в чистой копии ветки `stage1-core-prototype`: после каждой задачи `black --check`, `ruff check`, `mypy` и `pytest` проходят (итого 217 тестов, 4 пропуска — у пончика и перфорации площади не выведены, при k = 1 и k = 0.4). Код переносить дословно.

## Global Constraints

- Все ограничения стадии 1 в силе: Python ≥ 3.10 (`class X(str, Enum)`, без `StrEnum`/`Self`); объекты-данные — pydantic, без датаклассов (D14); фиксированные варианты — `Enum`, типы у всех полей, говорящие имена (D17, §2.1 спецификации); black 120, ruff, mypy strict; комментарии и сообщения — по-русски.
- Разработка локальная, в ветке `stage1-core-prototype`; в GitHub не отправлять, пока пользователь не скажет.
- Многострочный текст и markdown — только через Write/Edit: в этой среде Bash-heredoc теряет пустые строки и строки, начинающиеся с `#`. Сообщения коммитов — отдельными `-m` на абзац, последний — `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Проверка перед коммитом: `make check`.
- Формулы старой PANDA переносятся без изменений; все отличия (система отсчёта поры у капли и червя, перенос на k) — только в обвязке и проверяются регрессией.

## Review Focus

1. Щель с k ≠ 1, форма, которая касается боковых границ ячейки (например, φ у верхнего предела капли): ожидание — `feasible` = False по пределу, вычисленному по короткой стороне kL, а не по L. Тест — `test_droplet_matches_rectcell` (пределы при k = 0.2) и контракт с k = 0.4 — задача 1.
2. Перфорация при k < 1: пересчёт φ сохраняет объём отверстия, а не жидкости; ошибка здесь даёт φ_eq вне [0, 1] и неверный профиль. Тест — `test_perforation_matches_rectcell` и контракт ∫ρ = φ с k = 0.4 — задача 2.
3. Слой: φ двигает только край ступеньки; фит с `optimize=["phi"]` не должен падать с «параметр не влияет». Тест — `test_shape_without_contact_angle_fits_volume_fraction` — задача 4.
4. Капиллярная энергия формы без θ (слой): ожидание — S_LV + S_SL без обращения к θ. Тест — `test_layer_energy_is_two_flat_interfaces` — задача 4.
5. Регрессия при k = 1: перенос капли на общую осесимметричную базу не должен изменить ни одного числа относительно старой PANDA. Тест — `test_slit_matches_old_panda[droplet]` (остаётся зелёным в задаче 1), полный набор — задача 5.

---

### Задача 1: Прямоугольное основание k, осесимметричная база, капля на ней

**Files:**
- Modify: `src/panda_md/pores/slit/pore.py` (поле `base_aspect_ratio`, `base_area()`, `short_side_ratio()`, `equivalent_square()`, объём поры)
- Create: `src/panda_md/pores/slit/axisymmetric.py`
- Modify: `src/panda_md/pores/slit/droplet.py` (на базе `AxisymmetricSlitShape`, площади поверхностей)
- Create: `tests/unit/test_slit_rectangular_cell.py`
- Modify: `tests/contract/test_shapes.py` (щель с k = 0.4; формы без θ)

**Interfaces:**
- Consumes: `InterfaceShape`, `Pore`, `SurfaceTerms`, `pores.common` (стадия 1).
- Produces: `SlitGeometry(pore_length_ratio, base_aspect_ratio=1.0)` с `base_area() = k l²`, `short_side_ratio() = k l`, `equivalent_square() -> SlitGeometry`; `SlitPore(width_nm, length_nm, base_aspect_ratio=1.0)`, `volume_nm3() = k L² H`; `AxisymmetricSlitShape` — абстрактные `square_profile(coordinate, square, values)`, `square_phi_limits(square, values)`; переопределяемые `square_center_of_mass`, `equivalent_volume_fraction`, `volume_fraction_from_equivalent`, `profile_from_equivalent(geometry, values, coordinate, square_density)`; статические `accessible_width(values)`, `inside_accessible(coordinate, values)`; готовые `profile`, `phi_limits`, `center_of_mass`, `feasible` (строго внутри пределов). `SlitDroplet.surface_terms`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_slit_rectangular_cell.py`:

```python
"""Щель с основанием L × kL против независимых формул Shistko et al., Mol. Simul. (2025) при θ = 180°, δ = 0.

Формулы статьи (уравнения (6)–(27)) записаны здесь заново, в лоб. Пределы φ перфорации в тексте статьи
неоднозначны (уравнение (23)), поэтому они выведены здесь интегрированием объёма отверстия: при d = 1 отверстие
закрывается (φ_max), при d = kl касается своего образа (φ_min).
"""

import math

import numpy as np
import pytest

from panda_md import SlitPore

ASPECT_RATIOS = (0.2, 0.6, 1.0)
LENGTH_RATIOS = (2.0, 4.0)
COORDINATE = np.linspace(-0.4975, 0.4975, 200)
TOLERANCE = 1e-12
FULL_NON_WETTING = 180.0


def pore(aspect_ratio: float, length_ratio: float) -> SlitPore:
    return SlitPore(width_nm=1.0, length_nm=length_ratio, base_aspect_ratio=aspect_ratio)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_droplet_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    droplet = pore(aspect_ratio, length_ratio).interface("droplet")
    upper = min(math.pi / (6 * aspect_ratio * length_ratio**2), math.pi * aspect_ratio**2 * length_ratio / 6)  # (7)
    assert droplet.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = (3 * phi * aspect_ratio * length_ratio**2 / (4 * math.pi)) ** (1 / 3)  # (6)
    center = -0.5 + radius  # при θ = 180° сфера касается нижней стенки
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        math.pi / (aspect_ratio * length_ratio**2) * (radius**2 - (COORDINATE - center) ** 2),  # (8)
        0.0,
    )
    np.testing.assert_allclose(droplet.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = droplet.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(4 * math.pi * radius**2)  # (9)


@pytest.mark.parametrize("shape_name", ["droplet"])
def test_axisymmetric_shape_in_rectangle_equals_square_of_short_side(shape_name: str) -> None:
    """ρ(z; l, k, φ) = k · ρ(z; kl, 1, φ/k): ячейка L × kL эквивалентна квадратной kL × kL."""
    rectangle = SlitPore(width_nm=1.0, length_nm=5.0, base_aspect_ratio=0.4).interface(shape_name)
    square = SlitPore(width_nm=1.0, length_nm=2.0).interface(shape_name)
    values = {"theta": 140.0, "delta": 0.03}
    lower, upper = rectangle.phi_limits(**values, phi=0.1)
    phi = 0.5 * (lower + upper)
    np.testing.assert_allclose(
        rectangle.profile(COORDINATE, **values, phi=phi),
        0.4 * square.profile(COORDINATE, **values, phi=phi / 0.4),
        atol=TOLERANCE,
    )


def test_pore_volume_uses_rectangular_base() -> None:
    assert SlitPore(width_nm=2.0, length_nm=10.0, base_aspect_ratio=0.3).volume_nm3() == pytest.approx(60.0)
```

Заменить `tests/contract/test_shapes.py` целиком:

`tests/contract/test_shapes.py`:

```python
"""Контракт формы — проверяется автоматически для КАЖДОЙ зарегистрированной формы каждой поры.

Новая форма, подключённая декоратором @Pore.register, попадает сюда без правки тестов. Щель проверяется
и с квадратным, и с прямоугольным основанием (k = 0.4).
"""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import SlitPore, SquarePore
from panda_md.core.enums import ParameterName, Regime
from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.core.pore import Pore
from panda_md.core.shape import InterfaceShape, regime_of

PORES: tuple[Pore, ...] = (
    SlitPore(width_nm=8.0, length_nm=16.0),
    SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.4),  # прямоугольное основание (RectCell)
    SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
)
MODELS = [model for pore in PORES for model in pore.available_interfaces()]
INTEGRATION_GRID = np.linspace(-0.5, 0.5, 200_001)
INTEGRAL_TOLERANCE = 1e-4  # ступенька на краю профиля даёт ошибку трапеций ~ шаг сетки
CENTER_OF_MASS_TOLERANCE = 1e-4
SAMPLE_LIMIT = 12
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = np.linspace(0.02, 0.98, 25)


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
    integral_per_phi = model.pore.geometry().profile_integral()
    for values in feasible_samples(model):
        density = model.profile(INTEGRATION_GRID, **values)
        integral = trapezoid(density, INTEGRATION_GRID)
        assert integral == pytest.approx(values["phi"] * integral_per_phi, rel=INTEGRAL_TOLERANCE), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_profile_is_finite_bounded_and_zero_outside_pore(model: InterfaceModel) -> None:
    outside = np.array([-0.9, -0.51, 0.51, 0.9])
    for values in feasible_samples(model):
        density = model.profile(INTEGRATION_GRID, **values)
        assert np.all(np.isfinite(density)), values
        assert np.all(density >= 0.0), values
        assert np.all(density <= 1.0 + 1e-12), values
        assert np.all(model.profile(outside, **values) == 0.0), values


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_center_of_mass_matches_numerical_value(model: InterfaceModel) -> None:
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

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q`
Expected: ошибка сборки `tests/contract/test_shapes.py` — `ValidationError ... base_aspect_ratio ... Extra inputs are not permitted`.

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/pores/slit/pore.py` целиком:

`src/panda_md/pores/slit/pore.py`:

```python
"""Щелевая пора: две плоские стенки на расстоянии H, прямоугольное основание L × kL (0 < k ≤ 1).

Параметр k (squeeze parameter) — Shistko et al., Mol. Simul. (2025), DOI 10.1080/08927022.2025.2574643.
"""

from typing import ClassVar

from pydantic import Field

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry

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

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=self.length_nm / self.width_nm, base_aspect_ratio=self.base_aspect_ratio)

    def characteristic_size_nm(self) -> float:
        return self.width_nm

    def volume_nm3(self) -> float:
        return self.base_aspect_ratio * self.length_nm**2 * self.width_nm
```

`src/panda_md/pores/slit/axisymmetric.py`:

```python
"""Осесимметричные формы щели (капля, пончик, перфорация) в ячейке с основанием L × kL.

Формулы таких форм записаны для квадратной ячейки (Semenchuk et al., 2025). Если форма не касается боковых
границ ячейки, она «видит» только короткую сторону kL: ячейка L × kL эквивалентна квадратной ячейке kL × kL,
в которой та же жидкость занимает другую долю объёма. Подкласс задаёт формулы квадратной ячейки и правило
пересчёта φ; профиль, границы φ и центр масс в ячейке L × kL выводятся отсюда. При k = 1 пересчёт тождественный.
Проверка — формулы Shistko et al. (2025) при θ = 180°, δ = 0 (tests/unit/test_slit_rectangular_cell.py).
"""

from abc import abstractmethod
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import BoolArray, FloatArray
from panda_md.pores.slit.pore import SlitGeometry


class AxisymmetricSlitShape(InterfaceShape[SlitGeometry]):
    """База осесимметричной формы щели: формулы квадратной ячейки + пересчёт на ячейку L × kL."""

    frame: ClassVar[Frame] = Frame.PORE

    @abstractmethod
    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk в квадратной ячейке со стороной l (k = 1)."""

    @abstractmethod
    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """Интервал φ в квадратной ячейке."""

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        return InterfaceShape.center_of_mass(self, square, values)

    def equivalent_volume_fraction(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """φ в эквивалентной квадратной ячейке: объём жидкости сохраняется, площадь основания k l² → (k l)²."""
        return values[ParameterName.VOLUME_FRACTION] / geometry.base_aspect_ratio

    def volume_fraction_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, equivalent: float
    ) -> float:
        """Обратный пересчёт: φ в ячейке L × kL по φ эквивалентной квадратной ячейки."""
        return equivalent * geometry.base_aspect_ratio

    def profile_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, coordinate: FloatArray, square_density: FloatArray
    ) -> FloatArray:
        """ρ в ячейке L × kL: то же сечение жидкости, но основание в k раз меньше квадратного (kl)²/(k l²) = k."""
        return geometry.base_aspect_ratio * square_density

    def square_values(self, geometry: SlitGeometry, values: ParameterValues) -> dict[str, float]:
        converted = dict(values)
        converted[ParameterName.VOLUME_FRACTION] = self.equivalent_volume_fraction(geometry, values)
        return converted

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        square_density = self.square_profile(
            coordinate, geometry.equivalent_square(), self.square_values(geometry, values)
        )
        return self.profile_from_equivalent(geometry, values, coordinate, square_density)

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        lower, upper = self.square_phi_limits(geometry.equivalent_square(), self.square_values(geometry, values))
        return (
            self.volume_fraction_from_equivalent(geometry, values, lower),
            self.volume_fraction_from_equivalent(geometry, values, upper),
        )

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return self.square_center_of_mass(geometry.equivalent_square(), self.square_values(geometry, values))

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if self.accessible_width(values) <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return bool(lower < values[ParameterName.VOLUME_FRACTION] < upper)

    @staticmethod
    def accessible_width(values: ParameterValues) -> float:
        """1 − 2δ — ширина щели, доступная анализируемой жидкости."""
        return 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]

    @staticmethod
    def inside_accessible(coordinate: FloatArray, values: ParameterValues) -> BoolArray:
        """|z| < 1/2 − δ — точки, доступные анализируемой жидкости."""
        half = 0.5 * (1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS])
        return np.abs(coordinate) < half
```

Заменить `src/panda_md/pores/slit/droplet.py` целиком:

`src/panda_md/pores/slit/droplet.py`:

```python
"""Капля (droplet) в щели: сферический сегмент на нижней стенке.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (1), (3)–(5); границы существования —
по старой PANDA (`interface/alpha.py`, ветка θ ≥ 90°). Профиль записан в системе поры: капля стоит на слое δ
над нижней стенкой z = −1/2. Ячейка L × kL — через эквивалентную квадратную ячейку (axisymmetric.py).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import spherical_cap_centroid_offset, spherical_cap_volume_factor
from panda_md.pores.slit.axisymmetric import AxisymmetricSlitShape
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore


@SlitPore.register
class SlitDroplet(AxisymmetricSlitShape):
    name: ClassVar[str] = "droplet"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def sphere_radius(self, square: SlitGeometry, values: ParameterValues) -> float:
        """R из π R³ (2 − 3 cos θ + cos³ θ) / 3 = φ l², уравнение (1) (квадратная ячейка)."""
        volume = values[ParameterName.VOLUME_FRACTION] * square.pore_length_ratio**2
        return float(
            np.cbrt(3.0 * volume / (math.pi * spherical_cap_volume_factor(values[ParameterName.CONTACT_ANGLE])))
        )

    def sphere_center(self, square: SlitGeometry, values: ParameterValues) -> float:
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        return floor - self.sphere_radius(square, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ ≤ min(капля не выше доступной щели, сфера не шире ячейки) — уравнение (3) и alpha.py."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        by_height = (
            math.pi
            * accessible_width**3
            * (2.0 + math.cos(contact_angle))
            / (3.0 * length_ratio**2 * (1.0 - math.cos(contact_angle)))
        )
        by_length = math.pi * length_ratio * (2.0 + math.cos(contact_angle)) * math.sin(0.5 * contact_angle) ** 4 / 6.0
        return 0.0, min(by_height, by_length)

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        radius = self.sphere_radius(square, values)
        return self.sphere_center(square, values) + spherical_cap_centroid_offset(
            radius, values[ParameterName.CONTACT_ANGLE]
        )

    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / l² между опорой капли и её верхушкой, уравнение (5)."""
        radius = self.sphere_radius(square, values)
        center = self.sphere_center(square, values)
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        inside = (coordinate > floor) & (coordinate < center + radius)  # строгие границы, как в alpha.py
        section_area = math.pi * np.maximum(radius**2 - (coordinate - center) ** 2, 0.0)
        return np.where(inside, section_area / square.pore_length_ratio**2, 0.0)

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        """Сферический сегмент: S_LV = 2π R² (1 − cos θ), S_SL = π R² sin² θ (от k не зависят)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.sphere_radius(geometry.equivalent_square(), self.square_values(geometry, values))
        return SurfaceTerms(
            liquid_liquid_area=2.0 * math.pi * radius**2 * (1.0 - math.cos(contact_angle)),
            liquid_solid_area=math.pi * radius**2 * math.sin(contact_angle) ** 2,
        )
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `132 passed, 2 skipped`; `test_slit_matches_old_panda[droplet]` по-прежнему зелёный (перенос на базу ничего не изменил при k = 1).

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/pores/slit tests/unit/test_slit_rectangular_cell.py tests/contract/test_shapes.py
git commit -m "Щель: основание L × kL (RectCell), общая база осесимметричных форм, капля на ней" \
  -m "Ячейка L × kL для осесимметричной формы эквивалентна квадратной kL × kL; проверено по формулам Shistko et al. (2025) при θ = 180°." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 2: Пончик и перфорация

**Files:**
- Create: `src/panda_md/pores/slit/doughnut.py`, `src/panda_md/pores/slit/perforation.py`
- Modify: `src/panda_md/pores/slit/__init__.py`, `tests/unit/test_slit_rectangular_cell.py`, `tests/unit/test_model.py`

**Interfaces:**
- Consumes: `AxisymmetricSlitShape` (задача 1).
- Produces: `SlitDoughnut` (`"doughnut"`; `contact_spot_diameter`, `square_phi_limits`, `square_profile`; центр масс 0; площади не выведены), `SlitPerforation` (`"perforation"`; переопределяет пересчёт φ — сохраняется объём отверстия — и `profile_from_equivalent`: ρ = 1 − k(1 − ρ_кв) внутри щели; `hole_diameter`).

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_slit_rectangular_cell.py` целиком:

`tests/unit/test_slit_rectangular_cell.py`:

```python
"""Щель с основанием L × kL против независимых формул Shistko et al., Mol. Simul. (2025) при θ = 180°, δ = 0.

Формулы статьи (уравнения (6)–(27)) записаны здесь заново, в лоб. Пределы φ перфорации в тексте статьи
неоднозначны (уравнение (23)), поэтому они выведены здесь интегрированием объёма отверстия: при d = 1 отверстие
закрывается (φ_max), при d = kl касается своего образа (φ_min).
"""

import math

import numpy as np
import pytest

from panda_md import SlitPore

ASPECT_RATIOS = (0.2, 0.6, 1.0)
LENGTH_RATIOS = (2.0, 4.0)
COORDINATE = np.linspace(-0.4975, 0.4975, 200)
TOLERANCE = 1e-12
FULL_NON_WETTING = 180.0


def pore(aspect_ratio: float, length_ratio: float) -> SlitPore:
    return SlitPore(width_nm=1.0, length_nm=length_ratio, base_aspect_ratio=aspect_ratio)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_droplet_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    droplet = pore(aspect_ratio, length_ratio).interface("droplet")
    upper = min(math.pi / (6 * aspect_ratio * length_ratio**2), math.pi * aspect_ratio**2 * length_ratio / 6)  # (7)
    assert droplet.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = (3 * phi * aspect_ratio * length_ratio**2 / (4 * math.pi)) ** (1 / 3)  # (6)
    center = -0.5 + radius  # при θ = 180° сфера касается нижней стенки
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        math.pi / (aspect_ratio * length_ratio**2) * (radius**2 - (COORDINATE - center) ** 2),  # (8)
        0.0,
    )
    np.testing.assert_allclose(droplet.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = droplet.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(4 * math.pi * radius**2)  # (9)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_doughnut_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    doughnut = pore(aspect_ratio, length_ratio).interface("doughnut")
    side = aspect_ratio * length_ratio
    lower = math.pi / (6 * aspect_ratio * length_ratio**2)  # (11)
    upper = math.pi / (4 * aspect_ratio * length_ratio**2) * ((side - 1) ** 2 + math.pi * (side - 1) / 2 + 2 / 3)
    assert doughnut.phi_limits(theta=FULL_NON_WETTING, phi=0.1) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    spot = -math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * phi * aspect_ratio * length_ratio**2 / math.pi)
    expected = math.pi / (4 * aspect_ratio * length_ratio**2) * (np.sqrt(1 - 4 * COORDINATE**2) + spot) ** 2  # (12)
    np.testing.assert_allclose(doughnut.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_perforation_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    perforation = pore(aspect_ratio, length_ratio).interface("perforation")
    side = aspect_ratio * length_ratio
    base = aspect_ratio * length_ratio**2
    lower = 1 - math.pi / (4 * base) * (side**2 - math.pi * side / 2 + 2 / 3)  # отверстие касается образа
    upper = 1 - math.pi / (4 * base) * (5 / 3 - math.pi / 2)  # отверстие закрывается
    assert perforation.phi_limits(theta=FULL_NON_WETTING, phi=0.9) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    hole = math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * (1 - phi) * base / math.pi)  # (22)
    expected = 1 - math.pi / (4 * base) * (np.sqrt(1 - 4 * COORDINATE**2) - hole) ** 2  # (24)
    np.testing.assert_allclose(
        perforation.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE
    )


@pytest.mark.parametrize("shape_name", ["droplet", "doughnut"])
def test_axisymmetric_shape_in_rectangle_equals_square_of_short_side(shape_name: str) -> None:
    """ρ(z; l, k, φ) = k · ρ(z; kl, 1, φ/k): ячейка L × kL эквивалентна квадратной kL × kL."""
    rectangle = SlitPore(width_nm=1.0, length_nm=5.0, base_aspect_ratio=0.4).interface(shape_name)
    square = SlitPore(width_nm=1.0, length_nm=2.0).interface(shape_name)
    values = {"theta": 140.0, "delta": 0.03}
    lower, upper = rectangle.phi_limits(**values, phi=0.1)
    phi = 0.5 * (lower + upper)
    np.testing.assert_allclose(
        rectangle.profile(COORDINATE, **values, phi=phi),
        0.4 * square.profile(COORDINATE, **values, phi=phi / 0.4),
        atol=TOLERANCE,
    )


def test_pore_volume_uses_rectangular_base() -> None:
    assert SlitPore(width_nm=2.0, length_nm=10.0, base_aspect_ratio=0.3).volume_nm3() == pytest.approx(60.0)
```

Заменить `tests/unit/test_model.py` целиком (набор форм щели):

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


def test_surface_terms_not_derived_for_slit_roll() -> None:
    with pytest.raises(NotDerivedError):
        SLIT.interface("roll").surface_terms(theta=120.0, phi=0.4)


def test_square_pore_exposes_its_shapes() -> None:
    assert set(SquarePore.shape_names()) == {"filament_w1c0", "droplet_w1c0"}
    assert set(SlitPore.shape_names()) == {"droplet", "doughnut", "roll", "perforation"}


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
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q`
Expected: падают `test_square_pore_exposes_its_shapes` (нет `doughnut`, `perforation`) и тесты RectCell с `KeyError: "в поре slit нет формы doughnut; доступны: ['droplet', 'roll']"`.

- [ ] **Шаг 3: Реализация**

`src/panda_md/pores/slit/doughnut.py`:

```python
"""Пончик (doughnut) в щели: перемычка между стенками, осесимметричная вокруг вертикальной оси.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (6)–(9), с поправкой на слой δ
по старой PANDA (`interface/alpha.py`). Ячейка L × kL — через эквивалентную квадратную ячейку (axisymmetric.py).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.types import FloatArray
from panda_md.pores.slit.axisymmetric import AxisymmetricSlitShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore


@SlitPore.register
class SlitDoughnut(AxisymmetricSlitShape):
    name: ClassVar[str] = "doughnut"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def contact_spot_diameter(self, square: SlitGeometry, values: ParameterValues) -> float:
        """d_dou — диаметр пятна контакта, уравнение (6) с учётом δ (alpha.py)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        discriminant = (
            width**2 * arc_term**2 / cos_angle**4
            - (12.0 - 4.0 * arc_term * tan_angle) * width**2 / cos_angle**2
            + 64.0 * values[ParameterName.VOLUME_FRACTION] * length_ratio**2 / math.pi / width
            + 4.0 * width**2 / 3.0
        )
        return (
            0.25 * math.sqrt(max(discriminant, 0.0)) - 0.5 * width * tan_angle + 0.25 * width * arc_term / cos_angle**2
        )

    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min — пятно контакта исчезает; φ_max — пончик касается своего образа (уравнение (8) с δ, alpha.py)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        lower = (
            math.pi
            * width**3
            / length_ratio**2
            * (tan_angle**2 / 4.0 + 1.0 / 6.0 - arc_term * tan_angle / cos_angle**2 / 8.0)
        )
        upper = (
            math.pi
            * width
            / 16.0
            / length_ratio**2
            * (
                width**2 * tan_angle**2
                - 2.0 * width**2 * arc_term / cos_angle**3
                - (width**2 / 3.0 - 4.0 * length_ratio**2)
                - 4.0 * width * length_ratio * tan_angle
                + width * (7.0 * width - 2.0 * arc_term * length_ratio) / cos_angle**2
                - 4.0 * width * (width * tan_angle - 2.0 * length_ratio) / cos_angle
            )
        )
        return lower, upper

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # пончик симметричен относительно середины щели

    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π y² / l², y = (d + √((1−2δ)²/cos²θ − 4z²) + (1−2δ) tan θ) / 2 — уравнение (9) с δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        meniscus = np.sqrt(np.maximum(width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        radius = 0.5 * (self.contact_spot_diameter(square, values) + meniscus + width * math.tan(contact_angle))
        density = math.pi * radius**2 / square.pore_length_ratio**2
        return np.where(self.inside_accessible(coordinate, values), density, 0.0)
```

`src/panda_md/pores/slit/perforation.py`:

```python
"""Перфорация (perforation) в щели: слой жидкости между стенками с осесимметричным отверстием.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (19)–(22), с поправкой на слой δ
по старой PANDA (`interface/alpha.py`). Ячейка L × kL — через эквивалентную квадратную ячейку: сохраняется
объём отверстия, а не жидкости (axisymmetric.py).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.types import FloatArray
from panda_md.pores.slit.axisymmetric import AxisymmetricSlitShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore


@SlitPore.register
class SlitPerforation(AxisymmetricSlitShape):
    name: ClassVar[str] = "perforation"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def equivalent_volume_fraction(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """Объём отверстия ((1−2δ) − φ) k l² сохраняется в квадратной ячейке со стороной k l."""
        width = self.accessible_width(values)
        return width - (width - values[ParameterName.VOLUME_FRACTION]) / geometry.base_aspect_ratio

    def volume_fraction_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, equivalent: float
    ) -> float:
        width = self.accessible_width(values)
        return width - geometry.base_aspect_ratio * (width - equivalent)

    def profile_from_equivalent(
        self, geometry: SlitGeometry, values: ParameterValues, coordinate: FloatArray, square_density: FloatArray
    ) -> FloatArray:
        """Доля сечения, занятая отверстием, в k раз меньше квадратной: ρ = 1 − k (1 − ρ_кв) внутри щели."""
        density = 1.0 - geometry.base_aspect_ratio * (1.0 - square_density)
        return np.where(self.inside_accessible(coordinate, values), density, 0.0)

    def hole_diameter(self, square: SlitGeometry, values: ParameterValues) -> float:
        """d_perf — диаметр отверстия у стенки, уравнение (19) с учётом δ (alpha.py)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        discriminant = (
            width**2 * arc_term**2 / cos_angle**4
            - (12.0 - 4.0 * arc_term * tan_angle) * width**2 / cos_angle**2
            + 64.0 * (1.0 - values[ParameterName.VOLUME_FRACTION] / width) * length_ratio**2 / math.pi
            + 4.0 * width**2 / 3.0
        )
        return (
            0.25 * math.sqrt(max(discriminant, 0.0)) + 0.5 * width * tan_angle - 0.25 * width * arc_term / cos_angle**2
        )

    def square_phi_limits(self, square: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min — отверстие касается своего образа; φ_max — отверстие закрывается (уравнение (21) с δ, alpha.py)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        length_ratio = square.pore_length_ratio
        cos_angle = math.cos(contact_angle)
        tan_angle = math.tan(contact_angle)
        arc_term = math.pi - 2.0 * contact_angle
        lower = width * (1.0 - math.pi / 4.0) + math.pi * width**2 / 8.0 / length_ratio**2 * (
            2.0 * width / 3.0
            + 2.0 * length_ratio * tan_angle
            - (2.0 * width + arc_term * length_ratio - width * arc_term * tan_angle) / cos_angle**2
        )
        upper = width - math.pi * width**3 / (16.0 * length_ratio**2 * cos_angle**3) * (
            7.0 * cos_angle - math.cos(3.0 * contact_angle) / 3.0 - 2.0 * (arc_term + math.sin(2.0 * contact_angle))
        )
        return lower, upper

    def square_center_of_mass(self, square: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # перфорация симметрична относительно середины щели

    def square_profile(self, coordinate: FloatArray, square: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 1 − π y² / l², y = (d − √((1−2δ)²/cos²θ − 4z²) − (1−2δ) tan θ) / 2 — уравнение (22) с δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        width = self.accessible_width(values)
        meniscus = np.sqrt(np.maximum(width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        hole_radius = 0.5 * (self.hole_diameter(square, values) - meniscus - width * math.tan(contact_angle))
        density = 1.0 - math.pi * hole_radius**2 / square.pore_length_ratio**2
        return np.where(self.inside_accessible(coordinate, values), density, 0.0)
```

Заменить `src/panda_md/pores/slit/__init__.py` целиком:

`src/panda_md/pores/slit/__init__.py`:

```python
"""Щелевая пора и её формы. Импорт модулей форм регистрирует их в SlitPore."""

from panda_md.pores.slit import doughnut, droplet, perforation, roll
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

__all__ = ["SlitGeometry", "SlitPore", "doughnut", "droplet", "perforation", "roll"]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `165 passed, 6 skipped` (пропуски — площади пончика и перфорации при k = 1 и k = 0.4, плюс пока ещё не выведенные площади валика).

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/pores/slit tests/unit
git commit -m "Щель: пончик и перфорация (формулы alpha.py с δ, перенос на L × kL)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 3: Червь; площади валика; общие формулы кругового сегмента

**Files:**
- Modify: `src/panda_md/pores/common.py` (круговой сегмент), `src/panda_md/pores/square/filament.py` (берёт формулы оттуда), `src/panda_md/pores/slit/roll.py` (площади), `src/panda_md/pores/slit/__init__.py`, `tests/unit/test_slit_rectangular_cell.py`, `tests/unit/test_model.py`
- Create: `src/panda_md/pores/slit/worm.py`

**Interfaces:**
- Consumes: задачи 1–2.
- Produces: `pores.common.circular_segment_factor(contact_angle)`, `circular_segment_centroid_offset(radius, contact_angle)` (перенесены из `square/filament.py`); `SlitWorm` (`"worm"`; `cylinder_radius`, `cylinder_center`, `phi_limits`, `feasible`, `center_of_mass`, `surface_terms` — на длину k l, `profile` — система поры, строгие границы); `SlitRoll.surface_terms` (дуги мениска и два пятна контакта на длину k l).

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_slit_rectangular_cell.py` целиком:

`tests/unit/test_slit_rectangular_cell.py`:

```python
"""Щель с основанием L × kL против независимых формул Shistko et al., Mol. Simul. (2025) при θ = 180°, δ = 0.

Формулы статьи (уравнения (6)–(27)) записаны здесь заново, в лоб. Пределы φ перфорации в тексте статьи
неоднозначны (уравнение (23)), поэтому они выведены здесь интегрированием объёма отверстия: при d = 1 отверстие
закрывается (φ_max), при d = kl касается своего образа (φ_min).
"""

import math

import numpy as np
import pytest

from panda_md import SlitPore

ASPECT_RATIOS = (0.2, 0.6, 1.0)
LENGTH_RATIOS = (2.0, 4.0)
COORDINATE = np.linspace(-0.4975, 0.4975, 200)
TOLERANCE = 1e-12
FULL_NON_WETTING = 180.0


def pore(aspect_ratio: float, length_ratio: float) -> SlitPore:
    return SlitPore(width_nm=1.0, length_nm=length_ratio, base_aspect_ratio=aspect_ratio)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_droplet_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    droplet = pore(aspect_ratio, length_ratio).interface("droplet")
    upper = min(math.pi / (6 * aspect_ratio * length_ratio**2), math.pi * aspect_ratio**2 * length_ratio / 6)  # (7)
    assert droplet.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = (3 * phi * aspect_ratio * length_ratio**2 / (4 * math.pi)) ** (1 / 3)  # (6)
    center = -0.5 + radius  # при θ = 180° сфера касается нижней стенки
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        math.pi / (aspect_ratio * length_ratio**2) * (radius**2 - (COORDINATE - center) ** 2),  # (8)
        0.0,
    )
    np.testing.assert_allclose(droplet.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = droplet.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(4 * math.pi * radius**2)  # (9)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_doughnut_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    doughnut = pore(aspect_ratio, length_ratio).interface("doughnut")
    side = aspect_ratio * length_ratio
    lower = math.pi / (6 * aspect_ratio * length_ratio**2)  # (11)
    upper = math.pi / (4 * aspect_ratio * length_ratio**2) * ((side - 1) ** 2 + math.pi * (side - 1) / 2 + 2 / 3)
    assert doughnut.phi_limits(theta=FULL_NON_WETTING, phi=0.1) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    spot = -math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * phi * aspect_ratio * length_ratio**2 / math.pi)
    expected = math.pi / (4 * aspect_ratio * length_ratio**2) * (np.sqrt(1 - 4 * COORDINATE**2) + spot) ** 2  # (12)
    np.testing.assert_allclose(doughnut.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_worm_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    worm = pore(aspect_ratio, length_ratio).interface("worm")
    upper = min(math.pi / (4 * length_ratio), math.pi * length_ratio / 4)  # (15)
    assert worm.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = math.sqrt(phi * length_ratio / math.pi)  # (14)
    center = -0.5 + radius
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        2 / length_ratio * np.sqrt(np.maximum(radius**2 - (COORDINATE - center) ** 2, 0.0)),  # (16)
        0.0,
    )
    np.testing.assert_allclose(worm.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = worm.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
        2 * math.pi * aspect_ratio * length_ratio * radius  # (17)
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_roll_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    roll = pore(aspect_ratio, length_ratio).interface("roll")
    lower, upper = math.pi / (4 * length_ratio), 1 - (1 - math.pi / 4) / length_ratio  # (19)
    assert roll.phi_limits(theta=FULL_NON_WETTING, phi=0.5) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    spot = phi * length_ratio - math.pi / 4  # (18)
    expected = (np.sqrt(1 - 4 * COORDINATE**2) + spot) / length_ratio  # (20)
    np.testing.assert_allclose(roll.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = roll.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
        aspect_ratio * length_ratio * (2 * spot + math.pi)  # (21)
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_perforation_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    perforation = pore(aspect_ratio, length_ratio).interface("perforation")
    side = aspect_ratio * length_ratio
    base = aspect_ratio * length_ratio**2
    lower = 1 - math.pi / (4 * base) * (side**2 - math.pi * side / 2 + 2 / 3)  # отверстие касается образа
    upper = 1 - math.pi / (4 * base) * (5 / 3 - math.pi / 2)  # отверстие закрывается
    assert perforation.phi_limits(theta=FULL_NON_WETTING, phi=0.9) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    hole = math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * (1 - phi) * base / math.pi)  # (22)
    expected = 1 - math.pi / (4 * base) * (np.sqrt(1 - 4 * COORDINATE**2) - hole) ** 2  # (24)
    np.testing.assert_allclose(
        perforation.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE
    )


@pytest.mark.parametrize("shape_name", ["droplet", "doughnut"])
def test_axisymmetric_shape_in_rectangle_equals_square_of_short_side(shape_name: str) -> None:
    """ρ(z; l, k, φ) = k · ρ(z; kl, 1, φ/k): ячейка L × kL эквивалентна квадратной kL × kL."""
    rectangle = SlitPore(width_nm=1.0, length_nm=5.0, base_aspect_ratio=0.4).interface(shape_name)
    square = SlitPore(width_nm=1.0, length_nm=2.0).interface(shape_name)
    values = {"theta": 140.0, "delta": 0.03}
    lower, upper = rectangle.phi_limits(**values, phi=0.1)
    phi = 0.5 * (lower + upper)
    np.testing.assert_allclose(
        rectangle.profile(COORDINATE, **values, phi=phi),
        0.4 * square.profile(COORDINATE, **values, phi=phi / 0.4),
        atol=TOLERANCE,
    )


def test_pore_volume_uses_rectangular_base() -> None:
    assert SlitPore(width_nm=2.0, length_nm=10.0, base_aspect_ratio=0.3).volume_nm3() == pytest.approx(60.0)
```

Заменить `tests/unit/test_model.py` целиком (площади валика теперь выведены — тест «не выведено» переходит на пончик; набор форм):

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


def test_square_pore_exposes_its_shapes() -> None:
    assert set(SquarePore.shape_names()) == {"filament_w1c0", "droplet_w1c0"}
    assert set(SlitPore.shape_names()) == {"droplet", "doughnut", "worm", "roll", "perforation"}


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
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q`
Expected: падают `test_square_pore_exposes_its_shapes` (нет `worm`), тесты червя с `KeyError: "в поре slit нет формы worm; ..."` и `test_roll_matches_rectcell` (площади валика не выведены — `NotDerivedError`).

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/pores/common.py` целиком:

`src/panda_md/pores/common.py`:

```python
"""Геометрия, общая для форм разных пор: сферический сегмент."""

import math


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
```

Заменить `src/panda_md/pores/square/filament.py` целиком:

`src/panda_md/pores/square/filament.py`:

```python
"""Нити (filament) в квадратной поре. Формулы — PANDA_Polygon `square.py` (коммит 38ff491)."""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_centroid_offset, circular_segment_factor
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side


@SquarePore.register
class SquareFilamentOneWall(InterfaceShape[SquareGeometry]):
    """filament_w1c0: цилиндрический сегмент вдоль оси поры на нижней стенке, углы свободны."""

    name: ClassVar[str] = "filament_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {
        Regime.WETTING: (0.0, 90.0),
        Regime.NON_WETTING: (90.0, 180.0),
    }
    frame: ClassVar[Frame] = Frame.PORE

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
```

`src/panda_md/pores/slit/worm.py`:

```python
"""Червь (worm) в щели: цилиндрический сегмент на нижней стенке, однородный вдоль короткой стороны ячейки.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (10)–(14); границы существования —
по старой PANDA (`interface/alpha.py`, ветка θ > 90°). Профиль записан в системе поры: червь лежит на слое δ над
нижней стенкой. В ячейке L × kL ось червя идёт вдоль короткой стороны (так площадь меньше — Shistko et al., 2025),
сечение лежит в плоскости длинной стороны, поэтому от k профиль и границы не зависят.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_centroid_offset, circular_segment_factor
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore


@SlitPore.register
class SlitWorm(InterfaceShape[SlitGeometry]):
    name: ClassVar[str] = "worm"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def cylinder_radius(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """R = √(φ l / (θ − sin θ cos θ)): площадь сечения сегмента равна φ l, уравнение (10)."""
        segment = circular_segment_factor(values[ParameterName.CONTACT_ANGLE])
        return math.sqrt(values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio / segment)

    def cylinder_center(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        return floor - self.cylinder_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ ≤ min(червь не выше доступной щели, цилиндр не шире ячейки) — уравнение (12) и alpha.py."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length_ratio = geometry.pore_length_ratio
        double_segment = 2.0 * contact_angle - math.sin(2.0 * contact_angle)
        by_height = accessible_width**2 * double_segment / (2.0 * length_ratio * (1.0 - math.cos(contact_angle)) ** 2)
        by_length = length_ratio * double_segment / 8.0
        return 0.0, min(by_height, by_length)

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS] <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return bool(lower < values[ParameterName.VOLUME_FRACTION] < upper)

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        radius = self.cylinder_radius(geometry, values)
        offset = circular_segment_centroid_offset(radius, values[ParameterName.CONTACT_ANGLE])
        return self.cylinder_center(geometry, values) + offset

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        """Дуга 2Rθ и хорда 2R sin θ, умноженные на длину червя k l (короткая сторона)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.cylinder_radius(geometry, values)
        length = geometry.short_side_ratio()
        return SurfaceTerms(
            liquid_liquid_area=2.0 * radius * contact_angle * length,
            liquid_solid_area=2.0 * radius * math.sin(contact_angle) * length,
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 2 √(R² − (z − z_c)²) / l между опорой и верхушкой, уравнение (14)."""
        radius = self.cylinder_radius(geometry, values)
        center = self.cylinder_center(geometry, values)
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        inside = (coordinate > floor) & (coordinate < center + radius)  # строгие границы, как в alpha.py
        chord = 2.0 * np.sqrt(np.maximum(radius**2 - (coordinate - center) ** 2, 0.0))
        return np.where(inside, chord / geometry.pore_length_ratio, 0.0)
```

Заменить `src/panda_md/pores/slit/roll.py` целиком:

`src/panda_md/pores/slit/roll.py`:

```python
"""Валик (roll) в щели: перемычка между стенками, однородная вдоль одной оси.

Формулы: Semenchuk et al., Colloids Surf. A 725 (2025) 137586, уравнения (1)–(4);
при δ = 0 совпадают с Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (15)–(18).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore


@SlitPore.register
class SlitRoll(InterfaceShape[SlitGeometry]):
    name: ClassVar[str] = "roll"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def contact_spot_width(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """d_roll — ширина пятна контакта, уравнение (1)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length_ratio = geometry.pore_length_ratio
        return (
            values[ParameterName.VOLUME_FRACTION] * length_ratio / accessible_width
            + accessible_width * (math.pi - 2.0 * contact_angle) / (4.0 * math.cos(contact_angle) ** 2)
            - 0.5 * accessible_width * math.tan(contact_angle)
        )

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min и φ_max, уравнения (3) и (4)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length_ratio = geometry.pore_length_ratio
        cos_sq = math.cos(contact_angle) ** 2
        lower = (
            accessible_width**2
            / (2.0 * length_ratio)
            * (math.tan(contact_angle) - (math.pi - 2.0 * contact_angle) / (2.0 * cos_sq))
        )
        upper = accessible_width + accessible_width**2 / length_ratio * (
            1.0 / math.cos(contact_angle)
            - (math.pi - 2.0 * contact_angle) / (4.0 * cos_sq)
            - 0.5 * math.tan(contact_angle)
        )
        return lower, upper

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS] <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return lower < values[ParameterName.VOLUME_FRACTION] < upper

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # валик симметричен относительно середины щели

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        """Две дуги мениска R(2θ − π) · 2 = (1−2δ)(π − 2θ)/cos θ и два пятна контакта 2d — на длину валика k l.
        При θ = 180°, δ = 0 сумма — kl (2d + π), уравнение (21) Shistko et al. (2025).
        """
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length = geometry.short_side_ratio()
        return SurfaceTerms(
            liquid_liquid_area=length * accessible_width * (math.pi - 2.0 * contact_angle) / math.cos(contact_angle),
            liquid_solid_area=2.0 * self.contact_spot_width(geometry, values) * length,
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk, уравнение (2): (d + √((1−2δ)²/cos²θ − 4z²) + (1−2δ) tan θ) / l при |z| < 1/2 − δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        wall_layer = values[ParameterName.WALL_LAYER_THICKNESS]
        accessible_width = 1.0 - 2.0 * wall_layer
        inside = np.abs(coordinate) < SLIT_HALF_WIDTH - wall_layer
        meniscus = np.sqrt(np.maximum(accessible_width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        density = (
            self.contact_spot_width(geometry, values) + meniscus + accessible_width * math.tan(contact_angle)
        ) / geometry.pore_length_ratio
        return np.where(inside, density, 0.0)
```

Заменить `src/panda_md/pores/slit/__init__.py` целиком:

`src/panda_md/pores/slit/__init__.py`:

```python
"""Щелевая пора и её формы. Импорт модулей форм регистрирует их в SlitPore."""

from panda_md.pores.slit import doughnut, droplet, perforation, roll, worm
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

__all__ = ["SlitGeometry", "SlitPore", "doughnut", "droplet", "perforation", "roll", "worm"]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `191 passed, 4 skipped`; регрессия квадрата (`filament_w1c0`) не изменилась.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/pores tests/unit
git commit -m "Щель: червь в системе поры; площади валика; формулы кругового сегмента — общие для пор" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 4: Слой — форма без контактного угла

**Files:**
- Modify: `src/panda_md/core/shape.py` (`default_optimize` в контракте), `src/panda_md/core/model.py` (`default_optimize` из формы; энергия без θ), `src/panda_md/core/fitting.py` (проверка «параметр влияет» с несколькими шагами), `src/panda_md/pores/slit/__init__.py`, `tests/unit/test_slit_rectangular_cell.py`, `tests/unit/test_model.py`, `tests/unit/test_fitting.py`
- Create: `src/panda_md/pores/slit/layer.py`

**Interfaces:**
- Consumes: задачи 1–3.
- Produces: `InterfaceShape.default_optimize: ClassVar[tuple[str, ...]] = ("theta", "delta")`; `InterfaceModel.default_optimize` — из формы, отфильтрованный по параметрам модели; `InterfaceModel.capillary_energy` — для формы без θ возвращает S_LV + S_SL; `fitting.IDENTIFIABILITY_STEP_FRACTIONS = (1e-3, 1e-2, 5e-2)` вместо одного шага; `SlitLayer` (`"layer"`; параметры δ, φ; `regimes = {}`; `default_optimize = ("phi",)`; `phi_limits = (0, 1 − 2δ)`; профиль — ступенька |z| < φ/2; площадь 2 k l²).

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_slit_rectangular_cell.py` целиком (добавлен слой):

`tests/unit/test_slit_rectangular_cell.py`:

```python
"""Щель с основанием L × kL против независимых формул Shistko et al., Mol. Simul. (2025) при θ = 180°, δ = 0.

Формулы статьи (уравнения (6)–(27)) записаны здесь заново, в лоб. Пределы φ перфорации в тексте статьи
неоднозначны (уравнение (23)), поэтому они выведены здесь интегрированием объёма отверстия: при d = 1 отверстие
закрывается (φ_max), при d = kl касается своего образа (φ_min).
"""

import math

import numpy as np
import pytest

from panda_md import SlitPore

ASPECT_RATIOS = (0.2, 0.6, 1.0)
LENGTH_RATIOS = (2.0, 4.0)
COORDINATE = np.linspace(-0.4975, 0.4975, 200)
TOLERANCE = 1e-12
FULL_NON_WETTING = 180.0


def pore(aspect_ratio: float, length_ratio: float) -> SlitPore:
    return SlitPore(width_nm=1.0, length_nm=length_ratio, base_aspect_ratio=aspect_ratio)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_droplet_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    droplet = pore(aspect_ratio, length_ratio).interface("droplet")
    upper = min(math.pi / (6 * aspect_ratio * length_ratio**2), math.pi * aspect_ratio**2 * length_ratio / 6)  # (7)
    assert droplet.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = (3 * phi * aspect_ratio * length_ratio**2 / (4 * math.pi)) ** (1 / 3)  # (6)
    center = -0.5 + radius  # при θ = 180° сфера касается нижней стенки
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        math.pi / (aspect_ratio * length_ratio**2) * (radius**2 - (COORDINATE - center) ** 2),  # (8)
        0.0,
    )
    np.testing.assert_allclose(droplet.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = droplet.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(4 * math.pi * radius**2)  # (9)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_doughnut_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    doughnut = pore(aspect_ratio, length_ratio).interface("doughnut")
    side = aspect_ratio * length_ratio
    lower = math.pi / (6 * aspect_ratio * length_ratio**2)  # (11)
    upper = math.pi / (4 * aspect_ratio * length_ratio**2) * ((side - 1) ** 2 + math.pi * (side - 1) / 2 + 2 / 3)
    assert doughnut.phi_limits(theta=FULL_NON_WETTING, phi=0.1) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    spot = -math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * phi * aspect_ratio * length_ratio**2 / math.pi)
    expected = math.pi / (4 * aspect_ratio * length_ratio**2) * (np.sqrt(1 - 4 * COORDINATE**2) + spot) ** 2  # (12)
    np.testing.assert_allclose(doughnut.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_worm_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    worm = pore(aspect_ratio, length_ratio).interface("worm")
    upper = min(math.pi / (4 * length_ratio), math.pi * length_ratio / 4)  # (15)
    assert worm.phi_limits(theta=FULL_NON_WETTING, phi=0.01) == pytest.approx((0.0, upper), abs=TOLERANCE)
    phi = 0.5 * upper
    radius = math.sqrt(phi * length_ratio / math.pi)  # (14)
    center = -0.5 + radius
    expected = np.where(
        np.abs(COORDINATE - center) < radius,
        2 / length_ratio * np.sqrt(np.maximum(radius**2 - (COORDINATE - center) ** 2, 0.0)),  # (16)
        0.0,
    )
    np.testing.assert_allclose(worm.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = worm.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
        2 * math.pi * aspect_ratio * length_ratio * radius  # (17)
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_roll_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    roll = pore(aspect_ratio, length_ratio).interface("roll")
    lower, upper = math.pi / (4 * length_ratio), 1 - (1 - math.pi / 4) / length_ratio  # (19)
    assert roll.phi_limits(theta=FULL_NON_WETTING, phi=0.5) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    spot = phi * length_ratio - math.pi / 4  # (18)
    expected = (np.sqrt(1 - 4 * COORDINATE**2) + spot) / length_ratio  # (20)
    np.testing.assert_allclose(roll.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE)
    terms = roll.surface_terms(theta=FULL_NON_WETTING, phi=phi)
    assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
        aspect_ratio * length_ratio * (2 * spot + math.pi)  # (21)
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
@pytest.mark.parametrize("length_ratio", LENGTH_RATIOS)
def test_perforation_matches_rectcell(aspect_ratio: float, length_ratio: float) -> None:
    perforation = pore(aspect_ratio, length_ratio).interface("perforation")
    side = aspect_ratio * length_ratio
    base = aspect_ratio * length_ratio**2
    lower = 1 - math.pi / (4 * base) * (side**2 - math.pi * side / 2 + 2 / 3)  # отверстие касается образа
    upper = 1 - math.pi / (4 * base) * (5 / 3 - math.pi / 2)  # отверстие закрывается
    assert perforation.phi_limits(theta=FULL_NON_WETTING, phi=0.9) == pytest.approx((lower, upper), abs=TOLERANCE)
    phi = 0.5 * (lower + upper)
    hole = math.pi / 4 + math.sqrt((math.pi / 4) ** 2 - 2 / 3 + 4 * (1 - phi) * base / math.pi)  # (22)
    expected = 1 - math.pi / (4 * base) * (np.sqrt(1 - 4 * COORDINATE**2) - hole) ** 2  # (24)
    np.testing.assert_allclose(
        perforation.profile(COORDINATE, theta=FULL_NON_WETTING, phi=phi), expected, atol=TOLERANCE
    )


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
def test_layer_matches_rectcell(aspect_ratio: float) -> None:
    layer = pore(aspect_ratio, 3.0).interface("layer")
    expected = np.where(np.abs(COORDINATE) < 0.3, 1.0, 0.0)  # (26)
    np.testing.assert_array_equal(layer.profile(COORDINATE, phi=0.6), expected)
    assert layer.surface_terms(phi=0.6).liquid_liquid_area == pytest.approx(2 * aspect_ratio * 9.0)  # (27)


@pytest.mark.parametrize("shape_name", ["droplet", "doughnut"])
def test_axisymmetric_shape_in_rectangle_equals_square_of_short_side(shape_name: str) -> None:
    """ρ(z; l, k, φ) = k · ρ(z; kl, 1, φ/k): ячейка L × kL эквивалентна квадратной kL × kL."""
    rectangle = SlitPore(width_nm=1.0, length_nm=5.0, base_aspect_ratio=0.4).interface(shape_name)
    square = SlitPore(width_nm=1.0, length_nm=2.0).interface(shape_name)
    values = {"theta": 140.0, "delta": 0.03}
    lower, upper = rectangle.phi_limits(**values, phi=0.1)
    phi = 0.5 * (lower + upper)
    np.testing.assert_allclose(
        rectangle.profile(COORDINATE, **values, phi=phi),
        0.4 * square.profile(COORDINATE, **values, phi=phi / 0.4),
        atol=TOLERANCE,
    )


def test_pore_volume_uses_rectangular_base() -> None:
    assert SlitPore(width_nm=2.0, length_nm=10.0, base_aspect_ratio=0.3).volume_nm3() == pytest.approx(60.0)
```

Заменить `tests/unit/test_model.py` целиком (энергия слоя, полный набор форм):

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


def test_square_pore_exposes_its_shapes() -> None:
    assert set(SquarePore.shape_names()) == {"filament_w1c0", "droplet_w1c0"}
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

Заменить `tests/unit/test_fitting.py` целиком (в конце — фит слоя):

`tests/unit/test_fitting.py`:

```python
"""Общий оптимизатор: правила параметров, проверки входа, флаги результата, режимы смачивания.

Профили синтетические без шума: точное восстановление — признак того, что механизм фита верен.
"""

import numpy as np
import pytest

from panda_md import DensityProfile, FitResult, Frame, Regime, SlitPore, SquarePore
from panda_md.core.errors import (
    EmptyWindowError,
    InfeasibleParametersError,
    InvalidBoundsError,
    MissingParameterValueError,
    ProfileMismatchError,
    UnidentifiableParameterError,
    UnsupportedRegimeError,
)
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
ROLL_TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}


def roll_profile(slit_pore: SlitPore, **changes: float) -> DensityProfile:
    values = {**ROLL_TRUTH, **changes}
    return synthetic_profile(slit_pore.interface("roll"), values, density_bulk_per_nm3=DENSITY_BULK)


@pytest.mark.parametrize(
    ("optimize", "fixed"),
    [
        (["phi"], {"theta": 120.0, "delta": 0.03}),
        (["delta", "phi"], {"theta": 120.0}),
        (["theta", "delta", "phi"], {}),
    ],
)
def test_one_optimizer_handles_one_two_three_free_parameters(
    slit_pore: SlitPore, optimize: list[str], fixed: dict[str, float]
) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore), optimize=optimize, fixed=fixed, density_bulk_per_nm3=DENSITY_BULK
    )
    assert result.free_parameters == tuple(name for name in ("theta", "delta", "phi") if name in optimize)
    for name, value in fixed.items():
        assert result.values[name] == value  # фиксированные не меняются
    # δ задаётся краем профиля и определяется с точностью до полуширины бина (1/200/2 = 0.0025),
    assert result.values["theta"] == pytest.approx(120.0, abs=0.5)
    assert result.values["delta"] == pytest.approx(0.03, abs=0.0025)
    assert result.values["phi"] == pytest.approx(0.4, abs=0.003)


def test_default_optimize_is_theta_and_delta(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK)
    assert result.free_parameters == ("theta", "delta")
    assert result.volume_fraction_from_data == pytest.approx(0.4, abs=1e-3)
    assert result.wall_layer_thickness_nm == pytest.approx(0.03 * 8.0, abs=1e-3)


def test_result_flags_parameter_at_bound(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore, theta=170.0),
        bounds={"theta": (100.0, 150.0)},
        density_bulk_per_nm3=DENSITY_BULK,
    )
    assert result.values["theta"] == pytest.approx(150.0, abs=0.1)
    assert "theta" in result.at_bounds


def test_result_flags_regime_boundary(square_pore: SquarePore) -> None:
    filament = square_pore.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 90.2, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    result = filament.fit(profile, fixed={"phi": 0.05}, regime=Regime.NON_WETTING, density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(90.5, abs=0.01)  # θ ∈ (89.5°, 90.5°) не считается
    assert result.at_regime_boundary
    assert result.at_bounds == ()


def test_summary_mentions_flags(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore, theta=170.0), bounds={"theta": (100.0, 150.0)}, density_bulk_per_nm3=DENSITY_BULK
    )
    assert "у границы: theta" in str(result)


def test_result_round_trips_through_json(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK)
    restored = FitResult.model_validate_json(result.model_dump_json())
    assert restored.values == result.values
    assert restored.spec == result.spec


def test_empty_window_is_an_error(slit_pore: SlitPore) -> None:
    wrong_pore = SlitPore(width_nm=0.08, length_nm=0.16)  # ширина поры в 100 раз меньше, чем у данных
    with pytest.raises(EmptyWindowError, match="размер поры"):
        wrong_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK)


def test_mean_normalization_requires_phi(slit_pore: SlitPore) -> None:
    with pytest.raises(MissingParameterValueError, match="phi"):
        slit_pore.interface("roll").fit(roll_profile(slit_pore))


def test_infeasible_fixed_values_are_reported(slit_pore: SlitPore) -> None:
    with pytest.raises(InfeasibleParametersError, match="область существования"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), fixed={"phi": 0.999}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_fixed_angle_in_undeclared_regime_is_an_error(slit_pore: SlitPore) -> None:
    with pytest.raises(UnsupportedRegimeError, match="wetting"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), optimize=["delta"], fixed={"theta": 60.0}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_parameter_without_influence_is_rejected(slit_pore: SlitPore) -> None:
    droplet = slit_pore.interface("droplet")
    profile = synthetic_profile(
        droplet, {"theta": 130.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK, frame=Frame.LIQUID
    )
    with pytest.raises(UnidentifiableParameterError, match="delta"):
        droplet.fit(profile, density_bulk_per_nm3=DENSITY_BULK, fixed={"phi": 0.05})


def test_angle_range_across_90_is_split_by_regime(square_pore: SquarePore) -> None:
    filament = square_pore.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 60.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    result = filament.fit(profile, density_bulk_per_nm3=DENSITY_BULK, fixed={"phi": 0.05})
    assert result.regime is Regime.WETTING
    assert result.values["theta"] == pytest.approx(60.0, abs=0.05)
    assert [alternative.regime for alternative in result.alternatives] == [Regime.NON_WETTING]
    assert result.alternatives[0].loss > result.loss


def test_requested_regime_restricts_search(square_pore: SquarePore) -> None:
    filament = square_pore.interface("filament_w1c0")
    profile = synthetic_profile(
        filament, {"theta": 60.0, "delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
    )
    result = filament.fit(profile, density_bulk_per_nm3=DENSITY_BULK, fixed={"phi": 0.05}, regime=Regime.NON_WETTING)
    assert result.regime is Regime.NON_WETTING
    assert result.alternatives == ()
    assert 90.0 < result.values["theta"] <= 180.0


def test_synthetic_profile_refuses_nonexistent_shape(square_pore: SquarePore) -> None:
    with pytest.raises(InfeasibleParametersError):
        synthetic_profile(
            square_pore.interface("filament_w1c0"),
            {"theta": 60.0, "delta": 0.02, "phi": 0.2},
            density_bulk_per_nm3=DENSITY_BULK,
        )


def test_user_initial_point_is_used(slit_pore: SlitPore) -> None:
    result = slit_pore.interface("roll").fit(
        roll_profile(slit_pore), initial={"theta": 125.0}, starts_per_parameter=1, density_bulk_per_nm3=DENSITY_BULK
    )
    assert result.start_count == 2
    assert np.isclose(result.values["theta"], 120.0, atol=0.05)


def test_fixed_angle_inside_regime_gap_is_an_error(slit_pore: SlitPore) -> None:
    with pytest.raises(UnsupportedRegimeError, match="зазора"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), optimize=["delta"], fixed={"theta": 90.0}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_fixed_value_outside_parameter_bounds_is_rejected(slit_pore: SlitPore) -> None:
    with pytest.raises(InvalidBoundsError, match="delta"):
        slit_pore.interface("roll").fit(
            roll_profile(slit_pore), optimize=["theta"], fixed={"delta": -0.05}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_volume_fraction_from_data_outside_bounds_names_likely_causes(slit_pore: SlitPore) -> None:
    with pytest.raises(InvalidBoundsError, match="ρ_bulk"):
        slit_pore.interface("roll").fit(roll_profile(slit_pore), density_bulk_per_nm3=DENSITY_BULK / 10.0)


def test_pore_wider_than_data_is_rejected() -> None:
    data_pore = SlitPore(width_nm=8.0, length_nm=16.0)
    wrong_pore = SlitPore(width_nm=10.0, length_nm=20.0)  # l = L/H то же, H на 25 % больше
    with pytest.raises(ProfileMismatchError, match="не покрывают"):
        wrong_pore.interface("roll").fit(roll_profile(data_pore), density_bulk_per_nm3=DENSITY_BULK)


def test_profile_in_other_units_is_rejected(slit_pore: SlitPore) -> None:
    profile = roll_profile(slit_pore)
    in_angstrom = profile.model_copy(update={"coordinate_nm": profile.coordinate_nm * 10.0})
    with pytest.raises(ProfileMismatchError, match="за пределами поры"):
        slit_pore.interface("roll").fit(in_angstrom, density_bulk_per_nm3=DENSITY_BULK)


def test_liquid_frame_profile_wider_than_pore_is_rejected(square_pore: SquarePore) -> None:
    droplet = square_pore.interface("droplet_w1c0")
    values = {"theta": 140.0, "delta": 0.02, "phi": 0.05}
    profile = synthetic_profile(droplet, values, density_bulk_per_nm3=DENSITY_BULK, frame=Frame.LIQUID)
    stretched = profile.model_copy(update={"coordinate_nm": profile.coordinate_nm * 3.0})
    with pytest.raises(ProfileMismatchError, match="шире поры"):
        droplet.fit(
            stretched, optimize=["theta"], fixed={"delta": 0.02, "phi": 0.05}, density_bulk_per_nm3=DENSITY_BULK
        )


def test_shape_without_contact_angle_fits_volume_fraction(slit_pore: SlitPore) -> None:
    layer = slit_pore.interface("layer")
    profile = synthetic_profile(layer, {"delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=DENSITY_BULK)
    result = layer.fit(profile, optimize=["phi"], density_bulk_per_nm3=DENSITY_BULK)
    assert layer.default_optimize == ("phi",)
    assert result.regime is None
    assert result.values["phi"] == pytest.approx(0.4, abs=0.006)  # край ступеньки — с точностью до бина
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q`
Expected: падают тесты слоя с `KeyError: "в поре slit нет формы layer; ..."` и `test_square_pore_exposes_its_shapes` (нет `layer`).

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/core/shape.py`, `src/panda_md/core/model.py`, `src/panda_md/core/fitting.py` целиком:

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
from panda_md.core.errors import NotDerivedError
from panda_md.core.parameters import Parameter, ParameterValues
from panda_md.core.types import FloatArray, PandaModel

REGIME_BOUNDARY_DEG = 90.0  # граница смачивания и несмачивания
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
```

`src/panda_md/core/model.py`:

```python
"""Форма, подставленная в конкретную пору, — объект, который фитируют."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from panda_md.core.enums import LossFunction, Normalization, OptimizerMethod, ParameterName, Regime
from panda_md.core.errors import MissingParameterValueError
from panda_md.core.parameters import PROFILE_SHIFT, ParameterSet
from panda_md.core.profile import DensityProfile
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray

if TYPE_CHECKING:
    from panda_md.core.pore import Pore
    from panda_md.core.result import FitResult


class InterfaceModel:
    """Связка «пора + форма». Значения параметров на входе методов — в единицах пользователя."""

    pore: "Pore"
    # Any: форма параметризована геометрией своей поры; согласованность гарантирует реестр поры.
    shape: InterfaceShape[Any]
    parameters: ParameterSet

    def __init__(self, pore: "Pore", shape: InterfaceShape[Any]) -> None:
        self.pore = pore
        self.shape = shape
        self.parameters = ParameterSet([*shape.parameters, PROFILE_SHIFT])

    def __repr__(self) -> str:
        return f"InterfaceModel(pore={self.pore.name}, shape={self.shape.name})"

    @property
    def default_optimize(self) -> tuple[str, ...]:
        return tuple(name for name in self.shape.default_optimize if name in self.parameters.names)

    def internal_values(self, user_values: Mapping[str, float]) -> dict[str, float]:
        """Значения всех параметров во внутренних единицах; пропущенные — по умолчанию."""
        self.parameters.require_known(user_values)
        resolved: dict[str, float] = {}
        for name in self.parameters.names:
            parameter = self.parameters[name]
            if name in user_values:
                resolved[name] = parameter.to_internal(user_values[name])
            elif parameter.default is not None:
                resolved[name] = parameter.to_internal(parameter.default)
            else:
                raise MissingParameterValueError(f"задай значение параметра {name}")
        return resolved

    def profile(self, coordinate: ArrayLike, **user_values: float) -> FloatArray:
        """Модельный ρ/ρ_bulk в системе отсчёта формулы формы."""
        values = self.internal_values(user_values)
        return self.shape.profile(np.asarray(coordinate, dtype=np.float64), self.pore.geometry(), values)

    def is_feasible(self, **user_values: float) -> bool:
        return bool(self.shape.feasible(self.pore.geometry(), self.internal_values(user_values)))

    def phi_limits(self, **user_values: float) -> tuple[float, float]:
        return self.shape.phi_limits(self.pore.geometry(), self.internal_values(user_values))

    def surface_terms(self, **user_values: float) -> SurfaceTerms:
        return self.shape.surface_terms(self.pore.geometry(), self.internal_values(user_values))

    def capillary_energy(self, **user_values: float) -> float:
        values = self.internal_values(user_values)
        terms = self.shape.surface_terms(self.pore.geometry(), values)
        if ParameterName.CONTACT_ANGLE not in values:
            return terms.liquid_liquid_area + terms.liquid_solid_area  # формы без θ не касаются стенок: S_SL = 0
        return terms.capillary_energy(values[ParameterName.CONTACT_ANGLE])

    def fit(
        self,
        profile: DensityProfile,
        *,
        optimize: Sequence[str] | None = None,
        fixed: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        initial: Mapping[str, float] | None = None,
        normalization: Normalization = Normalization.AUTO,
        density_bulk_per_nm3: float | None = None,
        loss: LossFunction = LossFunction.L1,
        window: tuple[float, float] | None = None,
        starts_per_parameter: int = 5,
        optimizer: OptimizerMethod = OptimizerMethod.NELDER_MEAD,
        regime: Regime | None = None,
    ) -> "FitResult":
        from panda_md.core.fitting import fit
        from panda_md.core.spec import FitSpec

        spec = FitSpec(
            optimize=None if optimize is None else tuple(optimize),
            fixed=dict(fixed or {}),
            bounds=dict(bounds or {}),
            initial=dict(initial or {}),
            normalization=normalization,
            density_bulk_per_nm3=density_bulk_per_nm3,
            loss=loss,
            window=window,
            starts_per_parameter=starts_per_parameter,
            optimizer=optimizer,
            regime=regime,
        )
        return fit(self, profile, spec)
```

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
    PandaError,
    ProfileMismatchError,
    UnidentifiableParameterError,
    UnsupportedRegimeError,
)
from panda_md.core.parameters import FreeSpec, ParameterValues
from panda_md.core.profile import DensityProfile, ReducedProfile
from panda_md.core.result import FitResult
from panda_md.core.shape import REGIME_BOUNDARY_DEG, PoreGeometry, regime_of
from panda_md.core.spec import FitSpec
from panda_md.core.types import BoolArray, FloatArray

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel

REGIME_GAP_DEG = 0.5  # θ ∈ (90° − зазор, 90° + зазор) не считается: в формулах tan θ и 1/cos θ
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

    def __init__(self, regime: Regime | None, free_spec: FreeSpec, optimum: _LocalOptimum, start_count: int) -> None:
        self.regime = regime
        self.free_spec = free_spec
        self.optimum = optimum
        self.start_count = start_count


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
        return [(None, free_spec)]
    if CONTACT_ANGLE_NAME not in free_spec.free_names:
        contact_angle_deg = math.degrees(free_spec.fixed_values[CONTACT_ANGLE_NAME])
        if abs(contact_angle_deg - REGIME_BOUNDARY_DEG) < REGIME_GAP_DEG:
            raise UnsupportedRegimeError(
                f"θ = {contact_angle_deg:.2f}° внутри зазора ±{REGIME_GAP_DEG}° у 90°: формулы там не определены"
            )
        regime = regime_of(contact_angle_deg)
        if regime not in shape.regimes or (requested is not None and requested is not regime):
            raise UnsupportedRegimeError(
                f"форма {shape.name}: режим {regime.value} не объявлен (формулы не выведены) или не запрошен"
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
    if not free_spec.free_names:
        empty = np.zeros(0, dtype=np.float64)
        return _RegimeOutcome(
            regime, free_spec, _LocalOptimum(empty, objective(empty), True, "нет свободных параметров"), 1
        )

    starts = [
        point
        for point in _start_points(free_spec, spec.starts_per_parameter)
        if objective.is_feasible(free_spec.unpack(point))
    ]
    if not starts:
        raise InfeasibleParametersError(
            f"форма {objective.model.shape.name}, режим {regime.value if regime else '—'}: ни одна стартовая точка "
            f"не попадает в область существования; проверь фиксированные значения и границы"
        )
    _check_identifiable(objective, free_spec, starts[0])
    box: Bounds = Bounds(free_spec.lower_bounds, free_spec.upper_bounds)
    best: _LocalOptimum | None = None
    for start in starts:
        optimum = minimize(objective, start, method=method.value, bounds=box)
        if best is None or float(optimum.fun) < best.loss:
            best = _LocalOptimum(
                np.asarray(optimum.x, dtype=np.float64), float(optimum.fun), bool(optimum.success), str(optimum.message)
            )
    assert best is not None  # стартов не меньше одного
    return _RegimeOutcome(regime, free_spec, best, len(starts))


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
        evaluation_count=objective.evaluation_count,
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


def fit(model: "InterfaceModel", profile: DensityProfile, spec: FitSpec) -> FitResult:
    """Подобрать свободные параметры модели под профиль. Алгоритм — спецификация §4."""
    geometry = model.pore.geometry()
    normalization = _resolve_normalization(spec)
    reduced = model.pore.reduce(profile, normalization, spec.density_bulk_per_nm3)
    window_mask = _window_mask(reduced, geometry, spec.window)
    _check_profile_matches_pore(reduced, geometry)
    optimize = spec.optimize if spec.optimize is not None else model.default_optimize

    values_from_data: dict[str, float] = {}
    volume_fraction_from_data: float | None = None
    phi_is_open = VOLUME_FRACTION_NAME not in spec.fixed and VOLUME_FRACTION_NAME not in optimize
    if normalization is Normalization.BULK and VOLUME_FRACTION_NAME in model.parameters.names and phi_is_open:
        volume_fraction_from_data = float(trapezoid(reduced.density, reduced.coordinate) / geometry.profile_integral())
        values_from_data[VOLUME_FRACTION_NAME] = volume_fraction_from_data

    free_spec = model.parameters.resolve(optimize, spec.fixed, spec.bounds, spec.initial, values_from_data)
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

`src/panda_md/pores/slit/layer.py`:

```python
"""Слой (layer) в щели: плоский слой жидкости посередине, не касается стенок.

Профиль — ступенька ширины φ (Semenchuk et al., 2025; alpha.py); граница существования φ ≤ 1 − 2δ — alpha.py;
площадь — две плоские поверхности основания, S = 2 k l² (Shistko et al., 2025, уравнение (27)).
Контактного угла у слоя нет: формула от θ не зависит, поэтому θ не параметр этой формы, а свободна по умолчанию
только φ. От k профиль не зависит.
"""

from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SlitGeometry, SlitPore


@SlitPore.register
class SlitLayer(InterfaceShape[SlitGeometry]):
    name: ClassVar[str] = "layer"
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}  # контактного угла нет
    frame: ClassVar[Frame] = Frame.PORE
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        return 0.0, 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        lower, upper = self.phi_limits(geometry, values)
        return bool(lower < values[ParameterName.VOLUME_FRACTION] <= upper)

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # слой посередине щели

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        return SurfaceTerms(liquid_liquid_area=2.0 * geometry.base_area(), liquid_solid_area=0.0)

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 1 при |z| < φ/2, иначе 0."""
        half_thickness = 0.5 * values[ParameterName.VOLUME_FRACTION]
        return np.where(np.abs(coordinate) < half_thickness, 1.0, 0.0)
```

Заменить `src/panda_md/pores/slit/__init__.py` целиком:

`src/panda_md/pores/slit/__init__.py`:

```python
"""Щелевая пора и её формы. Импорт модулей форм регистрирует их в SlitPore."""

from panda_md.pores.slit import doughnut, droplet, layer, perforation, roll, worm
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

__all__ = ["SlitGeometry", "SlitPore", "doughnut", "droplet", "layer", "perforation", "roll", "worm"]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `208 passed, 4 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests/unit
git commit -m "Щель: слой — форма без контактного угла; default_optimize в контракте формы" \
  -m "Проверка «параметр влияет» пробует шаги 0.1 %, 1 % и 5 % границ: φ слоя двигает только край ступеньки." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 5: Регрессия всех шести форм щели на старую PANDA

Проверочная задача: формулы уже перенесены в задачах 1–4, здесь фиксируется их совпадение со старой PANDA до 1e-12 на сетке l, θ, δ, φ. Тест проходит сразу; если не проходит — перенос формулы неверен, разбирать по `superpowers:systematic-debugging`.

**Files:**
- Modify: `tests/regression/generate_slit_reference.py`, `tests/regression/test_reference_formulas.py`
- Regenerate: `tests/regression/data/slit_reference.json`

**Interfaces:**
- Consumes: шесть форм щели.
- Produces: эталон — по 108 случаев на droplet, doughnut, worm, roll, perforation и 9 на layer.

- [ ] **Шаг 1: Генератор и тест**

Заменить `tests/regression/generate_slit_reference.py` целиком:

`tests/regression/generate_slit_reference.py`:

```python
"""Эталон щели из старой PANDA (`interface/alpha.py`, v0.2.0, коммит 93c6dbb), основание квадратное (k = 1).

Запуск (нужно окружение со старой PANDA, см. ROADMAP «Фаза 0»):
    .venv-legacy/bin/python tests/regression/generate_slit_reference.py
Результат: tests/regression/data/slit_reference.json — коммитится, тесты читают только его.

Капля и червь в старом коде записаны от центра сферы / цилиндра (флаг center=False); в новой версии — в системе
поры, форма лежит на слое δ над стенкой z = −1/2. Поэтому их старый профиль считается в точках z − z_c,
где z_c = (−1/2 + δ) − R cos θ.
"""

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

alpha = importlib.import_module("panda.interface.alpha")

OUTPUT = Path(__file__).parent / "data" / "slit_reference.json"
COORDINATE = np.linspace(-0.4975, 0.4975, 200)  # центры 200 бинов: ни одна точка не лежит на стенке или краю δ
LENGTH_RATIOS = (1.0, 2.0, 3.5)
ANGLES_DEG = (100.0, 120.0, 150.0, 179.0)
WALL_LAYERS = (0.0, 0.03, 0.08)
PHI_POSITIONS = (0.2, 0.5, 0.8)  # положение φ внутри интервала существования
LAYER_ANGLE = math.radians(120.0)  # у слоя θ не участвует в формулах; alpha.py требует аргумент


def lying_shape_profile(shape: str, length_ratio: float, phi: float, angle: float, wall_layer: float) -> list[float]:
    """Капля или червь: старый профиль от центра сферы/цилиндра, перенесённый на опору у нижней стенки."""
    radius = float(getattr(alpha, f"r_{shape}_alpha")(length_ratio, phi, angle, wall_layer))
    center = (-0.5 + wall_layer) - radius * math.cos(angle)
    profile = getattr(alpha, f"rho_{shape}_alpha")(
        COORDINATE - center, length_ratio, phi, angle, wall_layer, center=False
    )
    return [float(value) for value in np.asarray(profile, dtype=float)]


def centered_shape_profile(shape: str, length_ratio: float, phi: float, angle: float, wall_layer: float) -> list[float]:
    profile = getattr(alpha, f"rho_{shape}_alpha")(COORDINATE, length_ratio, phi, angle, wall_layer)
    return [float(value) for value in np.asarray(profile, dtype=float)]


PROFILES: dict[str, Callable[[str, float, float, float, float], list[float]]] = {
    "roll": centered_shape_profile,
    "doughnut": centered_shape_profile,
    "perforation": centered_shape_profile,
    "droplet": lying_shape_profile,
    "worm": lying_shape_profile,
}


def shape_cases(shape: str) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for length_ratio in LENGTH_RATIOS:
        for angle_deg in ANGLES_DEG:
            for wall_layer in WALL_LAYERS:
                angle = math.radians(angle_deg)
                lower = float(getattr(alpha, f"phi_min_{shape}_alpha")(length_ratio, angle, wall_layer))
                upper = float(getattr(alpha, f"phi_max_{shape}_alpha")(length_ratio, angle, wall_layer))
                low, high = max(lower, 0.0), min(upper, 1.0)
                if not low < high:
                    continue  # форма не существует при этих l, θ, δ
                for position in PHI_POSITIONS:
                    phi = low + position * (high - low)
                    cases.append(
                        {
                            "length_ratio": length_ratio,
                            "theta": angle_deg,
                            "delta": wall_layer,
                            "phi": phi,
                            "phi_limits": [lower, upper],
                            "profile": PROFILES[shape](shape, length_ratio, phi, angle, wall_layer),
                        }
                    )
    return cases


def layer_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for wall_layer in WALL_LAYERS:
        upper = float(alpha.phi_max_layer_alpha(1.0, LAYER_ANGLE, wall_layer))
        for position in PHI_POSITIONS:
            phi = position * upper
            profile = alpha.rho_layer_alpha(COORDINATE, 1.0, phi, LAYER_ANGLE, wall_layer)
            cases.append(
                {
                    "length_ratio": 1.0,
                    "delta": wall_layer,
                    "phi": phi,
                    "phi_limits": [0.0, upper],
                    "profile": np.asarray(profile, dtype=float).tolist(),
                }
            )
    return cases


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reference: dict[str, object] = {
        "source": "github.com/FlufffyMelon/PANDA@93c6dbb panda/interface/alpha.py",
        "coordinate": COORDINATE.tolist(),
    }
    for shape in PROFILES:
        reference[shape] = shape_cases(shape)
    reference["layer"] = layer_cases()
    OUTPUT.write_text(json.dumps(reference))
    counts = ", ".join(f"{shape}: {len(reference[shape])}" for shape in (*PROFILES, "layer"))  # type: ignore[arg-type]
    print(f"записано в {OUTPUT}: {counts}")


if __name__ == "__main__":
    main()
```

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

from panda_md import SlitPore, SquarePore

DATA = Path(__file__).parent / "data"
TOLERANCE = 1e-12
SLIT_WIDTH_NM = 1.0


def load(name: str) -> dict[str, Any]:  # Any: структура JSON эталона описана в генераторе
    reference: dict[str, Any] = json.loads((DATA / name).read_text())
    return reference


SLIT = load("slit_reference.json")
SQUARE = load("square_reference.json")


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


@pytest.mark.parametrize("shape_name", ["filament_w1c0", "droplet_w1c0"])
def test_square_matches_panda_polygon(shape_name: str) -> None:
    coordinate = np.array(SQUARE["coordinate"])
    for case in SQUARE[shape_name]:
        side_nm = 1.0
        pore = SquarePore(
            side_nm=side_nm,
            length_nm=case["length_ratio"] * side_nm,
            layer_width_nm=SQUARE["layer_width_ratio"] * side_nm,
        )
        model = pore.interface(shape_name)
        values = {"theta": case["theta"], "delta": case["delta"], "phi": case["phi"]}
        assert model.is_feasible(**values) == case["feasible"], case_id(case)
        if not case["feasible"]:
            continue
        np.testing.assert_allclose(
            model.profile(coordinate, **values), case["profile"], rtol=0.0, atol=TOLERANCE, err_msg=case_id(case)
        )
        terms = model.surface_terms(**values)
        assert model.capillary_energy(**values) == pytest.approx(case["energy"], abs=TOLERANCE), case_id(case)
        assert terms.liquid_liquid_area + terms.liquid_solid_area == pytest.approx(
            case["area_sum"], abs=TOLERANCE
        ), case_id(case)
```

- [ ] **Шаг 2: Сгенерировать эталон**

Run: `.venv-legacy/bin/python tests/regression/generate_slit_reference.py`
Expected: `записано в .../slit_reference.json: roll: 108, doughnut: 108, perforation: 108, droplet: 108, worm: 108, layer: 9`.

- [ ] **Шаг 3: Тесты проходят**

Run: `uv run pytest -q tests/regression`
Expected: `8 passed`. Полный набор: `212 passed, 4 skipped`.

- [ ] **Шаг 4: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add tests/regression
git commit -m "Регрессия: все шесть форм щели совпадают со старой PANDA до 1e-12" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 6: Синтетика новых форм, описание тестов, документация стадии

Проверочная задача для синтетики (код написан в задачах 1–4).

**Files:**
- Modify: `tests/synthetic/test_recovery.py`, `tests/README.md`, `docs/superpowers/specs/2026-10-01-panda-architecture-design.md`, `ROADMAP.md`, `README.md`

- [ ] **Шаг 1: Синтетика**

Заменить `tests/synthetic/test_recovery.py` целиком (добавлены пончик, червь, перфорация, капля и валик при k = 0.4):

`tests/synthetic/test_recovery.py`:

```python
"""Синтетическая валидация: форма → профиль с шумом → фит → восстановленные параметры.

Пороги и найденные закономерности описаны в tests/README.md («Синтетика», «O2», «O3»).
"""

import numpy as np
import pytest

from panda_md import Frame, SlitPore, SquarePore
from panda_md.core.model import InterfaceModel
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0
NOISE_STD = 0.005  # шум 0.5 % ρ_bulk на бин, 200 бинов
SEEDS = range(5)
MAX_ANGLE_ERROR_DEG = 1.5  # в каждом прогоне
MEAN_ANGLE_ERROR_DEG = 1.0  # в среднем по прогонам (критерий ТЗ |Δθ| < 1°)

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)
SQUARE = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
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
Expected: `24 passed`. Пороги не ослаблять: падение — расхождение с `tests/README.md`, разбирать.

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
| параметры | `unit/test_parameters.py` | оптимизация | описание параметра (границы, умолчание, лишние поля), градусы ↔ радианы, разбор `optimize` / `fixed` / `bounds` / `initial`, упаковка в вектор и обратно | стандартные θ, δ, φ, shift | ошибки с понятным текстом; порядок параметров сохраняется | точное равенство |
| профиль | `unit/test_profile.py` | данные | проверка массивов (длины, NaN, возрастание координаты, знак плотности), неизменяемость, JSON, приведение по ρ_bulk и по средней плотности | ровный профиль 50 точек | безразмерная координата в долях H, плотность / ρ_bulk или / n̄ | 1e-12 |
| реестр | `unit/test_pore_registry.py` | расширяемость | форма, объявленная в тесте, подключается декоратором без правки ядра; повтор — ошибка; у каждого типа поры свой реестр | `FlatLayer`, `OtherPore` из теста | форма видна только своей поре | — |
| модель | `unit/test_model.py` | математика | параметры модели, единицы, ошибки имён, профиль валика в центре поры против формулы, допустимость по интервалу φ, капиллярный функционал (в т. ч. слоя без θ), набор форм каждой поры | щель 8 × 16 нм, квадрат 6 × 12 нм | совпадение с формулой | 1e-12 |
| ячейка L × kL | `unit/test_slit_rectangular_cell.py` | математика | все шесть форм щели против формул Shistko et al. (2025) при θ = 180°, δ = 0: размеры, пределы φ, профиль, площадь; ячейка L × kL для осесимметричной формы = квадратная kL × kL; объём поры | k ∈ {0.2, 0.6, 1}, l ∈ {2, 4} | совпадение с формулами статьи (пределы перфорации — выведены интегрированием, в тексте статьи уравнение (23) неоднозначно) | 1e-12 |
| контракт | `contract/test_shapes.py` | математика | для **каждой** зарегистрированной формы каждой поры (щель — и с k = 0.4): атрибуты контракта (у формы без θ режимов нет); допустимые точки в каждом объявленном режиме; ∫ρ dz = φ · (A / ly); 0 ≤ ρ ≤ 1 и ρ = 0 вне поры; центр масс формулы = численному; S_LV > 0, S_SL ≥ 0 | сетка θ (по 3 угла в режиме), δ ∈ {0, 0.03}, 25 значений φ | все проверки на всех допустимых точках | ∫ — 1e-4 (ступенька на краю), центр масс — 1e-4 |
| регрессия | `regression/test_reference_formulas.py` | математика | перенос формул без изменений: шесть форм щели (k = 1), две формы квадрата | эталоны из старой PANDA (`alpha.py`, 93c6dbb) и PANDA_Polygon (`square.py`, 38ff491); генераторы `regression/generate_*.py` | профиль, интервал φ, условие существования, энергия и сумма площадей совпадают | 1e-12 |
| фит | `unit/test_fitting.py` | оптимизация | один оптимизатор при 1, 2, 3 свободных параметрах; фиксированные не меняются; флаги «у границы» и «у 90°»; пустое окно, профиль не соответствует поре (пора шире данных, профиль в других единицах, профиль в системе жидкости шире поры), фиксированное значение или φ из данных вне границ, нет φ, нет допустимого старта, не объявленный режим, θ в зазоре у 90°, параметр без влияния — ошибки; деление по режимам смачивания | синтетика без шума, 200 бинов | восстановление в пределах дискретизации | θ — 0.5°, δ — 0.0025 (полбина), φ — 0.003 |
| синтетика | `synthetic/test_recovery.py` | физическая модель | полная цепочка «форма → профиль с шумом → фит» для 10 случаев (пять форм щели — все, кроме слоя: валик, капля, пончик, червь, перфорация; капля и валик при k = 0.4; капля и нить квадрата, нить при 60°); три режима ТЗ; нормировка по средней плотности (O2); ошибка положения стенок (O3) | шум 0.5 % ρ_bulk на бин, 5 зерён | см. ниже | максимум 1.5°, среднее 1° |

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
````

- [ ] **Шаг 3: Спецификация** (точечные замены через Edit)

В таблице решений после строки D17 добавить:

```markdown
| D18 | Щель с основанием L × kL: k — поле `SlitPore`; осесимметричные формы — через эквивалентную квадратную ячейку kL × kL, червь и валик — вдоль короткой стороны, от k не зависят (стадия 2) |
| D19 | Форма может не иметь θ (слой): тогда `regimes = {}`, свои `default_optimize`; капиллярная энергия — S_LV + S_SL |
```

§3.2: в блоке `class InterfaceShape` после строки `frame: ClassVar[Frame] ...` добавить строку
`    default_optimize: ClassVar[tuple[str, ...]] = ("theta", "delta")  # у формы без θ — свои`.

§3.3: `SlitPore(H, L, k=1.0, walls=None, axis="z")` → `SlitPore(width_nm, length_nm, base_aspect_ratio=1.0)` (стенки и ось — фаза 5, путь данных).

§4, правило 1: `1. \`optimize\` не задан → \`model.default_optimize\` (для щели — θ, δ).` →
`1. \`optimize\` не задан → \`model.default_optimize\` из формы (θ, δ; у слоя — φ).`

§11, строки O5 и O7 →

```markdown
| O5 | Площади S для общего (θ, δ) в щели — выведены для капли, червя, валика и слоя (геометрия, сверено с RectCell при θ = 180°); у пончика и перфорации — нет | вывести поверхности тора; влияет только на `stability` |
| O7 | ~~Обобщение k ≠ 1 на произвольные θ, δ~~ — сделано: осесимметричные формы через квадратную ячейку kL × kL, червь и валик от k не зависят; проверено контрактом с k = 0.4 и формулами RectCell | закрыт стадией 2 |
```

- [ ] **Шаг 4: ROADMAP и README**

`ROADMAP.md`, «Фаза 4. Щель полностью»: первой строкой списка добавить
`- **Статус: выполнено** — план \`docs/superpowers/plans/2026-10-16-stage2-slit-complete.md\`; 217 тестов; выводы — \`tests/README.md\` (стадия 2).`

`README.md`, блок «Status: pre-alpha»: `works for the slit pore\n> (\`roll\`, \`droplet\`)` → `works for the slit pore\n> (all six shapes: droplet, doughnut, worm, roll, perforation, layer; rectangular base L × kL)`.

- [ ] **Шаг 5: Полная проверка и коммит**

Run: `make check`, `make test-env`, `make wheel-check`; на Python 3.10: `UV_PROJECT_ENVIRONMENT=.venv-py310 uv run --no-sync pytest -q`. Всё зелёное, `217 passed, 4 skipped`.

```bash
git add tests docs ROADMAP.md README.md
git commit -m "Стадия 2 закрыта: синтетика новых форм, описание тестов, спецификация и ROADMAP" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
