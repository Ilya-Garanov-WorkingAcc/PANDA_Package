# Стадия 1 — прототип ядра PANDA: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Рабочее ядро `panda_md`: параметры, контракт формы и поры, профиль, общий фит с режимами смачивания и проверками — на двух настоящих порах (щель: `roll`, `droplet`; квадрат: `filament_w1c0`, `droplet_w1c0`), с тестами unit / contract / regression / synthetic и описанием тестов в `tests/README.md`.

**Architecture:** Слои с зависимостями только вниз: `pores/<тип>` (формулы и геометрия) → `core` (не знает ни одной поры). Форма — класс без состояния с обязательным `feasible()` и необязательными `profile()`, `phi_limits()`, `surface_terms()`; пора — модель pydantic с собственным реестром форм (`@Pore.register`); `InterfaceModel` = форма в поре; `fit()` один для любой формы и любого набора свободных параметров.

**Tech Stack:** Python ≥ 3.10, numpy, scipy (`minimize`, Nelder–Mead по умолчанию), pydantic v2, pytest, uv, black, ruff, mypy (strict + плагин pydantic).

**Spec:** `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (утверждена 15.10.2026). Разбор многоугольных пор — `docs/architecture/05_polygon_pores.md`.

**Проверено до написания плана.** Весь код ниже собран и прогнан по задачам в чистой копии репозитория: после каждой задачи `black --check`, `ruff check`, `mypy` и `pytest` проходят (итого 107 тестов, 2 пропуска — у форм щели площади не выведены). Код переносить дословно.

## Global Constraints

- Python ≥ 3.10: без `StrEnum`, `typing.Self`, вложенных одинаковых кавычек в f-строках; строковые перечисления — `class X(str, Enum)`.
- Датаклассы запрещены; объекты-данные — модели pydantic v2 от `PandaModel` (`frozen=True`, `extra="forbid"`). Решение D14.
- Любые фиксированные варианты — `Enum`; сравнения со строками в логике запрещены; числовые константы — именованные с комментарием. Решение D17, §2.1 спецификации.
- У всех полей и атрибутов классов указан тип; у функций — аннотации аргументов и результата; `Any` — только с комментарием-обоснованием.
- Имена объясняют смысл (`contact_angle`, `wall_layer`, `volume_fraction`); однобуквенные — только в формулах со ссылкой на уравнение.
- Стиль: black, длина строки 120; ruff с правилами из `pyproject.toml`; mypy strict для `src` и `tests`.
- Комментарии, docstring, сообщения ошибок — по-русски; имена в коде — по-английски.
- Импорт пакета — `panda_md`; имя на PyPI — `panda-md`; версия в этой стадии не меняется (0.0.1).
- Каждый коммит заканчивается отдельным абзацем `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`; абзацы сообщения — отдельными `-m` (в этой среде heredoc теряет пустые строки).
- Проверка перед коммитом: `make check` (black --check, ruff, mypy, pytest).

## Review Focus

Случаи, на которых пользователь вероятнее всего споткнётся; для каждого тест уже добавлен в задачу, которой принадлежит код.

1. Профиль записан в других единицах или пора задана неверного размера → в окне фита почти нет точек. Ожидание: `EmptyWindowError` с диапазоном координаты данных, а не тихий возврат стартовой точки. Тест `test_empty_window_is_an_error` — задача 6.
2. Пустая выборка атомов → профиль из одних нулей. Ожидание: ошибка «профиль пуст» при создании `DensityProfile`. Тест `test_profile_rejects_all_zero_density` — задача 2.
3. θ зафиксирован ровно на 90° (или внутри зазора ±0.5°). Ожидание: `UnsupportedRegimeError`, а не деление на cos 90°. Тест `test_fixed_angle_inside_regime_gap_is_an_error` — задача 6.
4. Ящик уже стороны квадратной поры (`layer_width_nm < side_nm`) → нормировка даёт ρ > 1. Ожидание: ошибка проверки при создании `SquarePore`. Тест `test_square_pore_rejects_layer_narrower_than_pore` — задача 5.
5. Границы на фит вне допустимых (`bounds={"theta": (190, 200)}`). Ожидание: `InvalidBoundsError`. Тест `test_resolve_rejects_bounds_outside_parameter_range` — задача 1.

---

### Задача 1: Базовые типы, перечисления, ошибки, параметры

**Files:**
- Modify: `pyproject.toml` (зависимость pydantic, плагин mypy, режим импорта pytest)
- Create: `src/panda_md/core/__init__.py`, `src/panda_md/core/types.py`, `src/panda_md/core/enums.py`, `src/panda_md/core/errors.py`, `src/panda_md/core/parameters.py`
- Test: `tests/unit/test_parameters.py`

**Interfaces:**
- Consumes: ничего (первая задача).
- Produces:
  - `panda_md.core.types`: `FloatArray = NDArray[np.float64]`, `BoolArray = NDArray[np.bool_]`, `Array1D` (аннотированный тип для полей pydantic), `PandaModel` (база моделей).
  - `panda_md.core.enums`: `ParameterName` (`CONTACT_ANGLE="theta"`, `WALL_LAYER_THICKNESS="delta"`, `VOLUME_FRACTION="phi"`, `PROFILE_SHIFT="shift"`), `Unit`, `Regime` (`WETTING`, `NON_WETTING`), `Frame` (`PORE`, `LIQUID`), `Normalization` (`AUTO`, `BULK`, `MEAN`), `LossFunction` (`L1`, `L2`), `OptimizerMethod` (`NELDER_MEAD`, `POWELL`, `LBFGSB`).
  - `panda_md.core.errors`: `PandaError` и наследники `NotDerivedError`, `UnknownParameterError`, `ParameterConflictError`, `InvalidBoundsError`, `MissingParameterValueError`, `UnidentifiableParameterError`, `InfeasibleParametersError`, `EmptyWindowError`, `UnsupportedRegimeError`, `InvalidProfileError`; `PandaWarning`.
  - `panda_md.core.parameters`: `ParameterValues = Mapping[str, float]`; `Parameter` (`name`, `default`, `bounds`, `unit`, `symbol`, `description`; `to_internal(float) -> float`, `to_user(float) -> float`); константы `CONTACT_ANGLE`, `WALL_LAYER_THICKNESS`, `VOLUME_FRACTION`, `PROFILE_SHIFT`; `FreeSpec` (`free_names`, `fixed_values`, `lower_bounds`, `upper_bounds`, `initial_values`; `pack`, `unpack`, `to_user`, `with_free_bounds`); `ParameterSet(parameters)` (`names`, `__getitem__`, `require_known`, `resolve(optimize, fixed, bounds, initial, values_from_data) -> FreeSpec`).

- [ ] **Шаг 1: Подготовить `pyproject.toml`**

В `[project] dependencies` добавить строку после `"scipy>=1.11",`:

```toml
    "pydantic>=2.6",
```

В `[tool.pytest.ini_options]` заменить `addopts`:

```toml
addopts = "-ra --strict-markers --strict-config --import-mode=importlib"
```

В `[tool.mypy]` после `warn_unreachable = true` добавить:

```toml
plugins = ["pydantic.mypy"]
```

Затем `uv sync` — в окружение поставится pydantic.

- [ ] **Шаг 2: Написать тесты параметров**

`tests/unit/test_parameters.py`:

```python
"""Параметры: проверка описания, единицы, разбор optimize / fixed / bounds / initial."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.enums import Unit
from panda_md.core.errors import (
    InvalidBoundsError,
    MissingParameterValueError,
    ParameterConflictError,
    UnknownParameterError,
)
from panda_md.core.parameters import (
    CONTACT_ANGLE,
    PROFILE_SHIFT,
    VOLUME_FRACTION,
    WALL_LAYER_THICKNESS,
    Parameter,
    ParameterSet,
)


def make_set() -> ParameterSet:
    return ParameterSet([CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION, PROFILE_SHIFT])


def test_parameter_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError):
        Parameter(name="x", default=None, bounds=(1.0, 0.0))


def test_parameter_rejects_default_outside_bounds() -> None:
    with pytest.raises(ValidationError):
        Parameter(name="x", default=2.0, bounds=(0.0, 1.0))


def test_parameter_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Parameter(name="x", default=None, bounds=(0.0, 1.0), bound=(0.0, 1.0))  # type: ignore[call-arg]


def test_parameter_is_immutable() -> None:
    with pytest.raises(ValidationError):
        CONTACT_ANGLE.default = 90.0  # type: ignore[misc]


def test_degree_parameter_converts_to_radians_and_back() -> None:
    assert CONTACT_ANGLE.unit is Unit.DEGREE
    assert CONTACT_ANGLE.to_internal(180.0) == pytest.approx(math.pi)
    assert CONTACT_ANGLE.to_user(math.pi / 2) == pytest.approx(90.0)
    assert WALL_LAYER_THICKNESS.to_internal(0.05) == 0.05


def test_resolve_splits_free_and_fixed_in_parameter_order() -> None:
    spec = make_set().resolve(["delta", "theta"], {"phi": 0.3}, {}, {}, {})
    assert spec.free_names == ("theta", "delta")
    assert spec.fixed_values == {"phi": 0.3, "shift": 0.0}
    assert spec.lower_bounds.tolist() == pytest.approx([0.0, 0.0])
    assert spec.upper_bounds.tolist() == pytest.approx([math.pi, 0.1])


def test_pack_unpack_round_trip_keeps_fixed_values() -> None:
    spec = make_set().resolve(["theta", "delta"], {"phi": 0.3}, {}, {}, {})
    values = spec.unpack(np.array([2.0, 0.04]))
    assert list(values) == ["theta", "delta", "phi", "shift"]
    assert values == {"theta": 2.0, "delta": 0.04, "phi": 0.3, "shift": 0.0}
    assert spec.pack(values).tolist() == [2.0, 0.04]


def test_resolve_rejects_unknown_name() -> None:
    with pytest.raises(UnknownParameterError, match="допустимые"):
        make_set().resolve(["thetta"], {"phi": 0.3}, {}, {}, {})


def test_resolve_rejects_parameter_both_free_and_fixed() -> None:
    with pytest.raises(ParameterConflictError):
        make_set().resolve(["theta"], {"theta": 120.0, "phi": 0.3}, {}, {}, {})


def test_resolve_requires_value_for_fixed_phi() -> None:
    with pytest.raises(MissingParameterValueError, match="phi"):
        make_set().resolve(["theta"], {}, {}, {}, {})


def test_resolve_takes_phi_from_data_when_not_given() -> None:
    spec = make_set().resolve(["theta"], {}, {}, {}, {"phi": 0.25})
    assert spec.fixed_values["phi"] == 0.25


def test_resolve_intersects_user_bounds_with_parameter_bounds() -> None:
    spec = make_set().resolve(["theta"], {"phi": 0.3}, {"theta": (100.0, 200.0)}, {}, {})
    assert spec.lower_bounds[0] == pytest.approx(math.radians(100.0))
    assert spec.upper_bounds[0] == pytest.approx(math.pi)


def test_resolve_rejects_initial_value_for_fixed_parameter() -> None:
    with pytest.raises(ParameterConflictError):
        make_set().resolve(["theta"], {"phi": 0.3}, {}, {"delta": 0.01}, {})


def test_parameter_set_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="дважды"):
        ParameterSet([CONTACT_ANGLE, CONTACT_ANGLE])


def test_resolve_rejects_bounds_outside_parameter_range() -> None:
    with pytest.raises(InvalidBoundsError, match="не пересекаются"):
        make_set().resolve(["theta"], {"phi": 0.3}, {"theta": (190.0, 200.0)}, {}, {})
```

- [ ] **Шаг 3: Убедиться, что тесты падают**

Run: `uv run pytest tests/unit/test_parameters.py -q`
Expected: ошибка сборки тестов `ModuleNotFoundError: No module named 'panda_md.core'`.

- [ ] **Шаг 4: Написать модули ядра**

`src/panda_md/core/__init__.py`:

```python
"""Ядро PANDA: параметры, контракт формы и поры, профиль, общий фит. Не знает ни одной конкретной поры."""
```

`src/panda_md/core/types.py`:

```python
"""Общая базовая модель pydantic и тип одномерного массива numpy с проверкой."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _validate_float_array(value: object) -> FloatArray:
    """Приводит вход к неизменяемому одномерному массиву float64 без NaN и бесконечностей."""
    array = np.array(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"ожидается одномерный массив, получена размерность {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError("массив содержит NaN или бесконечность")
    array.flags.writeable = False
    return array


def _serialize_float_array(array: FloatArray) -> list[float]:
    return [float(item) for item in array]


Array1D = Annotated[
    FloatArray,
    PlainValidator(_validate_float_array),
    PlainSerializer(_serialize_float_array, return_type=list[float]),
]


class PandaModel(BaseModel):
    """База всех объектов-данных: неизменяемые, без лишних полей, с проверкой значений по умолчанию."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )
```

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
    LIQUID = "liquid"  # 0 — центр масс анализируемой жидкости


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
```

`src/panda_md/core/errors.py`:

```python
"""Исключения и предупреждения библиотеки. Все исключения наследуют PandaError."""


class PandaError(Exception):
    """Базовое исключение PANDA."""


class NotDerivedError(PandaError):
    """Величина для этой формы не выведена (нет формулы)."""


class UnknownParameterError(PandaError):
    """Имя параметра не принадлежит модели."""


class ParameterConflictError(PandaError):
    """Параметр одновременно свободный и фиксированный."""


class InvalidBoundsError(PandaError):
    """Границы, заданные на фит, не пересекаются с допустимыми границами параметра."""


class MissingParameterValueError(PandaError):
    """У фиксированного параметра нет значения."""


class UnidentifiableParameterError(PandaError):
    """Свободный параметр не влияет на модельный профиль."""


class InfeasibleParametersError(PandaError):
    """Ни одна стартовая точка не попадает в область существования формы."""


class EmptyWindowError(PandaError):
    """В окне фита слишком мало точек профиля."""


class UnsupportedRegimeError(PandaError):
    """Режим смачивания не объявлен формой (формулы не выведены)."""


class InvalidProfileError(PandaError):
    """Профиль плотности не подходит для анализа."""


class PandaWarning(UserWarning):
    """Предупреждение PANDA: расчёт выполнен, но результату стоит не доверять без проверки."""
```

`src/panda_md/core/parameters.py`:

```python
"""Описание параметров модели и разделение их на свободные и фиксированные."""

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from pydantic import model_validator

from panda_md.core.enums import ParameterName, Unit
from panda_md.core.errors import (
    InvalidBoundsError,
    MissingParameterValueError,
    ParameterConflictError,
    UnknownParameterError,
)
from panda_md.core.types import FloatArray, PandaModel

ParameterValues = Mapping[str, float]
"""Значения параметров по имени, во внутренних единицах (θ в радианах)."""


class Parameter(PandaModel):
    """Один параметр модели: имя, значение по умолчанию, границы, единицы (всё — в единицах пользователя)."""

    name: str
    default: float | None
    bounds: tuple[float, float]
    unit: Unit = Unit.DIMENSIONLESS
    symbol: str = ""
    description: str = ""

    @model_validator(mode="after")
    def _check_bounds_and_default(self) -> "Parameter":
        lower, upper = self.bounds
        if not self.name:
            raise ValueError("имя параметра пустое")
        if not lower < upper:
            raise ValueError(f"{self.name}: нижняя граница {lower} не меньше верхней {upper}")
        if self.default is not None and not lower <= self.default <= upper:
            raise ValueError(f"{self.name}: значение по умолчанию {self.default} вне границ {self.bounds}")
        return self

    def to_internal(self, user_value: float) -> float:
        return math.radians(user_value) if self.unit is Unit.DEGREE else float(user_value)

    def to_user(self, internal_value: float) -> float:
        return math.degrees(internal_value) if self.unit is Unit.DEGREE else float(internal_value)


CONTACT_ANGLE = Parameter(
    name=ParameterName.CONTACT_ANGLE.value,
    default=120.0,
    bounds=(0.0, 180.0),
    unit=Unit.DEGREE,
    symbol="θ",
    description="контактный угол, измеренный через анализируемую жидкость",
)
WALL_LAYER_THICKNESS = Parameter(
    name=ParameterName.WALL_LAYER_THICKNESS.value,
    default=0.0,
    bounds=(0.0, 0.1),
    unit=Unit.PORE_SIZE,
    symbol="δ",
    description="толщина слоя у стенки, недоступного анализируемой жидкости",
)
VOLUME_FRACTION = Parameter(
    name=ParameterName.VOLUME_FRACTION.value,
    default=None,
    bounds=(0.0, 1.0),
    unit=Unit.DIMENSIONLESS,
    symbol="φ",
    description="объёмная доля анализируемой жидкости в поре",
)
PROFILE_SHIFT = Parameter(
    name=ParameterName.PROFILE_SHIFT.value,
    default=0.0,
    bounds=(-0.1, 0.1),
    unit=Unit.PORE_SIZE,
    symbol="s",
    description="сдвиг модели вдоль координаты профиля (досцентрирование фитом)",
)


class FreeSpec:
    """Результат разбора: какие параметры свободны, их границы и старт, значения фиксированных."""

    parameters: dict[str, Parameter]
    free_names: tuple[str, ...]
    fixed_values: dict[str, float]
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    initial_values: dict[str, float]

    def __init__(
        self,
        parameters: dict[str, Parameter],
        free_names: tuple[str, ...],
        fixed_values: dict[str, float],
        lower_bounds: FloatArray,
        upper_bounds: FloatArray,
        initial_values: dict[str, float],
    ) -> None:
        self.parameters = parameters
        self.free_names = free_names
        self.fixed_values = fixed_values
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.initial_values = initial_values

    def pack(self, values: ParameterValues) -> FloatArray:
        """Словарь (внутренние единицы) → вектор свободных параметров."""
        return np.array([values[name] for name in self.free_names], dtype=np.float64)

    def unpack(self, vector: FloatArray) -> dict[str, float]:
        """Вектор свободных параметров → полный словарь значений (внутренние единицы)."""
        free_values = {name: float(item) for name, item in zip(self.free_names, vector, strict=True)}
        return {name: free_values[name] if name in free_values else self.fixed_values[name] for name in self.parameters}

    def to_user(self, values: ParameterValues) -> dict[str, float]:
        return {name: self.parameters[name].to_user(value) for name, value in values.items()}

    def with_free_bounds(self, name: str, lower: float, upper: float) -> "FreeSpec":
        """Копия с другими внутренними границами одного свободного параметра (для режимов смачивания)."""
        index = self.free_names.index(name)
        lower_bounds = self.lower_bounds.copy()
        upper_bounds = self.upper_bounds.copy()
        lower_bounds[index] = lower
        upper_bounds[index] = upper
        return FreeSpec(
            self.parameters, self.free_names, self.fixed_values, lower_bounds, upper_bounds, self.initial_values
        )


class ParameterSet:
    """Все параметры одной модели; разбирает optimize / fixed / bounds / initial в FreeSpec."""

    parameters: dict[str, Parameter]

    def __init__(self, parameters: Iterable[Parameter]) -> None:
        self.parameters = {}
        for parameter in parameters:
            if parameter.name in self.parameters:
                raise ValueError(f"параметр {parameter.name} объявлен дважды")
            self.parameters[parameter.name] = parameter

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.parameters)

    def __getitem__(self, name: str) -> Parameter:
        self.require_known([name])
        return self.parameters[name]

    def require_known(self, names: Iterable[str]) -> None:
        unknown = [name for name in names if name not in self.parameters]
        if unknown:
            raise UnknownParameterError(f"неизвестные параметры {unknown}; допустимые: {list(self.parameters)}")

    def resolve(
        self,
        optimize: Sequence[str],
        fixed: Mapping[str, float],
        bounds: Mapping[str, tuple[float, float]],
        initial: Mapping[str, float],
        values_from_data: Mapping[str, float],
    ) -> FreeSpec:
        """Правила спецификации §4: каждый параметр либо свободный, либо фиксированный со значением."""
        self.require_known([*optimize, *fixed, *bounds, *initial])
        conflict = sorted(set(optimize) & set(fixed))
        if conflict:
            raise ParameterConflictError(f"параметры {conflict} одновременно в optimize и fixed")
        free_names = tuple(name for name in self.parameters if name in optimize)

        fixed_values: dict[str, float] = {}
        for name, parameter in self.parameters.items():
            if name in free_names:
                continue
            if name in fixed:
                user_value = fixed[name]
            elif name in values_from_data:
                user_value = parameter.to_user(values_from_data[name])
            elif parameter.default is not None:
                user_value = parameter.default
            else:
                raise MissingParameterValueError(f"задай значение {name} в fixed или освободи его в optimize")
            fixed_values[name] = parameter.to_internal(user_value)

        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        for name in free_names:
            parameter = self.parameters[name]
            lower, upper = parameter.bounds
            if name in bounds:
                lower, upper = max(lower, bounds[name][0]), min(upper, bounds[name][1])
                if not lower < upper:
                    raise InvalidBoundsError(f"{name}: границы {bounds[name]} не пересекаются с {parameter.bounds}")
            lower_bounds.append(parameter.to_internal(lower))
            upper_bounds.append(parameter.to_internal(upper))

        not_free = sorted(set(initial) - set(free_names))
        if not_free:
            raise ParameterConflictError(f"стартовые значения заданы для несвободных параметров {not_free}")
        initial_values = {name: self.parameters[name].to_internal(value) for name, value in initial.items()}
        return FreeSpec(
            self.parameters,
            free_names,
            fixed_values,
            np.array(lower_bounds, dtype=np.float64),
            np.array(upper_bounds, dtype=np.float64),
            initial_values,
        )
```

- [ ] **Шаг 5: Тесты проходят**

Run: `uv run pytest tests/unit/test_parameters.py -q`
Expected: `15 passed`.

- [ ] **Шаг 6: Полная проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add pyproject.toml src/panda_md/core tests/unit/test_parameters.py
git commit -m "Ядро: базовые типы, перечисления, ошибки, параметры" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 2: Контракт формы, профиль, пора с реестром, модель «форма в поре»

**Files:**
- Create: `src/panda_md/core/shape.py`, `src/panda_md/core/profile.py`, `src/panda_md/core/pore.py`, `src/panda_md/core/model.py`, `src/panda_md/pores/__init__.py`, `src/panda_md/pores/slit/pore.py`
- Test: `tests/unit/test_profile.py`, `tests/unit/test_pore_registry.py`

**Interfaces:**
- Consumes: всё из задачи 1.
- Produces:
  - `panda_md.core.shape`: `REGIME_BOUNDARY_DEG = 90.0`; `PoreGeometry` (абстрактная модель: `coordinate_bounds() -> tuple[float, float]`, `profile_integral() -> float`); `SurfaceTerms` (`liquid_liquid_area`, `liquid_solid_area`; `capillary_energy(contact_angle_rad) -> float`); `regime_of(contact_angle_deg) -> Regime`; `InterfaceShape[GeometryT]` (ClassVar `name`, `parameters`, `regimes`, `frame`; абстрактный `feasible(geometry, values) -> bool`; необязательные `profile(coordinate, geometry, values)`, `phi_limits(geometry, values)`, `surface_terms(geometry, values)` — по умолчанию `NotDerivedError`; `center_of_mass(geometry, values)` — по умолчанию численно; `has_phi_limits()`, `has_profile()`).
  - `panda_md.core.profile`: `DensityProfile` (`coordinate_nm`, `density_per_nm3`, `frame`, `particle_count`, `pore_volume_nm3`; `mean_density_per_nm3()`), `ReducedProfile` (`coordinate`, `density`, `frame`, `normalization`, `density_scale_per_nm3`), `reduce_profile(profile, characteristic_size_nm, normalization, density_bulk_per_nm3) -> ReducedProfile`.
  - `panda_md.core.pore`: `Pore` (абстрактная модель; ClassVar `name`; `own_registry()`, `register(shape_class)` — декоратор, `shape_names()`; абстрактные `geometry()`, `characteristic_size_nm()`, `volume_nm3()`; `reduce(profile, normalization, density_bulk_per_nm3)`, `interface(shape_name) -> InterfaceModel`, `available_interfaces()`).
  - `panda_md.core.model`: `InterfaceModel(pore, shape)` (`pore`, `shape`, `parameters`; `default_optimize`, `internal_values(user_values) -> dict[str, float]`, `profile(coordinate, **user_values)`, `is_feasible(**user_values)`, `phi_limits(**user_values)`, `surface_terms(**user_values)`, `capillary_energy(**user_values)`). Метод `fit` добавляется в задаче 6.
  - `panda_md.pores.slit.pore`: `SLIT_HALF_WIDTH = 0.5`, `SlitGeometry(pore_length_ratio)`, `SlitPore(width_nm, length_nm)`.

- [ ] **Шаг 1: Написать тесты профиля и реестра**

`tests/unit/test_profile.py`:

```python
"""Профиль плотности и его приведение к безразмерному виду."""

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.enums import Frame, Normalization
from panda_md.core.errors import InvalidProfileError
from panda_md.core.profile import DensityProfile
from panda_md.pores.slit.pore import SlitPore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


def make_profile(**changes: object) -> DensityProfile:
    fields: dict[str, object] = {
        "coordinate_nm": np.linspace(-4.0, 4.0, 50),
        "density_per_nm3": np.full(50, 10.0),
        "particle_count": 1000.0,
        "pore_volume_nm3": 2048.0,
    }
    fields.update(changes)
    return DensityProfile.model_validate(fields)


def test_profile_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError, match="длины"):
        make_profile(density_per_nm3=np.ones(49))


def test_profile_rejects_nan() -> None:
    density = np.full(50, 10.0)
    density[3] = np.nan
    with pytest.raises(ValidationError, match="NaN"):
        make_profile(density_per_nm3=density)


def test_profile_rejects_decreasing_coordinate() -> None:
    with pytest.raises(ValidationError, match="возрастает"):
        make_profile(coordinate_nm=np.linspace(4.0, -4.0, 50))


def test_profile_rejects_negative_density() -> None:
    with pytest.raises(ValidationError, match="отрицательна"):
        make_profile(density_per_nm3=np.full(50, -1.0))


def test_profile_arrays_are_read_only() -> None:
    profile = make_profile()
    with pytest.raises(ValueError, match="read-only"):
        profile.density_per_nm3[0] = 1.0


def test_profile_round_trips_through_json() -> None:
    profile = make_profile(frame=Frame.LIQUID)
    restored = DensityProfile.model_validate_json(profile.model_dump_json())
    assert restored.frame is Frame.LIQUID
    np.testing.assert_array_equal(restored.coordinate_nm, profile.coordinate_nm)


def test_slit_reduces_by_width_and_bulk_density() -> None:
    reduced = SLIT.reduce(make_profile(), Normalization.BULK, 20.0)
    assert reduced.coordinate[0] == pytest.approx(-0.5)
    assert reduced.density[0] == pytest.approx(0.5)


def test_slit_reduces_by_mean_density() -> None:
    reduced = SLIT.reduce(make_profile(), Normalization.MEAN, None)
    assert reduced.density_scale_per_nm3 == pytest.approx(1000.0 / 2048.0)


def test_bulk_normalization_needs_bulk_density() -> None:
    with pytest.raises(InvalidProfileError, match="density_bulk"):
        SLIT.reduce(make_profile(), Normalization.BULK, None)


def test_mean_normalization_needs_particle_count() -> None:
    with pytest.raises(InvalidProfileError, match="particle_count"):
        SLIT.reduce(make_profile(particle_count=None), Normalization.MEAN, None)


def test_profile_rejects_all_zero_density() -> None:
    with pytest.raises(ValidationError, match="профиль пуст"):
        make_profile(density_per_nm3=np.zeros(50))
```

`tests/unit/test_pore_registry.py`:

```python
"""Реестр форм поры: форма подключается без правки ядра, у каждого типа поры свой реестр."""

from collections.abc import Iterator
from typing import ClassVar

import pytest
from pydantic import ValidationError

from panda_md.core.enums import Frame, Regime
from panda_md.core.parameters import CONTACT_ANGLE, Parameter, ParameterValues
from panda_md.core.pore import Pore
from panda_md.core.shape import InterfaceShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


class FlatLayer(InterfaceShape[SlitGeometry]):
    """Форма, объявленная прямо в тесте: так же её мог бы объявить пользователь в своём скрипте."""

    name: ClassVar[str] = "test_layer"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE,)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        return True


class OtherPore(Pore):
    name: ClassVar[str] = "test_other"
    size_nm: float = 1.0

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=1.0)

    def characteristic_size_nm(self) -> float:
        return self.size_nm

    def volume_nm3(self) -> float:
        return self.size_nm**3


@pytest.fixture
def registered_layer() -> Iterator[type[FlatLayer]]:
    SlitPore.register(FlatLayer)
    yield FlatLayer
    del SlitPore.own_registry()["test_layer"]


def test_new_shape_registers_without_core_changes(registered_layer: type[FlatLayer]) -> None:
    assert "test_layer" in [model.shape.name for model in SLIT.available_interfaces()]
    model = SLIT.interface("test_layer")
    assert isinstance(model.shape, registered_layer)
    assert model.parameters.names == ("theta", "shift")


def test_duplicate_registration_is_rejected(registered_layer: type[FlatLayer]) -> None:
    with pytest.raises(ValueError, match="уже зарегистрирована"):
        SlitPore.register(registered_layer)


def test_registry_belongs_to_pore_type(registered_layer: type[FlatLayer]) -> None:
    assert "test_layer" in SlitPore.shape_names()
    assert "test_layer" not in OtherPore.shape_names()


def test_unknown_shape_lists_available() -> None:
    with pytest.raises(KeyError, match="доступны"):
        SLIT.interface("no_such_shape")


def test_pore_rejects_non_positive_size() -> None:
    with pytest.raises(ValidationError):
        SlitPore(width_nm=0.0, length_nm=10.0)


def test_pore_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SlitPore(width_nm=8.0, length_nm=16.0, height_nm=3.0)  # type: ignore[call-arg]
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/unit/test_profile.py tests/unit/test_pore_registry.py -q`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.core.profile'`.

- [ ] **Шаг 3: Написать контракт формы, профиль, пору, модель и щелевую пору**

`src/panda_md/core/shape.py`:

```python
"""Контракт формы интерфейса: только формулы, без состояния, данных и оптимизатора."""

import math
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

import numpy as np
from pydantic import Field
from scipy.integrate import trapezoid

from panda_md.core.enums import Frame, Regime
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

`src/panda_md/core/profile.py`:

```python
"""Профиль плотности из данных и его безразмерный вид."""

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, Normalization
from panda_md.core.errors import InvalidProfileError
from panda_md.core.types import Array1D, PandaModel

MIN_PROFILE_POINTS = 2


class DensityProfile(PandaModel):
    """Профиль плотности анализируемой жидкости вдоль оси профиля поры."""

    coordinate_nm: Array1D  # центры бинов, нм, в системе frame
    density_per_nm3: Array1D  # число частиц на нм³
    frame: Frame = Frame.PORE
    particle_count: float | None = Field(default=None, gt=0.0)  # N — частиц в поре
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)  # V — объём поры, к которому отнесена φ

    @model_validator(mode="after")
    def _check_arrays(self) -> "DensityProfile":
        if self.coordinate_nm.shape != self.density_per_nm3.shape:
            raise ValueError("длины координаты и плотности различаются")
        if self.coordinate_nm.size < MIN_PROFILE_POINTS:
            raise ValueError(f"в профиле меньше {MIN_PROFILE_POINTS} точек")
        if not np.all(np.diff(self.coordinate_nm) > 0):
            raise ValueError("координата профиля не возрастает")
        if np.any(self.density_per_nm3 < 0):
            raise ValueError("плотность отрицательна")
        if not np.any(self.density_per_nm3 > 0):
            raise ValueError("профиль пуст: плотность везде нулевая — проверь выборку атомов")
        return self

    def mean_density_per_nm3(self) -> float:
        """n̄ = N / V — средняя плотность жидкости в поре."""
        if self.particle_count is None or self.pore_volume_nm3 is None:
            raise InvalidProfileError("для нормировки mean нужны particle_count и pore_volume_nm3")
        return self.particle_count / self.pore_volume_nm3


class ReducedProfile(PandaModel):
    """Профиль в безразмерном виде: координата в долях размера поры, плотность после нормировки."""

    coordinate: Array1D
    density: Array1D
    frame: Frame
    normalization: Normalization
    density_scale_per_nm3: float = Field(gt=0.0)  # на что поделена плотность: ρ_bulk или n̄


def reduce_profile(
    profile: DensityProfile,
    characteristic_size_nm: float,
    normalization: Normalization,
    density_bulk_per_nm3: float | None,
) -> ReducedProfile:
    if normalization is Normalization.BULK:
        if density_bulk_per_nm3 is None:
            raise InvalidProfileError("для нормировки bulk нужна объёмная плотность density_bulk")
        scale = density_bulk_per_nm3
    elif normalization is Normalization.MEAN:
        scale = profile.mean_density_per_nm3()
    else:
        raise ValueError(f"нормировка {normalization.value} должна быть разрешена до приведения профиля")
    return ReducedProfile(
        coordinate=profile.coordinate_nm / characteristic_size_nm,
        density=profile.density_per_nm3 / scale,
        frame=profile.frame,
        normalization=normalization,
        density_scale_per_nm3=scale,
    )
```

`src/panda_md/core/pore.py`:

```python
"""Тип поры: геометрия, приведение профиля к безразмерному виду, реестр форм."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from panda_md.core.enums import Normalization
from panda_md.core.profile import DensityProfile, ReducedProfile, reduce_profile
from panda_md.core.shape import InterfaceShape, PoreGeometry
from panda_md.core.types import PandaModel

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel

# Any: реестр хранит формы с разными подклассами геометрии; конкретный тип известен только самой поре.
ShapeClassT = TypeVar("ShapeClassT", bound=type[InterfaceShape[Any]])


class Pore(PandaModel, ABC):
    """Конкретная пора с размерами. Каждый подкласс имеет свой реестр форм."""

    name: ClassVar[str]
    shape_registry: ClassVar[dict[str, type[InterfaceShape[Any]]]]

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

    def reduce(
        self, profile: DensityProfile, normalization: Normalization, density_bulk_per_nm3: float | None
    ) -> ReducedProfile:
        return reduce_profile(profile, self.characteristic_size_nm(), normalization, density_bulk_per_nm3)

    def interface(self, shape_name: str) -> "InterfaceModel":
        from panda_md.core.model import InterfaceModel

        registry = self.own_registry()
        if shape_name not in registry:
            raise KeyError(f"в поре {self.name} нет формы {shape_name}; доступны: {list(registry)}")
        return InterfaceModel(self, registry[shape_name]())

    def available_interfaces(self) -> list["InterfaceModel"]:
        return [self.interface(shape_name) for shape_name in self.own_registry()]
```

`src/panda_md/core/model.py`:

```python
"""Форма, подставленная в конкретную пору, — объект, который фитируют."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from panda_md.core.enums import ParameterName
from panda_md.core.errors import MissingParameterValueError
from panda_md.core.parameters import PROFILE_SHIFT, ParameterSet
from panda_md.core.shape import InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray

if TYPE_CHECKING:
    from panda_md.core.pore import Pore

DEFAULT_FREE_PARAMETERS = (ParameterName.CONTACT_ANGLE.value, ParameterName.WALL_LAYER_THICKNESS.value)


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
        return tuple(name for name in DEFAULT_FREE_PARAMETERS if name in self.parameters.names)

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
        return terms.capillary_energy(values[ParameterName.CONTACT_ANGLE])
```

`src/panda_md/pores/__init__.py`:

```python
"""Типы пор и их формы интерфейса. Каждая пора — отдельный подпакет."""
```

`src/panda_md/pores/slit/pore.py`:

```python
"""Щелевая пора: две плоские стенки на расстоянии H, квадратное основание L × L."""

from typing import ClassVar

from pydantic import Field

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry

SLIT_HALF_WIDTH = 0.5  # стенки щели в безразмерной координате: z = ±0.5


class SlitGeometry(PoreGeometry):
    """Безразмерная геометрия щели: l = L / H (Semenchuk et al., 2025)."""

    pore_length_ratio: float = Field(gt=0.0)  # l = L / H

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SLIT_HALF_WIDTH, SLIT_HALF_WIDTH)

    def profile_integral(self) -> float:
        return 1.0  # ∫ρ dz = φ: профиль нормирован на всю площадь основания


class SlitPore(Pore):
    name: ClassVar[str] = "slit"

    width_nm: float = Field(gt=0.0)  # H — расстояние между стенками
    length_nm: float = Field(gt=0.0)  # L — период ячейки вдоль стенки

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=self.length_nm / self.width_nm)

    def characteristic_size_nm(self) -> float:
        return self.width_nm

    def volume_nm3(self) -> float:
        return self.length_nm**2 * self.width_nm
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `34 passed`.

- [ ] **Шаг 5: Полная проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/core src/panda_md/pores tests/unit
git commit -m "Ядро: контракт формы, профиль плотности, пора с реестром форм, модель «форма в поре»" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 3: Щель — валик (roll), контрактные тесты для всех форм

**Files:**
- Create: `src/panda_md/pores/slit/roll.py`, `src/panda_md/pores/slit/__init__.py`, `tests/contract/test_shapes.py`, `tests/unit/test_model.py`
- Modify: `src/panda_md/__init__.py` (публичный API)

**Interfaces:**
- Consumes: задачи 1–2.
- Produces: `panda_md.pores.slit.roll.SlitRoll` (зарегистрирована как `"roll"`; `contact_spot_width`, `phi_limits`, `feasible`, `center_of_mass` = 0, `profile`); `panda_md.pores.slit` экспортирует `SlitGeometry`, `SlitPore` и импортирует модули форм (импорт = регистрация); `panda_md` экспортирует `DensityProfile`, `Frame`, `InterfaceModel`, `LossFunction`, `Normalization`, `OptimizerMethod`, `PandaError`, `PandaWarning`, `Parameter`, `ParameterName`, `Regime`, `SlitPore`, `Unit`, `__version__`. Контрактный тест параметризован по реестрам всех пор из списка `PORES` — новые формы попадают в него без правки.

- [ ] **Шаг 1: Написать тесты модели и контракта**

`tests/unit/test_model.py`:

```python
"""InterfaceModel — форма в конкретной поре; значения параметров в единицах пользователя."""

import math

import numpy as np
import pytest

from panda_md.core.errors import MissingParameterValueError, NotDerivedError, UnknownParameterError
from panda_md.pores.slit import SlitPore

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
```

`tests/contract/test_shapes.py`:

```python
"""Контракт формы — проверяется автоматически для КАЖДОЙ зарегистрированной формы каждой поры.

Новая форма, подключённая декоратором @Pore.register, попадает сюда без правки тестов.
"""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import SlitPore
from panda_md.core.enums import ParameterName, Regime
from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.core.pore import Pore
from panda_md.core.shape import InterfaceShape, regime_of

PORES: tuple[Pore, ...] = (SlitPore(width_nm=8.0, length_nm=16.0),)
MODELS = [model for pore in PORES for model in pore.available_interfaces()]
INTEGRATION_GRID = np.linspace(-0.5, 0.5, 200_001)
INTEGRAL_TOLERANCE = 1e-4  # ступенька на краю профиля даёт ошибку трапеций ~ шаг сетки
CENTER_OF_MASS_TOLERANCE = 1e-4
SAMPLE_LIMIT = 12
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = np.linspace(0.02, 0.98, 25)


def regime_angles(model: InterfaceModel) -> list[float]:
    """Три угла внутри каждого объявленного режима (градусы)."""
    angles: list[float] = []
    for lower, upper in model.shape.regimes.values():
        angles.extend(np.linspace(lower, upper, 5)[1:-1].tolist())
    return angles


def feasible_samples(model: InterfaceModel) -> list[dict[str, float]]:
    """Допустимые точки (θ, δ, φ) — не больше SAMPLE_LIMIT, равномерно по сетке."""
    samples = [
        {"theta": theta, "delta": delta, "phi": float(phi)}
        for theta, delta, phi in itertools.product(regime_angles(model), WALL_LAYERS, VOLUME_FRACTIONS)
        if model.is_feasible(theta=theta, delta=delta, phi=float(phi))
    ]
    step = max(1, len(samples) // SAMPLE_LIMIT)
    return samples[::step]


def model_id(model: InterfaceModel) -> str:
    return f"{model.pore.name}/{model.shape.name}"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_declares_contract_attributes(model: InterfaceModel) -> None:
    shape = model.shape
    assert isinstance(shape, InterfaceShape)
    assert shape.name in type(model.pore).own_registry()
    assert shape.regimes, "форма должна объявить хотя бы один режим смачивания"
    for regime, (lower, upper) in shape.regimes.items():
        assert isinstance(regime, Regime)
        assert 0.0 <= lower < upper <= 180.0
    assert ParameterName.CONTACT_ANGLE in model.parameters.names


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_exists_somewhere_in_each_regime(model: InterfaceModel) -> None:
    covered = {regime_of(sample["theta"]) for sample in feasible_samples(model)}
    assert covered == set(model.shape.regimes), "в каком-то режиме нет ни одной допустимой точки — проверь feasible"


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

Run: `uv run pytest tests/unit/test_model.py tests/contract -q`
Expected: ошибка сборки `ImportError: cannot import name 'SlitPore' from 'panda_md'` (или `No module named 'panda_md.pores.slit.roll'`).

- [ ] **Шаг 3: Написать форму валика, пакет щели и публичный API**

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
from panda_md.core.shape import InterfaceShape
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

`src/panda_md/pores/slit/__init__.py`:

```python
"""Щелевая пора и её формы. Импорт модулей форм регистрирует их в SlitPore."""

from panda_md.pores.slit import roll
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

__all__ = ["SlitGeometry", "SlitPore", "roll"]
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
from panda_md.pores.slit import SlitPore

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = [
    "DensityProfile",
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
    "Unit",
    "__version__",
]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `46 passed, 1 skipped` (пропуск — `slit/roll: площади не выведены`).

- [ ] **Шаг 5: Полная проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests
git commit -m "Щель: форма roll; контрактные тесты для каждой зарегистрированной формы" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 4: Щель — капля (droplet), регрессия на старую PANDA

**Files:**
- Create: `src/panda_md/pores/common.py`, `src/panda_md/pores/slit/droplet.py`, `tests/regression/generate_slit_reference.py`, `tests/regression/data/slit_reference.json` (генерируется), `tests/regression/test_reference_formulas.py`
- Modify: `src/panda_md/pores/slit/__init__.py` (импорт модуля `droplet`)

**Interfaces:**
- Consumes: задачи 1–3.
- Produces: `panda_md.pores.common.spherical_cap_volume_factor(contact_angle) -> float`, `spherical_cap_centroid_offset(radius, contact_angle) -> float` (используются и квадратной порой); `panda_md.pores.slit.droplet.SlitDroplet` (`"droplet"`; `sphere_radius`, `sphere_center`, `phi_limits`, `feasible`, `center_of_mass`, `profile`; система отсчёта — поры, капля на слое δ над нижней стенкой). Эталон `slit_reference.json` — 108 случаев валика и 108 капли.

- [ ] **Шаг 1: Написать генератор эталона и регрессионный тест**

`tests/regression/generate_slit_reference.py`:

```python
"""Эталон щели из старой PANDA (`interface/alpha.py`, v0.2.0, коммит 93c6dbb).

Запуск (нужно окружение со старой PANDA, см. ROADMAP «Фаза 0»):
    .venv-legacy/bin/python tests/regression/generate_slit_reference.py
Результат: tests/regression/data/slit_reference.json — коммитится, тесты читают только его.

Капля в старом коде записана от центра сферы (флаг center=False); в новой версии — в системе поры,
капля стоит на слое δ над стенкой z = −1/2. Поэтому старый профиль считается в точках z − z_c,
где z_c = (−1/2 + δ) − R cos θ.
"""

import importlib
import json
import math
from pathlib import Path

import numpy as np

alpha = importlib.import_module("panda.interface.alpha")

OUTPUT = Path(__file__).parent / "data" / "slit_reference.json"
COORDINATE = np.linspace(-0.4975, 0.4975, 200)  # центры 200 бинов: ни одна точка не лежит на стенке или краю δ
LENGTH_RATIOS = (1.0, 2.0, 3.5)
ANGLES_DEG = (100.0, 120.0, 150.0, 179.0)
WALL_LAYERS = (0.0, 0.03, 0.08)
PHI_POSITIONS = (0.2, 0.5, 0.8)  # положение φ внутри интервала существования


def roll_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for length_ratio in LENGTH_RATIOS:
        for angle_deg in ANGLES_DEG:
            for wall_layer in WALL_LAYERS:
                angle = math.radians(angle_deg)
                lower = float(alpha.phi_min_roll_alpha(length_ratio, angle, wall_layer))
                upper = float(alpha.phi_max_roll_alpha(length_ratio, angle, wall_layer))
                for position in PHI_POSITIONS:
                    phi = max(lower, 0.0) + position * (min(upper, 1.0) - max(lower, 0.0))
                    profile = alpha.rho_roll_alpha(COORDINATE, length_ratio, phi, angle, wall_layer)
                    cases.append(
                        {
                            "length_ratio": length_ratio,
                            "theta": angle_deg,
                            "delta": wall_layer,
                            "phi": phi,
                            "phi_limits": [lower, upper],
                            "profile": np.asarray(profile, dtype=float).tolist(),
                        }
                    )
    return cases


def droplet_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for length_ratio in LENGTH_RATIOS:
        for angle_deg in ANGLES_DEG:
            for wall_layer in WALL_LAYERS:
                angle = math.radians(angle_deg)
                upper = float(alpha.phi_max_droplet_alpha(length_ratio, angle, wall_layer))
                for position in PHI_POSITIONS:
                    phi = position * upper
                    radius = float(alpha.r_droplet_alpha(length_ratio, phi, angle, wall_layer))
                    sphere_center = (-0.5 + wall_layer) - radius * math.cos(angle)
                    profile = alpha.rho_droplet_alpha(
                        COORDINATE - sphere_center, length_ratio, phi, angle, wall_layer, center=False
                    )
                    cases.append(
                        {
                            "length_ratio": length_ratio,
                            "theta": angle_deg,
                            "delta": wall_layer,
                            "phi": phi,
                            "phi_limits": [0.0, upper],
                            "profile": np.asarray(profile, dtype=float).tolist(),
                        }
                    )
    return cases


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    reference = {
        "source": "github.com/FlufffyMelon/PANDA@93c6dbb panda/interface/alpha.py",
        "coordinate": COORDINATE.tolist(),
        "roll": roll_cases(),
        "droplet": droplet_cases(),
    }
    OUTPUT.write_text(json.dumps(reference))
    print(f"записано {len(reference['roll'])} + {len(reference['droplet'])} случаев в {OUTPUT}")


if __name__ == "__main__":
    main()
```

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

from panda_md import SlitPore

DATA = Path(__file__).parent / "data"
TOLERANCE = 1e-12
SLIT_WIDTH_NM = 1.0


def load(name: str) -> dict[str, Any]:  # Any: структура JSON эталона описана в генераторе
    reference: dict[str, Any] = json.loads((DATA / name).read_text())
    return reference


SLIT = load("slit_reference.json")


def case_id(case: dict[str, Any]) -> str:
    return f"l={case['length_ratio']}-θ={case['theta']}-δ={case['delta']}-φ={case['phi']:.4f}"


@pytest.mark.parametrize("shape_name", ["roll", "droplet"])
def test_slit_matches_old_panda(shape_name: str) -> None:
    coordinate = np.array(SLIT["coordinate"])
    for case in SLIT[shape_name]:
        pore = SlitPore(width_nm=SLIT_WIDTH_NM, length_nm=case["length_ratio"] * SLIT_WIDTH_NM)
        model = pore.interface(shape_name)
        values = {"theta": case["theta"], "delta": case["delta"], "phi": case["phi"]}
        np.testing.assert_allclose(
            model.profile(coordinate, **values), case["profile"], rtol=0.0, atol=TOLERANCE, err_msg=case_id(case)
        )
        np.testing.assert_allclose(
            model.phi_limits(**values), case["phi_limits"], rtol=0.0, atol=TOLERANCE, err_msg=case_id(case)
        )
```

- [ ] **Шаг 2: Сгенерировать эталон**

Нужно окружение со старой PANDA (`.venv-legacy`, см. ROADMAP «Фаза 0»):

Run: `.venv-legacy/bin/python tests/regression/generate_slit_reference.py`
Expected: `записано 108 + 108 случаев в .../tests/regression/data/slit_reference.json`.

- [ ] **Шаг 3: Убедиться, что тест падает**

Run: `uv run pytest tests/regression -q`
Expected: `test_slit_matches_old_panda[droplet]` падает с `KeyError: 'в поре slit нет формы droplet; доступны: ['roll']'`; `[roll]` проходит.

- [ ] **Шаг 4: Написать каплю и общий модуль сферического сегмента**

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
```

`src/panda_md/pores/slit/droplet.py`:

```python
"""Капля (droplet) в щели: сферический сегмент на нижней стенке.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (1), (3)–(5); границы существования —
по старой PANDA (`interface/alpha.py`, ветка θ ≥ 90°). Профиль записан в системе поры: капля стоит на слое δ
над нижней стенкой z = −1/2.
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import FloatArray
from panda_md.pores.common import spherical_cap_centroid_offset, spherical_cap_volume_factor
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore


@SlitPore.register
class SlitDroplet(InterfaceShape[SlitGeometry]):
    name: ClassVar[str] = "droplet"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def sphere_radius(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """R из π R³ (2 − 3 cos θ + cos³ θ) / 3 = φ l², уравнение (1)."""
        volume = values[ParameterName.VOLUME_FRACTION] * geometry.pore_length_ratio**2
        return float(
            np.cbrt(3.0 * volume / (math.pi * spherical_cap_volume_factor(values[ParameterName.CONTACT_ANGLE])))
        )

    def sphere_center(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        return floor - self.sphere_radius(geometry, values) * math.cos(values[ParameterName.CONTACT_ANGLE])

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ ≤ min(капля не выше доступной щели, пятно не шире ячейки) — уравнение (3) и alpha.py."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]
        length_ratio = geometry.pore_length_ratio
        by_height = (
            math.pi
            * accessible_width**3
            * (2.0 + math.cos(contact_angle))
            / (3.0 * length_ratio**2 * (1.0 - math.cos(contact_angle)))
        )
        by_length = math.pi * length_ratio * (2.0 + math.cos(contact_angle)) * math.sin(0.5 * contact_angle) ** 4 / 6.0
        return 0.0, min(by_height, by_length)

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS] <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return lower < values[ParameterName.VOLUME_FRACTION] <= upper

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        radius = self.sphere_radius(geometry, values)
        return self.sphere_center(geometry, values) + spherical_cap_centroid_offset(
            radius, values[ParameterName.CONTACT_ANGLE]
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = π (R² − (z − z_c)²) / l² между опорой капли и её верхушкой, уравнение (5)."""
        radius = self.sphere_radius(geometry, values)
        center = self.sphere_center(geometry, values)
        floor = -SLIT_HALF_WIDTH + values[ParameterName.WALL_LAYER_THICKNESS]
        inside = (coordinate > floor) & (coordinate < center + radius)  # строгие границы, как в alpha.py
        section_area = math.pi * np.maximum(radius**2 - (coordinate - center) ** 2, 0.0)
        return np.where(inside, section_area / geometry.pore_length_ratio**2, 0.0)
```

Заменить `src/panda_md/pores/slit/__init__.py` целиком:

`src/panda_md/pores/slit/__init__.py`:

```python
"""Щелевая пора и её формы. Импорт модулей форм регистрирует их в SlitPore."""

from panda_md.pores.slit import droplet, roll
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

__all__ = ["SlitGeometry", "SlitPore", "droplet", "roll"]
```

- [ ] **Шаг 5: Тесты проходят**

Run: `uv run pytest -q`
Expected: `53 passed, 2 skipped`; контрактные тесты включили `slit/droplet` без правки тестов.

- [ ] **Шаг 6: Полная проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/pores tests/regression
git commit -m "Щель: форма droplet в системе поры; регрессия на старую PANDA (alpha.py, 93c6dbb)" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 5: Квадратная пора — filament_w1c0 и droplet_w1c0, регрессия на PANDA_Polygon

**Files:**
- Create: `src/panda_md/pores/square/pore.py`, `src/panda_md/pores/square/filament.py`, `src/panda_md/pores/square/droplet.py`, `src/panda_md/pores/square/__init__.py`, `tests/regression/generate_square_reference.py`, `tests/regression/data/square_reference.json` (генерируется)
- Modify: `src/panda_md/__init__.py` (экспорт `SquarePore`), `tests/regression/test_reference_formulas.py` (тест квадрата), `tests/unit/test_model.py` (тесты квадрата), `tests/contract/test_shapes.py` (квадрат в `PORES`)

**Interfaces:**
- Consumes: задачи 1–4 (`pores.common`, контракт формы, `Pore`).
- Produces: `panda_md.pores.square.pore`: `SQUARE_HALF_SIDE`, `SQUARE_SECTION_AREA`, `accessible_side(wall_layer) -> float`, `SquareGeometry(pore_length_ratio, layer_width_ratio)`, `SquarePore(side_nm, length_nm, layer_width_nm)` (проверка: слой не уже стороны). Формы: `SquareFilamentOneWall` (`"filament_w1c0"`, режимы смачивания и несмачивания; `arc_radius`, `arc_center`, `feasible`, `center_of_mass`, `surface_terms`, `profile`), `SquareDropletOneWall` (`"droplet_w1c0"`, только несмачивание; `sphere_radius`, `sphere_center`, `feasible`, `center_of_mass`, `surface_terms`, `profile`). Эталон — по 150 случаев на форму.

- [ ] **Шаг 1: Дописать тесты**

Заменить `tests/unit/test_model.py` целиком (добавлены три теста квадрата в конце и импорты):

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
    assert set(SlitPore.shape_names()) == {"roll", "droplet"}


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

Заменить `tests/contract/test_shapes.py` целиком (квадрат в `PORES`):

`tests/contract/test_shapes.py`:

```python
"""Контракт формы — проверяется автоматически для КАЖДОЙ зарегистрированной формы каждой поры.

Новая форма, подключённая декоратором @Pore.register, попадает сюда без правки тестов.
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
    SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0),
)
MODELS = [model for pore in PORES for model in pore.available_interfaces()]
INTEGRATION_GRID = np.linspace(-0.5, 0.5, 200_001)
INTEGRAL_TOLERANCE = 1e-4  # ступенька на краю профиля даёт ошибку трапеций ~ шаг сетки
CENTER_OF_MASS_TOLERANCE = 1e-4
SAMPLE_LIMIT = 12
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = np.linspace(0.02, 0.98, 25)


def regime_angles(model: InterfaceModel) -> list[float]:
    """Три угла внутри каждого объявленного режима (градусы)."""
    angles: list[float] = []
    for lower, upper in model.shape.regimes.values():
        angles.extend(np.linspace(lower, upper, 5)[1:-1].tolist())
    return angles


def feasible_samples(model: InterfaceModel) -> list[dict[str, float]]:
    """Допустимые точки (θ, δ, φ) — не больше SAMPLE_LIMIT, равномерно по сетке."""
    samples = [
        {"theta": theta, "delta": delta, "phi": float(phi)}
        for theta, delta, phi in itertools.product(regime_angles(model), WALL_LAYERS, VOLUME_FRACTIONS)
        if model.is_feasible(theta=theta, delta=delta, phi=float(phi))
    ]
    step = max(1, len(samples) // SAMPLE_LIMIT)
    return samples[::step]


def model_id(model: InterfaceModel) -> str:
    return f"{model.pore.name}/{model.shape.name}"


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_declares_contract_attributes(model: InterfaceModel) -> None:
    shape = model.shape
    assert isinstance(shape, InterfaceShape)
    assert shape.name in type(model.pore).own_registry()
    assert shape.regimes, "форма должна объявить хотя бы один режим смачивания"
    for regime, (lower, upper) in shape.regimes.items():
        assert isinstance(regime, Regime)
        assert 0.0 <= lower < upper <= 180.0
    assert ParameterName.CONTACT_ANGLE in model.parameters.names


@pytest.mark.parametrize("model", MODELS, ids=model_id)
def test_shape_exists_somewhere_in_each_regime(model: InterfaceModel) -> None:
    covered = {regime_of(sample["theta"]) for sample in feasible_samples(model)}
    assert covered == set(model.shape.regimes), "в каком-то режиме нет ни одной допустимой точки — проверь feasible"


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

Заменить `tests/regression/test_reference_formulas.py` целиком (эталон квадрата и тест):

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
    return f"l={case['length_ratio']}-θ={case['theta']}-δ={case['delta']}-φ={case['phi']:.4f}"


@pytest.mark.parametrize("shape_name", ["roll", "droplet"])
def test_slit_matches_old_panda(shape_name: str) -> None:
    coordinate = np.array(SLIT["coordinate"])
    for case in SLIT[shape_name]:
        pore = SlitPore(width_nm=SLIT_WIDTH_NM, length_nm=case["length_ratio"] * SLIT_WIDTH_NM)
        model = pore.interface(shape_name)
        values = {"theta": case["theta"], "delta": case["delta"], "phi": case["phi"]}
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

Создать генератор:

`tests/regression/generate_square_reference.py`:

```python
"""Эталон квадратной поры из PANDA_Polygon (`docs/model/square.py`, коммит 38ff491).

Запуск из корня репозитория (нужна локальная копия PANDA_Polygon/, см. ROADMAP «Фаза 4а»):
    uv run python tests/regression/generate_square_reference.py
Результат: tests/regression/data/square_reference.json — коммитится, тесты читают только его.
"""

import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np

SOURCE = Path(__file__).parents[2] / "PANDA_Polygon" / "docs" / "model" / "square.py"
OUTPUT = Path(__file__).parent / "data" / "square_reference.json"
COORDINATE = np.linspace(-0.4975, 0.4975, 200)  # центры 200 бинов: ни одна точка не лежит на стенке или краю δ
LENGTH_RATIOS = (1.0, 2.0, 4.0)
LAYER_WIDTH_RATIO = 1.0
ANGLES_DEG = (60.0, 100.0, 120.0, 150.0, 175.0)
WALL_LAYERS = (0.0, 0.03)
VOLUME_FRACTIONS = (0.02, 0.05, 0.1, 0.2, 0.3)
SHAPES = ("filament_w1c0", "droplet_w1c0")


def load_square_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("panda_polygon_square", SOURCE)
    assert spec is not None and spec.loader is not None, f"нет файла {SOURCE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    square = load_square_module()
    reference: dict[str, object] = {
        "source": "github.com/Vandyshge/PANDA_Polygon@38ff491 docs/model/square.py",
        "coordinate": COORDINATE.tolist(),
        "layer_width_ratio": LAYER_WIDTH_RATIO,
    }
    for shape_name in SHAPES:
        cases = []
        for length_ratio, angle_deg, wall_layer, phi in itertools.product(
            LENGTH_RATIOS, ANGLES_DEG, WALL_LAYERS, VOLUME_FRACTIONS
        ):
            angle = math.radians(angle_deg)
            feasible = bool(square.COND_SQUARE[shape_name](length_ratio, phi, angle, wall_layer))
            case: dict[str, object] = {
                "length_ratio": length_ratio,
                "theta": angle_deg,
                "delta": wall_layer,
                "phi": phi,
                "feasible": feasible,
            }
            if feasible:
                profile = square.RHO_SQUARE[shape_name](
                    COORDINATE, length_ratio, LAYER_WIDTH_RATIO, phi, angle, wall_layer
                )
                case["profile"] = np.asarray(profile, dtype=float).tolist()
                case["energy"] = float(square.AREA_SQUARE[shape_name](length_ratio, phi, angle, wall_layer, flag=False))
                case["area_sum"] = float(
                    square.AREA_SQUARE[shape_name](length_ratio, phi, angle, wall_layer, flag=True)
                )
            cases.append(case)
        reference[shape_name] = cases
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(reference))
    print(f"записано в {OUTPUT}: " + ", ".join(f"{name}: {len(reference[name])}" for name in SHAPES))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
```

- [ ] **Шаг 2: Сгенерировать эталон**

Нужна локальная копия `PANDA_Polygon/` в корне репозитория на коммите `38ff491` (`git -C PANDA_Polygon log -1 --oneline`).

Run: `uv run python tests/regression/generate_square_reference.py`
Expected: `записано в .../square_reference.json: filament_w1c0: 150, droplet_w1c0: 150`.

- [ ] **Шаг 3: Убедиться, что тесты падают**

Run: `uv run pytest -q`
Expected: ошибки сборки тестов — модуля `panda_md.pores.square` и имени `SquarePore` в `panda_md` ещё нет.

- [ ] **Шаг 4: Написать квадратную пору и две формы**

`src/panda_md/pores/square/pore.py`:

```python
"""Квадратная пора: канал квадратного сечения со стороной a и длиной L вдоль оси.

Соглашения — как в PANDA_Polygon (`docs/model/square.py`, коммит 38ff491): a = 1, z = 0 в центре сечения,
профиль поперёк одной стенки, нормировка на слой шириной ly (в долях a), φ — относительно исходного сечения.
"""

from typing import ClassVar

from pydantic import Field, model_validator

from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry

SQUARE_HALF_SIDE = 0.5  # стенки квадрата в безразмерной координате: z = ±0.5
SQUARE_SECTION_AREA = 1.0  # площадь исходного сечения при a = 1


def accessible_side(wall_layer: float) -> float:
    """a′ = 1 − 2δ — сторона квадрата, доступная жидкости."""
    return 1.0 - 2.0 * wall_layer


class SquareGeometry(PoreGeometry):
    pore_length_ratio: float = Field(gt=0.0)  # l = L / a
    layer_width_ratio: float = Field(gt=0.0)  # ly — ширина слоя нормировки профиля в долях a

    def coordinate_bounds(self) -> tuple[float, float]:
        return (-SQUARE_HALF_SIDE, SQUARE_HALF_SIDE)

    def profile_integral(self) -> float:
        return SQUARE_SECTION_AREA / self.layer_width_ratio


class SquarePore(Pore):
    name: ClassVar[str] = "square"

    side_nm: float = Field(gt=0.0)  # a — сторона квадратного сечения
    length_nm: float = Field(gt=0.0)  # L — длина канала (период вдоль оси)
    layer_width_nm: float = Field(gt=0.0)  # ширина слоя, на которую нормирован профиль (ширина ящика)

    @model_validator(mode="after")
    def _check_layer_covers_pore(self) -> "SquarePore":
        if self.layer_width_nm < self.side_nm:
            raise ValueError(f"ширина слоя {self.layer_width_nm} нм меньше стороны поры {self.side_nm} нм")
        return self

    def geometry(self) -> SquareGeometry:
        return SquareGeometry(
            pore_length_ratio=self.length_nm / self.side_nm, layer_width_ratio=self.layer_width_nm / self.side_nm
        )

    def characteristic_size_nm(self) -> float:
        return self.side_nm

    def volume_nm3(self) -> float:
        return self.side_nm**2 * self.length_nm
```

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
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side


def circular_segment_factor(contact_angle: float) -> float:
    """θ − sin θ cos θ: площадь сегмента = R² · (этот множитель)."""
    return contact_angle - math.sin(contact_angle) * math.cos(contact_angle)


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
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        radius = self.arc_radius(geometry, values)
        offset = (
            4.0 * radius * math.sin(contact_angle) ** 3 / (3.0 * (2.0 * contact_angle - math.sin(2.0 * contact_angle)))
        )
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

`src/panda_md/pores/square/droplet.py`:

```python
"""Капли (droplet) в квадратной поре. Формулы — PANDA_Polygon `square.py` (коммит 38ff491)."""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import REGIME_BOUNDARY_DEG, InterfaceShape, SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import spherical_cap_centroid_offset, spherical_cap_volume_factor
from panda_md.pores.square.pore import SquareGeometry, SquarePore, accessible_side


@SquarePore.register
class SquareDropletOneWall(InterfaceShape[SquareGeometry]):
    """droplet_w1c0: сферический сегмент на нижней стенке, не касается других стенок и углов."""

    name: ClassVar[str] = "droplet_w1c0"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

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
```

`src/panda_md/pores/square/__init__.py`:

```python
"""Квадратная пора и её формы. Импорт модулей форм регистрирует их в SquarePore."""

from panda_md.pores.square import droplet, filament
from panda_md.pores.square.pore import SquareGeometry, SquarePore

__all__ = ["SquareGeometry", "SquarePore", "droplet", "filament"]
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
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = [
    "DensityProfile",
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
    "Unit",
    "__version__",
]
```

- [ ] **Шаг 5: Тесты проходят**

Run: `uv run pytest -q`
Expected: `70 passed, 2 skipped`; контракт включает `square/filament_w1c0` (оба режима) и `square/droplet_w1c0`; регрессия квадрата совпадает до 1e-12 по профилю, условию существования, энергии и сумме площадей.

- [ ] **Шаг 6: Полная проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests
git commit -m "Квадратная пора: filament_w1c0 и droplet_w1c0; регрессия на PANDA_Polygon (38ff491)" \
  -m "Вторая настоящая пора подключена без правок core/." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 6: Общий фит, результат, синтетический профиль

**Files:**
- Create: `src/panda_md/core/spec.py`, `src/panda_md/core/result.py`, `src/panda_md/core/fitting.py`, `src/panda_md/synthetic/__init__.py`, `src/panda_md/synthetic/profiles.py`, `tests/conftest.py`, `tests/unit/test_fitting.py`
- Modify: `src/panda_md/core/model.py` (метод `fit`), `src/panda_md/__init__.py` (экспорт `FitResult`, `FitSpec`)

**Interfaces:**
- Consumes: задачи 1–5.
- Produces:
  - `panda_md.core.spec.FitSpec` (`optimize`, `fixed`, `bounds`, `initial`, `normalization`, `density_bulk_per_nm3`, `loss`, `window`, `starts_per_parameter` = 5, `optimizer` = `NELDER_MEAD`, `regime`).
  - `panda_md.core.result.FitResult` (`pore_name`, `shape_name`, `regime`, `frame`, `normalization`, `values`, `free_parameters`, `fixed_parameters`, `loss`, `success`, `message`, `start_count`, `evaluation_count`, `feasible`, `at_bounds`, `at_regime_boundary`, `wall_layer_thickness_nm`, `density_bulk_derived_per_nm3`, `volume_fraction_from_data`, `coordinate`, `data`, `model_curve`, `alternatives`, `spec`; `summary()`, `__str__`).
  - `panda_md.core.fitting.fit(model, profile, spec) -> FitResult`; константы `REGIME_GAP_DEG = 0.5`, `MIN_WINDOW_POINTS = 10` и др.
  - `InterfaceModel.fit(profile, *, optimize, fixed, bounds, initial, normalization, density_bulk_per_nm3, loss, window, starts_per_parameter, optimizer, regime) -> FitResult`.
  - `panda_md.synthetic.profiles.synthetic_profile(model, values, *, density_bulk_per_nm3, bin_count=200, noise_std=0.0, frame=Frame.PORE, offset=0.0, seed=None) -> DensityProfile` (отказывает, если форма не существует при `values`).

- [ ] **Шаг 1: Написать фикстуры и тесты фита**

`tests/conftest.py`:

```python
"""Общие поры для тестов: размеры порядка реальных МД-систем."""

import pytest

from panda_md import SlitPore, SquarePore


@pytest.fixture
def slit_pore() -> SlitPore:
    return SlitPore(width_nm=8.0, length_nm=16.0)


@pytest.fixture
def square_pore() -> SquarePore:
    return SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
```

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
    MissingParameterValueError,
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
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/unit/test_fitting.py -q`
Expected: ошибка сборки `ImportError: cannot import name 'FitResult' from 'panda_md'`.

- [ ] **Шаг 3: Написать настройки фита, результат и алгоритм**

`src/panda_md/core/spec.py`:

```python
"""Все настройки одного фита. Сохраняются в результате и читаются из конфига CLI."""

from pydantic import Field

from panda_md.core.enums import LossFunction, Normalization, OptimizerMethod, Regime
from panda_md.core.types import PandaModel


class FitSpec(PandaModel):
    optimize: tuple[str, ...] | None = None  # None — параметры модели по умолчанию (θ, δ)
    fixed: dict[str, float] = Field(default_factory=dict)  # единицы пользователя
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)  # сузить границы на этот фит
    initial: dict[str, float] = Field(default_factory=dict)  # дополнительная стартовая точка
    normalization: Normalization = Normalization.AUTO
    density_bulk_per_nm3: float | None = Field(default=None, gt=0.0)
    loss: LossFunction = LossFunction.L1
    window: tuple[float, float] | None = None  # окно по безразмерной координате; None — вся пора
    starts_per_parameter: int = Field(default=5, ge=1)
    optimizer: OptimizerMethod = OptimizerMethod.NELDER_MEAD
    regime: Regime | None = None  # None — все режимы, объявленные формой и разрешённые границами θ
```

`src/panda_md/core/result.py`:

```python
"""Итог фита: значения параметров, качество, флаги проверок, данные для графика."""

from panda_md.core.enums import Frame, Normalization, Regime
from panda_md.core.spec import FitSpec
from panda_md.core.types import Array1D, PandaModel


class FitResult(PandaModel):
    pore_name: str
    shape_name: str
    regime: Regime | None  # None — у формы нет параметра θ
    frame: Frame
    normalization: Normalization
    values: dict[str, float]  # все параметры, единицы пользователя (θ в градусах, δ в долях размера поры)
    free_parameters: tuple[str, ...]
    fixed_parameters: tuple[str, ...]
    loss: float
    success: bool
    message: str
    start_count: int
    evaluation_count: int
    feasible: bool
    at_bounds: tuple[str, ...]
    at_regime_boundary: bool
    wall_layer_thickness_nm: float | None
    density_bulk_derived_per_nm3: float | None  # нормировка mean: ρ_bulk = n̄ / φ
    volume_fraction_from_data: float | None  # нормировка bulk: φ = ∫ρ/ρ_bulk, если φ не задана
    coordinate: Array1D  # безразмерная координата окна фита
    data: Array1D  # безразмерная плотность данных
    model_curve: Array1D  # модель в системе отсчёта данных
    alternatives: tuple["FitResult", ...] = ()  # лучшие результаты в других режимах смачивания
    spec: FitSpec

    def summary(self) -> str:
        regime_label = self.regime.value if self.regime is not None else "—"
        lines = [f"{self.pore_name}/{self.shape_name}  режим: {regime_label}  loss: {self.loss:.4g}"]
        for name, value in self.values.items():
            marker = "свободный" if name in self.free_parameters else "фиксирован"
            lines.append(f"  {name:>6} = {value:.6g}  ({marker})")
        flags = []
        if not self.success:
            flags.append(f"оптимизатор: {self.message}")
        if not self.feasible:
            flags.append("результат вне области существования формы")
        if self.at_bounds:
            flags.append(f"у границы: {', '.join(self.at_bounds)}")
        if self.at_regime_boundary:
            flags.append("θ прижат к 90° (граница режимов)")
        lines.extend(f"  ! {flag}" for flag in flags)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
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
IDENTIFIABILITY_STEP_FRACTION = 1e-3  # шаг проверки «параметр влияет», доля ширины его границ
IDENTIFIABILITY_TOLERANCE = 1e-9  # относительное изменение профиля, ниже которого параметр «не влияет»
BOUND_PROXIMITY_FRACTION = 1e-3  # ближе этой доли ширины границ — параметр «у границы»
BARRIER_LEVEL_FACTOR = 10.0  # уровень барьера: во столько раз выше ошибки нулевой модели (+1)
BARRIER_SLOPE = 100.0  # наклон барьера по величине выхода φ за границы существования

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
        step = IDENTIFIABILITY_STEP_FRACTION * (free_spec.upper_bounds[index] - free_spec.lower_bounds[index])
        changes_profile = False
        for direction in (1.0, -1.0):
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

Заменить `src/panda_md/core/model.py` целиком (добавлен метод `fit` и импорты для него):

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

DEFAULT_FREE_PARAMETERS = (ParameterName.CONTACT_ANGLE.value, ParameterName.WALL_LAYER_THICKNESS.value)


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
        return tuple(name for name in DEFAULT_FREE_PARAMETERS if name in self.parameters.names)

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

- [ ] **Шаг 4: Написать синтетический профиль и обновить публичный API**

`src/panda_md/synthetic/__init__.py`:

```python
"""Синтетические данные из аналитических моделей — для тестов и валидации."""
```

`src/panda_md/synthetic/profiles.py`:

```python
"""Синтетические профили плотности из аналитической модели — для тестов и валидации."""

from collections.abc import Mapping

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.errors import InfeasibleParametersError
from panda_md.core.model import InterfaceModel
from panda_md.core.profile import DensityProfile


def synthetic_profile(
    model: InterfaceModel,
    values: Mapping[str, float],
    *,
    density_bulk_per_nm3: float,
    bin_count: int = 200,
    noise_std: float = 0.0,
    frame: Frame = Frame.PORE,
    offset: float = 0.0,
    seed: int | None = None,
) -> DensityProfile:
    """Профиль, который дала бы МД при идеальной форме с параметрами values (единицы пользователя).

    noise_std — шум в долях ρ_bulk на бин; offset — сдвиг интерфейса вдоль оси профиля в долях размера поры
    (моделирует ошибку положения стенок); frame — в какой системе отсчёта вернуть координату.
    """
    pore = model.pore
    geometry = pore.geometry()
    lower, upper = geometry.coordinate_bounds()
    edges = np.linspace(lower, upper, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    internal = model.internal_values(values)
    if not model.shape.feasible(geometry, internal):
        raise InfeasibleParametersError(f"{model}: форма не существует при {dict(values)}")
    density = density_bulk_per_nm3 * model.shape.profile(centers - offset, geometry, internal)
    if noise_std > 0.0:
        generator = np.random.default_rng(seed)
        density = density + generator.normal(0.0, noise_std * density_bulk_per_nm3, size=density.shape)
    density = np.clip(density, 0.0, None)

    coordinate = centers
    if frame is Frame.LIQUID:
        coordinate = centers - (model.shape.center_of_mass(geometry, internal) + offset)
    volume_fraction = internal[ParameterName.VOLUME_FRACTION]
    return DensityProfile(
        coordinate_nm=coordinate * pore.characteristic_size_nm(),
        density_per_nm3=density,
        frame=frame,
        particle_count=density_bulk_per_nm3 * volume_fraction * pore.volume_nm3(),
        pore_volume_nm3=pore.volume_nm3(),
    )
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
    "Unit",
    "__version__",
]
```

- [ ] **Шаг 5: Тесты проходят**

Run: `uv run pytest -q`
Expected: `88 passed, 2 skipped` за 1–3 с.

- [ ] **Шаг 6: Полная проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests
git commit -m "Общий фит: свободные и фиксированные параметры, режимы смачивания, барьер, проверки; результат; синтетика" \
  -m "Оптимизатор по умолчанию — Nelder–Mead: на L1-ошибке L-BFGS-B застревает." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 7: Синтетическая валидация и описание тестов

**Files:**
- Create: `tests/synthetic/test_recovery.py`, `tests/README.md`

**Interfaces:**
- Consumes: `synthetic_profile`, `InterfaceModel.fit`, `FitResult` из задачи 6.
- Produces: тесты с маркером `slow` (запуск без них — `uv run pytest -m "not slow"`); `tests/README.md` — описание всех наборов и выводы по O2, O3, оптимизатору и дискретизации.

- [ ] **Шаг 1: Написать синтетические тесты**

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
CASES: list[tuple[InterfaceModel, dict[str, float]]] = [
    (SLIT.interface("roll"), {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
    (SLIT.interface("droplet"), {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
    (SQUARE.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.02, "phi": 0.05}),
    (SQUARE.interface("filament_w1c0"), {"theta": 130.0, "delta": 0.02, "phi": 0.1}),
    (SQUARE.interface("filament_w1c0"), {"theta": 60.0, "delta": 0.02, "phi": 0.05}),
]


def case_id(case: tuple[InterfaceModel, dict[str, float]]) -> str:
    model, truth = case
    return f"{model.pore.name}/{model.shape.name}/θ={truth['theta']:.0f}"


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

- [ ] **Шаг 2: Запустить**

Run: `uv run pytest tests/synthetic -q`
Expected: `19 passed` примерно за 10–15 с. Тесты проверяют уже написанный код; если какой-то падает — это расхождение с выводами в `tests/README.md`, его нужно разобрать, а не ослаблять порог.

- [ ] **Шаг 3: Написать описание тестов**

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
| модель | `unit/test_model.py` | математика | параметры модели, единицы, ошибки имён, профиль валика в центре поры против формулы, допустимость по интервалу φ, капиллярный функционал | щель 8 × 16 нм, квадрат 6 × 12 нм | совпадение с формулой | 1e-12 |
| контракт | `contract/test_shapes.py` | математика | для **каждой** зарегистрированной формы: атрибуты контракта; допустимые точки в каждом объявленном режиме; ∫ρ dz = φ · (A / ly); 0 ≤ ρ ≤ 1 и ρ = 0 вне поры; центр масс формулы = численному; S_LV > 0, S_SL ≥ 0 | сетка θ (по 3 угла в режиме), δ ∈ {0, 0.03}, 25 значений φ | все проверки на всех допустимых точках | ∫ — 1e-4 (ступенька на краю), центр масс — 1e-4 |
| регрессия | `regression/test_reference_formulas.py` | математика | перенос формул без изменений | эталоны из старой PANDA (`alpha.py`, 93c6dbb) и PANDA_Polygon (`square.py`, 38ff491); генераторы `regression/generate_*.py` | профиль, интервал φ, условие существования, энергия и сумма площадей совпадают | 1e-12 |
| фит | `unit/test_fitting.py` | оптимизация | один оптимизатор при 1, 2, 3 свободных параметрах; фиксированные не меняются; флаги «у границы» и «у 90°»; пустое окно, нет φ, нет допустимого старта, не объявленный режим, параметр без влияния — ошибки; деление по режимам смачивания | синтетика без шума, 200 бинов | восстановление в пределах дискретизации | θ — 0.5°, δ — 0.0025 (полбина), φ — 0.003 |
| синтетика | `synthetic/test_recovery.py` | физическая модель | полная цепочка «форма → профиль с шумом → фит» для 5 случаев (валик, две капли, нить при 130° и 60°); три режима ТЗ; нормировка по средней плотности (O2); ошибка положения стенок (O3) | шум 0.5 % ρ_bulk на бин, 5 зерён | см. ниже | максимум 1.5°, среднее 1° |

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
````

- [ ] **Шаг 4: Полная проверка и коммит**

Run: `make check` — без ошибок; итог `107 passed, 2 skipped`.

```bash
git add tests
git commit -m "Синтетическая валидация (O2, O3) и описание тестов в tests/README.md" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 8: Документация стадии и CI

**Files:**
- Modify: `docs/superpowers/specs/2026-10-01-panda-architecture-design.md`, `ROADMAP.md`, `README.md`

**Interfaces:**
- Consumes: выводы из `tests/README.md` (задача 7).
- Produces: спецификация и roadmap согласованы с реализацией; стадия 1 закрыта.

- [ ] **Шаг 1: Обновить спецификацию**
Точечные замены в `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (старый текст → новый).
§3.0, таблица моделей, строка `Centering`, `FitSpec`:
```markdown
| `Centering`, `FitSpec` | модель | допустимые значения перечислений, `starts ≥ 1`, `n_jobs ≥ 1`, окно `lo < hi` |
```
→
```markdown
| `Centering`, `FitSpec` | модель | допустимые значения перечислений, `starts_per_parameter ≥ 1`, `density_bulk > 0` |
```
§3.0, список «Следствия»:
```markdown
- ошибки проверки pydantic (`ValidationError`) оборачиваются в `PandaError` с понятным текстом;
```
→
```markdown
- ошибки проверки pydantic (`ValidationError`, наследник `ValueError`) пропускаются как есть: в них указаны поле
  и причина; собственные ошибки библиотеки — наследники `PandaError`;
```
§3.6, первый абзац (`FitSpec: ...` — три строки) → 
```markdown
`FitSpec`: `optimize`, `fixed`, `bounds`, `initial`, `normalization` (`auto` / `bulk` / `mean`), `density_bulk_per_nm3`,
`loss` (`l1` / `l2`), `window`, `starts_per_parameter` (5), `optimizer` (`Nelder-Mead`), `regime` (`None` — все
объявленные формой режимы, либо `wetting` / `non_wetting`). Параллельный запуск (`n_jobs`) — вместе с фитом по
блокам (фаза 6).
```
§4, псевдокод: строку `стартовые точки: сетка starts^k по свободным (+ initial); при k > 3 — латинский гиперкуб` →
`стартовые точки: центры ячеек сетки starts^k по свободным (+ initial)`; строку
`из каждой точки: Optimizer.minimize(objective, x0, box)` → `из каждой точки: scipy.optimize.minimize(objective, x0, method, bounds)`.
§4, абзац **Оптимизатор** (три строки) →
```markdown
**Оптимизатор.** `scipy.optimize.minimize` с методом из `OptimizerMethod`; по умолчанию **Nelder–Mead**: на L1-ошибке
L-BFGS-B с конечными разностями застревает (прототип, 15.10.2026, `tests/README.md`); доступны `Powell` и `L-BFGS-B`.
Аналитический градиент формы и параллельный запуск — после прототипа, когда понадобятся по скорости.
```
§11, строки O2 и O3 →
```markdown
| O2 | ~~Определяется ли φ вместе с θ, δ в нормировке `mean`~~ — для капель и нитей да (\|Δφ\| < 0.005), для валика нет: нужна известная φ | закрыт прототипом, `tests/README.md` |
| O3 | ~~Центрирование по умолчанию~~ — по поре; у форм на одной стенке ошибка положения стенок уходит в δ, у валика нужен свободный `shift` | закрыт прототипом, `tests/README.md` |
```
В конец таблицы §11 добавить:
```markdown
| O10 | Точность δ ограничена шириной бина (δ задаётся краем профиля); при свободных θ, δ, φ разброс θ до 0.5° без шума | учесть в `validation/` (фаза 6) и в статье |
```
- [ ] **Шаг 2: Обновить ROADMAP**

В «Фаза 3. Прототип ядра» добавить первой строкой списка:

```markdown
- **Статус: выполнено** — план `docs/superpowers/plans/2026-10-15-stage1-core-prototype.md`; 107 тестов; выводы по O2, O3, оптимизатору — `tests/README.md`.
```

- [ ] **Шаг 3: Обновить README**

В README заменить блок `> **Status: pre-alpha.** ...` на:

```markdown
> **Status: pre-alpha.** The core (parameters, pore/shape contract, generic fit) works for the slit pore
> (`roll`, `droplet`) and the square pore (`filament_w1c0`, `droplet_w1c0`). Trajectory processing and CLI are
> not implemented yet; the published 0.0.1 contains no code.

```python
import panda_md as panda
from panda_md.synthetic.profiles import synthetic_profile

pore = panda.SlitPore(width_nm=8.0, length_nm=16.0)
roll = pore.interface("roll")
profile = synthetic_profile(roll, {"theta": 120.0, "delta": 0.03, "phi": 0.4}, density_bulk_per_nm3=30.0)
result = roll.fit(profile, optimize=["theta", "delta"], density_bulk_per_nm3=30.0)
print(result)
```
```

- [ ] **Шаг 4: Проверить, отправить, дождаться CI**

Run: `make check` — без ошибок. Затем:

```bash
git add docs ROADMAP.md README.md
git commit -m "Стадия 1 закрыта: спецификация, ROADMAP и README согласованы с прототипом" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
git push
gh run watch $(gh run list --limit 1 --json databaseId -q '.[0].databaseId') --exit-status
```

Expected: CI зелёный на Python 3.10–3.13 (lint, test, build).
