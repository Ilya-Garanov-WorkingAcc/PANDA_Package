# Стадия 4 — путь данных: план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** От траектории МД до профиля для фита: чтение кадров (MDAnalysis или массивы), поиск стенок щели с учётом PBC, центрирование по поре или по жидкости, бины, средние по блокам времени, сохранение; сверка со старой PANDA и проверка на реальной МД (вода в щели графита, бензол на подложке SOLD, кальцит).

**Architecture:** Интерфейс `Trajectory` (кадры: время, ящик, позиции запрошенных групп) — в ядре, поэтому поры могут искать по нему свои стенки, а MDAnalysis импортируется только в `trajectory/reader.py` (необязательная зависимость `panda-md[md]`). Пора знает ось профиля, выборку стенок и как по ним найти свой центр (щель — середина наибольшего пустого промежутка между атомами подложки, в том числе через границу ящика). `density_profile` потоково строит `ProfileSet` (блоки × бины); `.mean()` даёт `DensityProfile` для фита.

**Tech Stack:** Python ≥ 3.10, numpy, scipy, pydantic v2, MDAnalysis ≥ 2.7 (extra `md`), pytest, uv, black, ruff, mypy (strict + плагин pydantic).

**Spec:** `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (§3.3, §3.5, §5, §9), разбор `docs/architecture/03_centering_data.md`. Данные МД (вне репозитория): `PANDA_Reborn/graphite_decane_tip4p_1.0`, `roughness_2/angles/base` (бензол/SOLD), `roughness_2/angles/Dev_PANDA/.../calcite_decane_tip4p_sc_2.5_offset` (один кадр). В тесты входят только посчитанные по ним профили и найденные поры (≈ 50 КБ).

**Решения, принятые при прототипировании** (вынесены в спецификацию в задаче 7, D21–D23):
- Стратегии центрирования две: `pore` и `liquid`. «Без сдвига» из спецификации — это `pore` с заданным `pore_center_nm` (например, 0 для уже отцентрированных данных).
- Поиск стенок по атомам — только у щели. У треугольной и квадратной пор пока нет данных МД; для них центр задаётся явно (`pore_center_nm`), иначе — `NotDerivedError` с подсказкой.
- Осевое центрирование (`+axis`, ось формы `SymmetryAxis`), `per="molecule"`, `wall_offset` и утилита статистической неэффективности переносятся в фазу 6: на одномерный профиль они не влияют, нужны для 2D-карт, облаков точек и выбора длины блока.
- Неполный последний блок отбрасывается с предупреждением; в системе жидкости бины покрывают ящик первого кадра.

**Проверено до написания плана.** Весь код ниже собран и прогнан по задачам в чистом клоне ветки `stage1-core-prototype` (коммит `ef11e48`): для каждой задачи её тесты падают до реализации с указанной ошибкой, после — `black --check`, `ruff check`, `mypy` и `pytest` проходят (итого 400 тестов, 28 пропусков); на Python 3.10 с MDAnalysis 2.9 — тоже. Код переносить дословно.

## Global Constraints

- Все ограничения стадий 1–3 в силе: Python ≥ 3.10 (`class X(str, Enum)`, без `StrEnum`/`Self`); объекты-данные — pydantic, без датаклассов (D14); фиксированные варианты — `Enum`, типы у всех полей, говорящие имена (D17, §2.1 спецификации); black 120, ruff, mypy strict; комментарии и сообщения — по-русски.
- Слои (§2 спецификации): `core` не импортирует `pores` и `trajectory`; MDAnalysis импортируется только в `trajectory/reader.py`, и только внутри функции — `import panda_md` его не тянет (проверяется тестом).
- Все длины — нм, время — пс; MDAnalysis отдаёт Å, перевод — в `reader.py`.
- Разработка локальная, в ветке `stage1-core-prototype`; в GitHub не отправлять, пока пользователь не скажет.
- Многострочный текст и markdown — только через Write/Edit (Bash-heredoc в этой среде теряет пустые строки и строки, начинающиеся с `#`). Сообщения коммитов — отдельными `-m` на абзац, последний — `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Проверка перед коммитом: `make check`. Окружение разработки — `uv sync --all-extras` (с задачи 2).

## Review Focus

1. Щель, пересекающая границу ящика (подложка у дна, её образ сверху — как в реальной системе графита): ожидание — H и центр поры находятся через наибольший пустой промежуток с учётом PBC, координата заворачивается в [−L/2, L/2). Тесты — `test_largest_gap_finds_pore_through_box_boundary` (задача 1), `test_slit_from_trajectory_finds_walls_through_box_boundary` (задача 3), `test_walls_found_in_every_frame_follow_a_drifting_pore` (задача 4).
2. Центр жидкости, пересекающей границу ящика: ожидание — круговое среднее, а не середина ящика. Тесты — `test_circular_mean_of_cloud_across_boundary_is_not_box_middle` (задача 1), `test_liquid_centering_puts_center_of_mass_at_zero_across_boundary` (задача 4), регрессия со старой PANDA (задача 6).
3. Длина траектории не кратна блоку: ожидание — неполный блок отброшен с предупреждением, а не усреднён с меньшим числом кадров. Тест — `test_blocks_by_time_and_partial_block_is_dropped` (задача 4).
4. Неверные размеры поры или выборка (часть частиц вне поры, промежуток стенок ≠ H, основание ящика ≠ kL²): ожидание — предупреждение, а не молча неверная плотность. Тест — `test_suspicious_geometry_gives_warnings` (задача 4).
5. Окружение без MDAnalysis: ожидание — `import panda_md` работает, `load_trajectory` даёт понятную ошибку с `pip install 'panda-md[md]'`. Тесты — `test_load_trajectory_reports_missing_files_and_package` (задача 2), `test_importing_package_does_not_import_mdanalysis` (задача 5).

---

### Задача 1: Общие основы — массивы, перечисления, PBC, время

**Files:**
- Modify: `src/panda_md/core/types.py` (`Array2D`, `Array3D`, `IndexArray1D`), `src/panda_md/core/enums.py` (`Axis`, `ProfileCentering`, `Orientation`, `TimeUnit`), `src/panda_md/core/errors.py` (`InvalidTrajectoryError`, `MissingDependencyError`)
- Create: `src/panda_md/core/pbc.py`, `src/panda_md/core/timing.py`
- Create: `tests/unit/test_pbc_and_time.py`

**Interfaces:**
- Consumes: `PandaModel`, `FloatArray` (стадия 1).
- Produces: `Array2D`, `Array3D` (проверенные неизменяемые массивы float64), `IndexArray = NDArray[np.int64]`, `IndexArray1D`; `Axis.X/Y/Z` с `.column` (0, 1, 2) и `AXIS_ORDER`; `ProfileCentering.PORE/LIQUID`; `Orientation.AUTO/KEEP`; `TimeUnit` с `.picoseconds`; `wrap_centered(values, period)`, `circular_mean(values, period) -> float` (в [0, L)), `largest_gap(values, period) -> (start, end)`; `TimeValue = float | str`, `to_picoseconds(value) -> float`; `InvalidTrajectoryError(PandaError)`, `MissingDependencyError(PandaError, ImportError)`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_pbc_and_time.py`:

```python
"""Периодические границы вдоль оси ящика и время в пользовательских единицах."""

import numpy as np
import pytest

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import circular_mean, largest_gap, wrap_centered
from panda_md.core.timing import to_picoseconds

PERIOD = 10.0


def test_wrap_centered_maps_to_nearest_image_around_zero() -> None:
    values = np.array([-12.0, -5.0, -4.9, 0.0, 4.9, 5.0, 17.5])
    np.testing.assert_allclose(wrap_centered(values, PERIOD), [-2.0, -5.0, -4.9, 0.0, 4.9, -5.0, -2.5])


def test_circular_mean_of_cloud_across_boundary_is_not_box_middle() -> None:
    cloud = np.array([9.0, 9.5, 0.5, 1.0])  # облако вокруг 0 ≡ 10, а не вокруг 5
    center = circular_mean(cloud, PERIOD)
    assert min(center, PERIOD - center) == pytest.approx(0.0, abs=1e-12)
    assert circular_mean(np.array([2.0, 3.0, 4.0]), PERIOD) == pytest.approx(3.0)


def test_largest_gap_finds_pore_through_box_boundary() -> None:
    walls = np.array([0.2, 1.0, 2.0, 2.9])  # подложка внизу, её образ сверху: пора от 2.9 до 10.2
    start, end = largest_gap(walls, PERIOD)
    assert (start, end) == pytest.approx((2.9, 10.2))
    inner = np.array([1.0, 1.5, 8.0, 8.5])  # две стенки внутри ящика: пора от 1.5 до 8.0
    assert largest_gap(inner, PERIOD) == pytest.approx((1.5, 8.0))


def test_empty_selections_are_errors() -> None:
    with pytest.raises(InvalidTrajectoryError):
        circular_mean(np.array([]), PERIOD)
    with pytest.raises(InvalidTrajectoryError):
        largest_gap(np.array([]), PERIOD)


@pytest.mark.parametrize(
    ("value", "picoseconds"),
    [
        (2000.0, 2000.0),
        (5, 5.0),
        ("2 ns", 2000.0),
        ("1.5ns", 1500.0),
        ("500 ps", 500.0),
        ("1 us", 1e6),
        ("10 fs", 0.01),
    ],
)
def test_time_values_are_converted_to_picoseconds(value: float | str, picoseconds: float) -> None:
    assert to_picoseconds(value) == pytest.approx(picoseconds)


@pytest.mark.parametrize("value", ["2", "two ns", "2 hours"])
def test_unparsable_time_is_an_error(value: str) -> None:
    with pytest.raises(ValueError, match=r"время|единица"):
        to_picoseconds(value)
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ImportError: cannot import name 'InvalidTrajectoryError' from 'panda_md.core.errors'`.

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/core/types.py` целиком:

`src/panda_md/core/types.py`:

```python
"""Общая базовая модель pydantic, типы массивов numpy с проверкой и неизменяемый словарь."""

from collections.abc import Callable, Iterator, Mapping
from typing import Annotated, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, PlainValidator

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int64]

ValueT = TypeVar("ValueT")


class FrozenMapping(Mapping[str, ValueT], Generic[ValueT]):
    """Неизменяемый словарь с хешем: поля-словари моделей pydantic с frozen=True иначе остаются изменяемыми."""

    __slots__ = ("_items",)
    _items: dict[str, ValueT]

    def __init__(self, items: Mapping[str, ValueT]) -> None:
        self._items = dict(items)

    def __getitem__(self, key: str) -> ValueT:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    def __repr__(self) -> str:
        return f"FrozenMapping({self._items!r})"


def _freeze_floats(items: Mapping[str, float]) -> FrozenMapping[float]:
    return FrozenMapping(items)


def _freeze_bounds(items: Mapping[str, tuple[float, float]]) -> FrozenMapping[tuple[float, float]]:
    return FrozenMapping(items)


def _thaw_floats(items: Mapping[str, float]) -> dict[str, float]:
    return dict(items)


def _thaw_bounds(items: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return dict(items)


FrozenFloats = Annotated[
    Mapping[str, float],
    AfterValidator(_freeze_floats),
    PlainSerializer(_thaw_floats, return_type=dict[str, float]),
]
"""Поле «имя → число»: на входе любой словарь, внутри — FrozenMapping, в JSON — обычный объект."""

FrozenBounds = Annotated[
    Mapping[str, tuple[float, float]],
    AfterValidator(_freeze_bounds),
    PlainSerializer(_thaw_bounds, return_type=dict[str, tuple[float, float]]),
]
"""Поле «имя → (нижняя, верхняя граница)»."""


def _float_array_validator(dimensions: int) -> Callable[[object], FloatArray]:
    """Проверка: неизменяемый массив float64 заданной размерности без NaN и бесконечностей."""

    def validate(value: object) -> FloatArray:
        array = np.array(value, dtype=np.float64)
        if array.ndim != dimensions:
            raise ValueError(f"ожидается массив размерности {dimensions}, получена размерность {array.ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError("массив содержит NaN или бесконечность")
        array.flags.writeable = False
        return array

    return validate


def _validate_index_array(value: object) -> IndexArray:
    """Проверка: неизменяемый одномерный массив неотрицательных целых индексов."""
    array = np.array(value)
    if array.ndim != 1:
        raise ValueError(f"ожидается одномерный массив индексов, получена размерность {array.ndim}")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise ValueError("индексы должны быть целыми")
    array = array.astype(np.int64)
    if np.any(array < 0):
        raise ValueError("индексы не могут быть отрицательными")
    array.flags.writeable = False
    return array


def _serialize_float_array(array: FloatArray) -> list[float]:
    return [float(item) for item in array]


def _serialize_nested_array(array: FloatArray) -> list[object]:
    serialized: list[object] = array.tolist()
    return serialized


def _serialize_index_array(array: IndexArray) -> list[int]:
    return [int(item) for item in array]


Array1D = Annotated[
    FloatArray,
    PlainValidator(_float_array_validator(1)),
    PlainSerializer(_serialize_float_array, return_type=list[float]),
]
Array2D = Annotated[
    FloatArray,
    PlainValidator(_float_array_validator(2)),
    PlainSerializer(_serialize_nested_array, return_type=list[object]),
]
Array3D = Annotated[
    FloatArray,
    PlainValidator(_float_array_validator(3)),
    PlainSerializer(_serialize_nested_array, return_type=list[object]),
]
IndexArray1D = Annotated[
    IndexArray,
    PlainValidator(_validate_index_array),
    PlainSerializer(_serialize_index_array, return_type=list[int]),
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
    LIQUID = "liquid"  # центр масс выбранной жидкости с учётом PBC — система LIQUID


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

Заменить `src/panda_md/core/errors.py` целиком:

`src/panda_md/core/errors.py`:

```python
"""Исключения и предупреждения библиотеки. Все исключения наследуют PandaError."""


class PandaError(Exception):
    """Базовое исключение PANDA."""


class NotDerivedError(PandaError):
    """Величина для этой формы не выведена (нет формулы)."""


class UnknownParameterError(PandaError):
    """Имя параметра не принадлежит модели."""


class UnknownShapeError(PandaError, KeyError):
    """В поре нет формы с таким именем (наследует KeyError: поиск по реестру)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


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


class ProfileMismatchError(InvalidProfileError):
    """Профиль не согласуется с порой: другой размер, другие единицы или другая система отсчёта."""


class InvalidTrajectoryError(PandaError):
    """Траектория или выборка не подходят: пустая выборка, нет стенок, неверный ящик, пустой интервал времени."""


class MissingDependencyError(PandaError, ImportError):
    """Нужен необязательный пакет (например, MDAnalysis из panda-md[md])."""


class PandaWarning(UserWarning):
    """Предупреждение PANDA: расчёт выполнен, но результату стоит не доверять без проверки."""
```

`src/panda_md/core/pbc.py`:

```python
"""Периодические граничные условия вдоль одной оси ящика: заворот, круговое среднее, наибольший пустой промежуток."""

import math

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.types import FloatArray


def wrap_centered(values: FloatArray, period: float) -> FloatArray:
    """Завернуть координаты в [−L/2, L/2) — ближайший образ относительно нуля."""
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
    if values.size == 0:
        raise InvalidTrajectoryError("в выборке стенок нет атомов")
    ordered = np.sort(np.mod(values, period))
    gaps = np.diff(np.append(ordered, ordered[0] + period))
    widest = int(np.argmax(gaps))
    start = float(ordered[widest])
    return start, start + float(gaps[widest])
```

`src/panda_md/core/timing.py`:

```python
"""Время в пользовательских единицах: «2 ns», «500 ps» или число пикосекунд."""

import re

from panda_md.core.enums import TimeUnit

TimeValue = float | str  # число — пикосекунды; строка — «<число> <единица>»
TIME_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*([a-zA-Zµ]+)\s*$")


def to_picoseconds(value: TimeValue) -> float:
    """Перевести время в пикосекунды: 2000.0 → 2000.0, «2 ns» → 2000.0, «1.5ns» → 1500.0."""
    if isinstance(value, (int, float)):
        return float(value)
    match = TIME_PATTERN.match(value)
    if match is None:
        raise ValueError(f"время {value!r} не разобрано: ожидается число пс или строка вида «2 ns»")
    number, unit_text = match.groups()
    unit_text = unit_text.lower().replace("µ", "u")
    try:
        unit = TimeUnit(unit_text)
    except ValueError:
        known = ", ".join(unit.value for unit in TimeUnit)
        raise ValueError(f"единица времени {unit_text!r} неизвестна; доступны: {known}") from None
    return float(number) * unit.picoseconds
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `366 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/core tests/unit/test_pbc_and_time.py
git commit -m "Путь данных, основы: массивы 2D/3D и индексы, оси, стратегии центрирования, PBC, время в единицах" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 2: Траектория — интерфейс, массивы numpy, MDAnalysis

**Files:**
- Modify: `pyproject.toml` (extra `md`, исключение mypy для MDAnalysis), `Makefile` (`install`, `test-env` с `[md]`), `.github/workflows/ci.yml` (`--all-extras` / `--extra md`)
- Create: `src/panda_md/core/trajectory.py`, `src/panda_md/trajectory/__init__.py`, `src/panda_md/trajectory/arrays.py`, `src/panda_md/trajectory/reader.py`
- Create: `tests/unit/test_trajectory.py` (без тестов щели — они в задаче 3)

**Interfaces:**
- Consumes: типы, `TimeValue`, `to_picoseconds`, ошибки (задача 1).
- Produces: `TrajectoryFrame(index, time_ps, box_nm, positions_nm: tuple[Array2D, ...])`; абстрактный `Trajectory`: `select(selection: str | Sequence[int]) -> IndexArray`, `times_ps() -> FloatArray`, `_read(frame_indices, groups)`, готовые `frames(groups, *, start, stop, stride)`, `frame_indices(*, start, stop, stride) -> list[int]`, `frame_at(time, groups)`; `check_box(box_nm)`. `ArrayTrajectory(positions_nm, boxes_nm, times_ps, groups={})` (pydantic). `MDATrajectory(universe)`, `load_trajectory(topology, *trajectory_files) -> MDATrajectory`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_trajectory.py`:

```python
"""Источники кадров: массивы numpy и MDAnalysis."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.reader import MDATrajectory, load_trajectory

BOX_NM = np.array([6.0, 4.0, 10.0])
WALL_LAYERS_NM = (0.2, 0.55, 0.9)  # подложка у дна ящика; её образ сверху — граница поры через PBC
FRAME_COUNT = 5
TIME_STEP_PS = 2.0


def wall_atoms() -> np.ndarray:
    grid = np.array([[x, y] for x in np.arange(0.0, 6.0, 1.0) for y in np.arange(0.0, 4.0, 1.0)])
    return np.vstack([np.column_stack([grid, np.full(len(grid), z)]) for z in WALL_LAYERS_NM])


def slit_trajectory() -> ArrayTrajectory:
    walls = wall_atoms()
    liquid = np.column_stack([np.full(3, 1.0), np.full(3, 2.0), [3.0, 5.0, 7.0]])
    frame = np.vstack([walls, liquid])
    positions = np.repeat(frame[np.newaxis], FRAME_COUNT, axis=0)
    return ArrayTrajectory(
        positions_nm=positions,
        boxes_nm=np.tile(BOX_NM, (FRAME_COUNT, 1)),
        times_ps=TIME_STEP_PS * np.arange(FRAME_COUNT),
        groups={"walls": np.arange(len(walls)), "liquid": len(walls) + np.arange(3)},
    )


def test_array_trajectory_checks_shapes_and_order() -> None:
    with pytest.raises(ValidationError, match="ящики"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((3, 3)), times_ps=[0.0, 1.0])
    with pytest.raises(ValidationError, match="времена"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((2, 3)), times_ps=[1.0, 0.0])
    with pytest.raises(ValidationError, match="группа"):
        ArrayTrajectory(positions_nm=np.zeros((1, 3, 3)), boxes_nm=np.ones((1, 3)), times_ps=[0.0], groups={"g": [5]})


def test_selection_by_group_name_or_indices() -> None:
    trajectory = slit_trajectory()
    assert trajectory.select("liquid").tolist() == [72, 73, 74]
    assert trajectory.select([0, 1]).tolist() == [0, 1]
    with pytest.raises(InvalidTrajectoryError, match="группы"):
        trajectory.select("water")
    with pytest.raises(InvalidTrajectoryError, match="вне"):
        trajectory.select([1000])


def test_frames_are_chosen_by_time_and_stride() -> None:
    trajectory = slit_trajectory()
    assert trajectory.frame_indices() == [0, 1, 2, 3, 4]
    assert trajectory.frame_indices(start="0.002 ns", stop=6.0) == [1, 2, 3]
    assert trajectory.frame_indices(start=0.0, stride=2) == [0, 2, 4]
    assert trajectory.frame_at(3.0, [trajectory.select("liquid")]).index == 2
    with pytest.raises(InvalidTrajectoryError, match="нет кадров"):
        trajectory.frame_indices(start="1 ns")
    with pytest.raises(ValueError, match="stride"):
        trajectory.frame_indices(stride=0)


def memory_universe(dimensions: list[float]) -> Any:  # noqa: ANN401 — MDAnalysis без аннотаций типов
    mdanalysis = pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mdanalysis.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    universe.add_TopologyAttr("names", ["C1", "C2", "OW", "HW"])
    universe.add_TopologyAttr("resnames", ["WALL", "WATER"])
    coordinates = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)  # Å
    universe.load_new(coordinates, format=MemoryReader, dimensions=dimensions, dt=2.0)
    return universe


def test_mdanalysis_trajectory_converts_angstrom_to_nm() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 90.0]))
    oxygen = trajectory.select("resname WATER and name OW")
    assert oxygen.tolist() == [2]
    np.testing.assert_allclose(trajectory.times_ps(), [0.0, 2.0, 4.0])
    frames = list(trajectory.frames([oxygen], start=2.0))
    assert [frame.index for frame in frames] == [1, 2]
    np.testing.assert_allclose(frames[0].positions_nm[0], [[1.8, 1.9, 2.0]])
    np.testing.assert_allclose(frames[0].box_nm, [5.0, 4.0, 10.0])
    with pytest.raises(InvalidTrajectoryError, match="пуста"):
        trajectory.select("resname DECAN")


def test_mdanalysis_trajectory_rejects_triclinic_box() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 60.0]))
    with pytest.raises(InvalidTrajectoryError, match="не прямоугольный"):
        list(trajectory.frames([trajectory.select("name OW")]))


def test_load_trajectory_reports_missing_files_and_package(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("MDAnalysis")
    with pytest.raises(InvalidTrajectoryError, match="не найден"):
        load_trajectory("нет.gro", "нет.xtc")
    monkeypatch.setitem(__import__("sys").modules, "MDAnalysis", None)
    with pytest.raises(MissingDependencyError, match=r"panda-md\[md\]"):
        load_trajectory("нет.gro")
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.trajectory'`.

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

test-env:  ## тесты в отдельном чистом окружении: только пакет с [md] и группа test
	UV_PROJECT_ENVIRONMENT=.venv-test uv sync --no-default-groups --group test --extra md
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
      # только пакет с [md] и группа test — тесты не должны зависеть от инструментов разработки
      - run: uv sync --no-default-groups --group test --extra md
      - run: uv run --no-sync pytest

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv build
      - run: uv run --isolated --no-project --with dist/*.whl python -c "import panda_md; print(panda_md.__version__)"
```

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
```

`src/panda_md/trajectory/__init__.py`:

```python
"""Источники кадров: массивы numpy и файлы МД (MDAnalysis), профили плотности по траектории."""
```

`src/panda_md/trajectory/arrays.py`:

```python
"""Траектория из массивов numpy — для синтетических данных и тестов (MDAnalysis не нужен)."""

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.trajectory import BOX_DIMENSIONS, Trajectory, TrajectoryFrame
from panda_md.core.types import Array1D, Array2D, Array3D, FloatArray, IndexArray, IndexArray1D, PandaModel


class ArrayTrajectory(PandaModel, Trajectory):
    """Кадры из массивов: позиции (кадры × атомы × 3, нм), ящики (кадры × 3, нм), времена (пс).

    Выборка — явные индексы или имя группы из `groups` (например, {"water": [...], "walls": [...]}).
    """

    positions_nm: Array3D
    boxes_nm: Array2D
    times_ps_values: Array1D = Field(alias="times_ps")
    groups: Mapping[str, IndexArray1D] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_shapes(self) -> "ArrayTrajectory":
        frame_count, atom_count, dimensions = self.positions_nm.shape
        if dimensions != BOX_DIMENSIONS or frame_count == 0:
            raise ValueError(f"позиции формы {self.positions_nm.shape}: нужно (кадры ≥ 1, атомы, 3)")
        if self.boxes_nm.shape != (frame_count, BOX_DIMENSIONS) or np.any(self.boxes_nm <= 0.0):
            raise ValueError(f"ящики формы {self.boxes_nm.shape}: нужно ({frame_count}, 3) с положительными длинами")
        if self.times_ps_values.shape != (frame_count,) or np.any(np.diff(self.times_ps_values) <= 0.0):
            raise ValueError("времена: по одному на кадр, строго по возрастанию")
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

`src/panda_md/trajectory/reader.py`:

```python
"""Чтение файлов МД через MDAnalysis (необязательная зависимость: `pip install panda-md[md]`).

MDAnalysis хранит длины в ångström; наружу отдаются нм. Единственный модуль пакета, импортирующий MDAnalysis.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError
from panda_md.core.trajectory import Trajectory, TrajectoryFrame, check_box
from panda_md.core.types import FloatArray, IndexArray

ANGSTROM_PER_NM = 10.0
RECTANGULAR_ANGLE_DEG = 90.0
ANGLE_TOLERANCE_DEG = 1e-3


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
        if indices.size == 0:
            raise InvalidTrajectoryError(f"выборка {selection!r} пуста — проверь имена остатков и атомов")
        return indices

    def times_ps(self) -> FloatArray:
        reader = self.universe.trajectory
        times: FloatArray = reader[0].time + reader.dt * np.arange(len(reader), dtype=np.float64)
        return times

    def _read(self, frame_indices: Sequence[int], groups: Sequence[IndexArray]) -> Iterator[TrajectoryFrame]:
        atoms = self.universe.atoms
        for timestep in self.universe.trajectory[list(frame_indices)]:
            dimensions = timestep.dimensions
            if dimensions is None or np.any(np.abs(dimensions[3:] - RECTANGULAR_ANGLE_DEG) > ANGLE_TOLERANCE_DEG):
                raise InvalidTrajectoryError(f"кадр {timestep.frame}: ящик не прямоугольный ({dimensions})")
            box_nm = np.asarray(dimensions[:3], dtype=np.float64) / ANGSTROM_PER_NM
            check_box(box_nm)
            positions = np.asarray(atoms.positions, dtype=np.float64) / ANGSTROM_PER_NM
            yield TrajectoryFrame(
                index=int(timestep.frame),
                time_ps=float(timestep.time),
                box_nm=box_nm,
                positions_nm=tuple(positions[group] for group in groups),
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

Run: `uv sync --all-extras` — ставит MDAnalysis в окружение разработки.

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `372 passed, 28 skipped` (тесты MDAnalysis не пропущены).

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add pyproject.toml Makefile .github/workflows/ci.yml src/panda_md/core/trajectory.py src/panda_md/trajectory tests/unit/test_trajectory.py
git commit -m "Траектория: интерфейс в ядре, кадры из массивов numpy и из файлов МД через MDAnalysis (panda-md[md])" \
  -m "MDAnalysis импортируется только внутри load_trajectory; длины переводятся из Å в нм, непрямоугольный ящик — ошибка." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 3: Пора в ящике моделирования — ось профиля, стенки, щель по траектории

**Files:**
- Modify: `src/panda_md/core/pore.py` (поля `normal_axis`, `wall_selection`; `bin_cross_section_nm2`, `base_area_nm2`, `coordinate_bounds_nm`, `locate_center_nm`), `src/panda_md/pores/slit/pore.py` (`from_trajectory`, поиск центра), `src/panda_md/pores/polygon.py` (сечение бина L × ly)
- Modify: `tests/unit/test_trajectory.py` (тесты щели)

**Interfaces:**
- Consumes: `Trajectory`, `largest_gap`, `Axis` (задачи 1–2).
- Produces: `Pore.normal_axis: Axis = Axis.Z`, `Pore.wall_selection: str | None = None`; абстрактный `bin_cross_section_nm2(box_nm) -> float`; `base_area_nm2() -> float | None` (щель — kL²); `coordinate_bounds_nm() -> (lower, upper)`; `locate_center_nm(wall_coordinates_nm, period_nm) -> float` (по умолчанию `NotDerivedError`). `SlitPore.from_trajectory(trajectory, *, walls, length_axis=Axis.X, normal_axis=Axis.Z, time=0.0) -> SlitPore`.

- [ ] **Шаг 1: Тесты**

Заменить `tests/unit/test_trajectory.py` целиком:

`tests/unit/test_trajectory.py`:

```python
"""Источники кадров (массивы numpy и MDAnalysis) и щель, построенная по атомам стенок."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from panda_md import SlitPore
from panda_md.core.enums import Axis
from panda_md.core.errors import InvalidTrajectoryError, MissingDependencyError
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.reader import MDATrajectory, load_trajectory

BOX_NM = np.array([6.0, 4.0, 10.0])
WALL_LAYERS_NM = (0.2, 0.55, 0.9)  # подложка у дна ящика; её образ сверху — граница поры через PBC
FRAME_COUNT = 5
TIME_STEP_PS = 2.0


def wall_atoms() -> np.ndarray:
    grid = np.array([[x, y] for x in np.arange(0.0, 6.0, 1.0) for y in np.arange(0.0, 4.0, 1.0)])
    return np.vstack([np.column_stack([grid, np.full(len(grid), z)]) for z in WALL_LAYERS_NM])


def slit_trajectory() -> ArrayTrajectory:
    walls = wall_atoms()
    liquid = np.column_stack([np.full(3, 1.0), np.full(3, 2.0), [3.0, 5.0, 7.0]])
    frame = np.vstack([walls, liquid])
    positions = np.repeat(frame[np.newaxis], FRAME_COUNT, axis=0)
    return ArrayTrajectory(
        positions_nm=positions,
        boxes_nm=np.tile(BOX_NM, (FRAME_COUNT, 1)),
        times_ps=TIME_STEP_PS * np.arange(FRAME_COUNT),
        groups={"walls": np.arange(len(walls)), "liquid": len(walls) + np.arange(3)},
    )


def test_array_trajectory_checks_shapes_and_order() -> None:
    with pytest.raises(ValidationError, match="ящики"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((3, 3)), times_ps=[0.0, 1.0])
    with pytest.raises(ValidationError, match="времена"):
        ArrayTrajectory(positions_nm=np.zeros((2, 3, 3)), boxes_nm=np.ones((2, 3)), times_ps=[1.0, 0.0])
    with pytest.raises(ValidationError, match="группа"):
        ArrayTrajectory(positions_nm=np.zeros((1, 3, 3)), boxes_nm=np.ones((1, 3)), times_ps=[0.0], groups={"g": [5]})


def test_selection_by_group_name_or_indices() -> None:
    trajectory = slit_trajectory()
    assert trajectory.select("liquid").tolist() == [72, 73, 74]
    assert trajectory.select([0, 1]).tolist() == [0, 1]
    with pytest.raises(InvalidTrajectoryError, match="группы"):
        trajectory.select("water")
    with pytest.raises(InvalidTrajectoryError, match="вне"):
        trajectory.select([1000])


def test_frames_are_chosen_by_time_and_stride() -> None:
    trajectory = slit_trajectory()
    assert trajectory.frame_indices() == [0, 1, 2, 3, 4]
    assert trajectory.frame_indices(start="0.002 ns", stop=6.0) == [1, 2, 3]
    assert trajectory.frame_indices(start=0.0, stride=2) == [0, 2, 4]
    assert trajectory.frame_at(3.0, [trajectory.select("liquid")]).index == 2
    with pytest.raises(InvalidTrajectoryError, match="нет кадров"):
        trajectory.frame_indices(start="1 ns")
    with pytest.raises(ValueError, match="stride"):
        trajectory.frame_indices(stride=0)


def test_slit_from_trajectory_finds_walls_through_box_boundary() -> None:
    pore = SlitPore.from_trajectory(slit_trajectory(), walls="walls")
    assert pore.width_nm == pytest.approx(BOX_NM[2] + WALL_LAYERS_NM[0] - WALL_LAYERS_NM[-1])
    assert pore.length_nm == pytest.approx(6.0)
    assert pore.base_aspect_ratio == pytest.approx(4.0 / 6.0)
    assert pore.wall_selection == "walls"
    assert pore.locate_center_nm(wall_atoms()[:, 2], BOX_NM[2]) == pytest.approx(
        (WALL_LAYERS_NM[-1] + BOX_NM[2] + WALL_LAYERS_NM[0]) / 2.0
    )


def test_slit_from_trajectory_rejects_wrong_axes() -> None:
    with pytest.raises(InvalidTrajectoryError, match="length_axis='x'"):
        SlitPore.from_trajectory(slit_trajectory(), walls="walls", length_axis=Axis.Y)
    with pytest.raises(InvalidTrajectoryError, match="совпадает"):
        SlitPore.from_trajectory(slit_trajectory(), walls="walls", length_axis=Axis.Z)


def memory_universe(dimensions: list[float]) -> Any:  # noqa: ANN401 — MDAnalysis без аннотаций типов
    mdanalysis = pytest.importorskip("MDAnalysis")
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mdanalysis.Universe.empty(4, n_residues=2, atom_resindex=[0, 0, 1, 1], trajectory=True)
    universe.add_TopologyAttr("names", ["C1", "C2", "OW", "HW"])
    universe.add_TopologyAttr("resnames", ["WALL", "WATER"])
    coordinates = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)  # Å
    universe.load_new(coordinates, format=MemoryReader, dimensions=dimensions, dt=2.0)
    return universe


def test_mdanalysis_trajectory_converts_angstrom_to_nm() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 90.0]))
    oxygen = trajectory.select("resname WATER and name OW")
    assert oxygen.tolist() == [2]
    np.testing.assert_allclose(trajectory.times_ps(), [0.0, 2.0, 4.0])
    frames = list(trajectory.frames([oxygen], start=2.0))
    assert [frame.index for frame in frames] == [1, 2]
    np.testing.assert_allclose(frames[0].positions_nm[0], [[1.8, 1.9, 2.0]])
    np.testing.assert_allclose(frames[0].box_nm, [5.0, 4.0, 10.0])
    with pytest.raises(InvalidTrajectoryError, match="пуста"):
        trajectory.select("resname DECAN")


def test_mdanalysis_trajectory_rejects_triclinic_box() -> None:
    trajectory = MDATrajectory(memory_universe([50.0, 40.0, 100.0, 90.0, 90.0, 60.0]))
    with pytest.raises(InvalidTrajectoryError, match="не прямоугольный"):
        list(trajectory.frames([trajectory.select("name OW")]))


def test_load_trajectory_reports_missing_files_and_package(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("MDAnalysis")
    with pytest.raises(InvalidTrajectoryError, match="не найден"):
        load_trajectory("нет.gro", "нет.xtc")
    monkeypatch.setitem(__import__("sys").modules, "MDAnalysis", None)
    with pytest.raises(MissingDependencyError, match=r"panda-md\[md\]"):
        load_trajectory("нет.gro")
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: `2 failed` — оба теста щели с `AttributeError: from_trajectory`.

- [ ] **Шаг 3: Реализация**

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
from typing import ClassVar

from pydantic import Field

from panda_md.core.enums import AXIS_ORDER, Axis
from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import largest_gap
from panda_md.core.pore import Pore
from panda_md.core.shape import PoreGeometry
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory
from panda_md.core.types import FloatArray

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

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Площадь основания ящика — произведение двух рёбер, перпендикулярных оси профиля."""
        return math.prod(float(box_nm[axis.column]) for axis in AXIS_ORDER if axis is not self.normal_axis)

    def base_area_nm2(self) -> float:
        """k L² — основание ячейки; должно совпадать с основанием ящика."""
        return self.base_aspect_ratio * self.length_nm**2

    def locate_center_nm(self, wall_coordinates_nm: FloatArray, period_nm: float) -> float:
        """Середина наибольшего пустого промежутка между атомами стенок, в [0, L) — в том числе через границу ящика."""
        start, end = largest_gap(wall_coordinates_nm, period_nm)
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
    ) -> "SlitPore":
        """Щель по кадру траектории: H — наибольший промежуток между атомами стенок (поверхности — по центрам
        крайних атомов), L — ребро ящика вдоль length_axis, k — второе ребро основания / L.
        """
        if length_axis is normal_axis:
            raise InvalidTrajectoryError(f"ось длины {length_axis.value} совпадает с осью профиля")
        indices = trajectory.select(walls)
        frame = trajectory.frame_at(time, [indices])
        box = frame.box_nm
        start, end = largest_gap(frame.positions_nm[0][:, normal_axis.column], float(box[normal_axis.column]))
        (other_axis,) = (axis for axis in AXIS_ORDER if axis not in (length_axis, normal_axis))
        length = float(box[length_axis.column])
        aspect_ratio = float(box[other_axis.column]) / length
        if aspect_ratio > 1.0:
            raise InvalidTrajectoryError(
                f"ребро {other_axis.value} ({box[other_axis.column]:.3f} нм) длиннее {length_axis.value} "
                f"({length:.3f} нм): L — длинная сторона основания, задай length_axis={other_axis.value!r}"
            )
        return cls(
            width_nm=end - start,
            length_nm=length,
            base_aspect_ratio=aspect_ratio,
            normal_axis=normal_axis,
            wall_selection=walls,
        )
```

Заменить `src/panda_md/pores/polygon.py` целиком:

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
from panda_md.core.types import FloatArray


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

    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Слой ящика L × ly, на который нормирован профиль (от ящика не зависит)."""
        return self.length_nm * self.layer_width_nm

    def characteristic_size_nm(self) -> float:
        return self.side_nm

    def volume_nm3(self) -> float:
        return self.geometry_class.section_area * self.side_nm**2 * self.length_nm
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `374 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/core/pore.py src/panda_md/pores tests/unit/test_trajectory.py
git commit -m "Пора в ящике: ось профиля и стенки; щель по траектории — H, L, k по атомам подложки через PBC" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 4: Профиль плотности по траектории — центрирование, бины, блоки

**Files:**
- Modify: `src/panda_md/core/profile.py` (`DensityProfile`: ошибка среднего, `flipped`, `frame_count`, `save`/`load`; новый `ProfileSet`)
- Create: `src/panda_md/trajectory/density.py`
- Create: `tests/unit/test_density_profile.py`

**Interfaces:**
- Consumes: `Trajectory`, `check_box` (задача 2), `Pore` в ящике (задача 3), `wrap_centered`, `circular_mean`, `to_picoseconds` (задача 1).
- Produces: `DensityProfile(..., density_sem_per_nm3=None, flipped=False, frame_count=None)`, `.save(path)`, `DensityProfile.load(path)` (.npz, .csv); `ProfileSet(bin_edges_nm, block_densities_per_nm3, block_start_ps, block_stop_ps, block_frame_counts, block_particle_counts, frame, centering, selection, pore_volume_nm3=None, flipped=False)` с `coordinate_nm`, `block_count`, `std()`, `sem()`, `mean()`, `block(index)`, `save`, `load`; `density_profile(trajectory, pore, select, *, centering="pore", bins=200, bin_width_nm=None, start=0, stop=None, block=None, stride=1, pore_center_nm=None, orientation="auto") -> ProfileSet`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_density_profile.py`:

```python
"""Профиль плотности по траектории: центрирование, бины, блоки, ориентация, проверки, сохранение."""

from pathlib import Path

import numpy as np
import pytest

from panda_md import SlitPore, TrianglePore
from panda_md.core.enums import Frame, Orientation, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError, NotDerivedError, PandaWarning
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile

WIDTH_NM = 4.0
LENGTH_NM = 6.0
ASPECT_RATIO = 0.5
BOX_NM = np.array([LENGTH_NM, ASPECT_RATIO * LENGTH_NM, 5.0])
LOWER_WALL_NM = 3.5  # стенки при z = 3.5 и z = 7.5 ≡ 2.5 (подложка — 1 нм): пора пересекает границу ящика
PORE_CENTER_NM = LOWER_WALL_NM + WIDTH_NM / 2.0
TIME_STEP_PS = 10.0
SEED = 7


def walls(frame_shift_nm: float = 0.0) -> np.ndarray:
    grid = np.array([[x, y] for x in np.arange(0.0, LENGTH_NM, 0.5) for y in np.arange(0.0, BOX_NM[1], 0.5)])
    lower = np.column_stack([grid, np.full(len(grid), LOWER_WALL_NM + frame_shift_nm)])
    upper = np.column_stack([grid, np.full(len(grid), np.mod(LOWER_WALL_NM + WIDTH_NM + frame_shift_nm, BOX_NM[2]))])
    return np.vstack([lower, upper])


def trajectory_from(
    liquid_z_per_frame: list[np.ndarray], frame_shifts_nm: list[float] | None = None
) -> ArrayTrajectory:
    """Жидкость — заданные z (в системе поры), x и y — равномерно; стенки — двойная решётка."""
    generator = np.random.default_rng(SEED)
    shifts = frame_shifts_nm or [0.0] * len(liquid_z_per_frame)
    frames = []
    for liquid_z, shift in zip(liquid_z_per_frame, shifts, strict=True):
        lateral = generator.uniform(0.0, 1.0, size=(liquid_z.size, 2)) * BOX_NM[:2]
        liquid = np.column_stack([lateral, np.mod(liquid_z + PORE_CENTER_NM + shift, BOX_NM[2])])
        frames.append(np.vstack([walls(shift), liquid]))
    wall_count = walls().shape[0]
    frame_count = len(frames)
    return ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.tile(BOX_NM, (frame_count, 1)),
        times_ps=TIME_STEP_PS * np.arange(frame_count),
        groups={"walls": np.arange(wall_count), "liquid": wall_count + np.arange(liquid_z_per_frame[0].size)},
    )


def pore() -> SlitPore:
    return SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO, wall_selection="walls")


def uniform_slab(lower_nm: float, upper_nm: float, count: int, frames: int) -> list[np.ndarray]:
    generator = np.random.default_rng(SEED)
    return [generator.uniform(lower_nm, upper_nm, count) for _ in range(frames)]


def test_uniform_slab_gives_flat_density_aligned_to_walls() -> None:
    count = 20000
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, count, 4))
    profiles = density_profile(trajectory, pore(), "liquid", bins=40)
    assert profiles.frame is Frame.PORE
    np.testing.assert_allclose(profiles.bin_edges_nm[[0, -1]], [-WIDTH_NM / 2, WIDTH_NM / 2])
    mean = profiles.mean()
    expected = count / (LENGTH_NM * ASPECT_RATIO * LENGTH_NM * 2.0)
    inside = np.abs(mean.coordinate_nm) < 1.0
    assert np.mean(mean.density_per_nm3[inside]) == pytest.approx(expected, rel=0.01)
    assert np.all(mean.density_per_nm3[np.abs(mean.coordinate_nm) > 1.05] == 0.0)
    assert mean.particle_count == count
    assert mean.pore_volume_nm3 == pytest.approx(pore().volume_nm3())


def test_walls_found_in_every_frame_follow_a_drifting_pore() -> None:
    slabs = uniform_slab(-1.9, -1.0, 5000, 3)  # слой у нижней стенки
    shifts = [0.0, 0.3, 0.6]
    by_walls = density_profile(trajectory_from(slabs, shifts), pore(), "liquid", bins=40, orientation=Orientation.KEEP)
    fixed = density_profile(
        trajectory_from(slabs), pore(), "liquid", bins=40, pore_center_nm=PORE_CENTER_NM, orientation=Orientation.KEEP
    )
    np.testing.assert_allclose(by_walls.mean().density_per_nm3, fixed.mean().density_per_nm3)


def test_liquid_centering_puts_center_of_mass_at_zero_across_boundary() -> None:
    trajectory = trajectory_from(uniform_slab(-0.5, 0.5, 4000, 2), [0.0, 0.7])
    profiles = density_profile(trajectory, pore(), "liquid", centering=ProfileCentering.LIQUID, bins=50)
    assert profiles.frame is Frame.LIQUID
    np.testing.assert_allclose(profiles.bin_edges_nm[[0, -1]], [-BOX_NM[2] / 2, BOX_NM[2] / 2])
    mean = profiles.mean()
    center_of_mass = np.sum(mean.coordinate_nm * mean.density_per_nm3) / np.sum(mean.density_per_nm3)
    assert center_of_mass == pytest.approx(0.0, abs=0.02)
    assert not profiles.flipped


def test_blocks_by_time_and_partial_block_is_dropped() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, 500, 7))
    with pytest.warns(PandaWarning, match="1 последних кадров"):
        profiles = density_profile(trajectory, pore(), "liquid", bins=20, block="20 ps")
    assert profiles.block_count == 3
    assert profiles.block_frame_counts == (2, 2, 2)
    np.testing.assert_allclose(profiles.block_start_ps, [0.0, 20.0, 40.0])
    np.testing.assert_allclose(profiles.block_stop_ps, [10.0, 30.0, 50.0])
    np.testing.assert_allclose(profiles.mean().density_per_nm3, profiles.block_densities_per_nm3.mean(axis=0))
    np.testing.assert_allclose(profiles.sem(), profiles.std() / np.sqrt(3))
    assert profiles.mean().density_sem_per_nm3 is not None
    window = density_profile(trajectory, pore(), "liquid", bins=20, start="0.02 ns", stop=40.0, stride=2)
    assert window.block_frame_counts == (2,)
    with pytest.raises(InvalidTrajectoryError, match="короче одного блока"):
        density_profile(trajectory, pore(), "liquid", bins=20, block="1 ns")


def test_liquid_at_upper_wall_is_flipped_in_pore_frame() -> None:
    trajectory = trajectory_from(uniform_slab(1.0, 1.9, 3000, 2))
    flipped = density_profile(trajectory, pore(), "liquid", bins=40)
    kept = density_profile(trajectory, pore(), "liquid", bins=40, orientation=Orientation.KEEP)
    assert flipped.flipped and not kept.flipped
    np.testing.assert_allclose(flipped.mean().density_per_nm3, kept.mean().density_per_nm3[::-1])
    assert flipped.mean().flipped


def test_bin_width_sets_bin_count() -> None:
    profiles = density_profile(trajectory_from(uniform_slab(-1.0, 1.0, 100, 1)), pore(), "liquid", bin_width_nm=0.1)
    assert profiles.block_densities_per_nm3.shape == (1, 40)


def test_problems_with_pore_and_selection_are_reported() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 1.0, 100, 1))
    no_walls = SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO)
    with pytest.raises(InvalidTrajectoryError, match="pore_center_nm"):
        density_profile(trajectory, no_walls, "liquid")
    too_wide = SlitPore(width_nm=12.0, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO)
    with pytest.raises(InvalidTrajectoryError, match="шире ящика"):
        density_profile(trajectory, too_wide, "liquid", pore_center_nm=PORE_CENTER_NM)
    triangle = TrianglePore(side_nm=4.0, length_nm=6.0, layer_width_nm=4.0, wall_selection="walls")
    with pytest.raises(NotDerivedError, match="pore_center_nm"):
        density_profile(trajectory, triangle, "liquid")
    with pytest.raises(InvalidTrajectoryError, match="пуста"):
        density_profile(trajectory, pore(), [])


def test_suspicious_geometry_gives_warnings() -> None:
    trajectory = trajectory_from(uniform_slab(-1.0, 3.0, 1000, 1))  # четверть частиц — за верхней стенкой
    with pytest.warns(PandaWarning, match="вне поры"):
        density_profile(trajectory, pore(), "liquid", bins=20)
    inside = trajectory_from(uniform_slab(-1.0, 1.0, 1000, 1))
    narrow = SlitPore(width_nm=3.5, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO, wall_selection="walls")
    with pytest.warns(PandaWarning, match="промежуток между стенками"):
        density_profile(inside, narrow, "liquid", bins=20)
    other_base = SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, wall_selection="walls")
    with pytest.warns(PandaWarning, match="площадь бина"):
        density_profile(inside, other_base, "liquid", bins=20)


def test_profiles_survive_saving(tmp_path: Path) -> None:
    profiles = density_profile(trajectory_from(uniform_slab(-1.0, 1.0, 800, 4)), pore(), "liquid", block="20 ps")
    profiles.save(tmp_path / "set.npz")
    assert ProfileSet.load(tmp_path / "set.npz").model_dump_json() == profiles.model_dump_json()
    mean = profiles.mean()
    for name in ("mean.npz", "mean.csv"):
        mean.save(tmp_path / name)
        loaded = DensityProfile.load(tmp_path / name)
        np.testing.assert_allclose(loaded.density_per_nm3, mean.density_per_nm3, rtol=1e-12)
        assert loaded.model_dump(exclude={"coordinate_nm", "density_per_nm3", "density_sem_per_nm3"}) == (
            mean.model_dump(exclude={"coordinate_nm", "density_per_nm3", "density_sem_per_nm3"})
        )
    with pytest.raises(ValueError, match="формат"):
        mean.save(tmp_path / "mean.txt")


def test_centering_and_orientation_accept_strings() -> None:
    trajectory = trajectory_from(uniform_slab(1.0, 1.9, 500, 1))
    by_name = density_profile(trajectory, pore(), "liquid", centering="pore", orientation="keep", bins=20)
    by_enum = density_profile(
        trajectory, pore(), "liquid", centering=ProfileCentering.PORE, orientation=Orientation.KEEP, bins=20
    )
    np.testing.assert_allclose(by_name.block_densities_per_nm3, by_enum.block_densities_per_nm3)
    with pytest.raises(ValueError, match="none"):
        density_profile(trajectory, pore(), "liquid", centering="none")
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ImportError: cannot import name 'ProfileSet' from 'panda_md.core.profile'`.

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/core/profile.py` целиком:

`src/panda_md/core/profile.py`:

```python
"""Профиль плотности из данных, набор профилей по блокам времени и безразмерный вид профиля."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.enums import Frame, Normalization, ProfileCentering
from panda_md.core.errors import InvalidProfileError
from panda_md.core.types import Array1D, Array2D, FloatArray, PandaModel

MIN_PROFILE_POINTS = 2
CSV_COMMENT = "# "
CSV_METADATA_FIELDS = ("frame", "particle_count", "pore_volume_nm3", "flipped", "frame_count")


class DensityProfile(PandaModel):
    """Профиль плотности анализируемой жидкости вдоль оси профиля поры."""

    coordinate_nm: Array1D  # центры бинов, нм, в системе frame
    density_per_nm3: Array1D  # число частиц на нм³
    density_sem_per_nm3: Array1D | None = None  # ошибка среднего по блокам
    frame: Frame = Frame.PORE
    particle_count: float | None = Field(default=None, gt=0.0)  # N — частиц в поре
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)  # V — объём поры, к которому отнесена φ
    flipped: bool = False  # профиль отражён z → −z (жидкость была у верхней стенки)
    frame_count: int | None = Field(default=None, ge=1)  # сколько кадров усреднено

    @model_validator(mode="after")
    def _check_arrays(self) -> "DensityProfile":
        if self.coordinate_nm.shape != self.density_per_nm3.shape:
            raise ValueError("длины координаты и плотности различаются")
        if self.density_sem_per_nm3 is not None and self.density_sem_per_nm3.shape != self.density_per_nm3.shape:
            raise ValueError("длины плотности и её ошибки различаются")
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

    def save(self, path: str | Path) -> None:
        """Сохранить: .npz — полностью; .csv — столбцы и метаданные в строках-комментариях."""
        target = Path(path)
        if target.suffix == ".npz":
            _save_npz(target, _profile_arrays(self), self._metadata())
        elif target.suffix == ".csv":
            header = "\n".join(f"{key}: {json.dumps(value)}" for key, value in self._metadata().items())
            arrays = _profile_arrays(self)
            columns = list(arrays)
            table = np.column_stack([arrays[name] for name in columns])
            np.savetxt(target, table, delimiter=",", header=header + "\n" + ",".join(columns), comments=CSV_COMMENT)
        else:
            raise ValueError(f"формат {target.suffix!r} не поддерживается: .npz или .csv")

    @classmethod
    def load(cls, path: str | Path) -> "DensityProfile":
        source = Path(path)
        if source.suffix == ".npz":
            with np.load(source) as archive:
                metadata = json.loads(str(archive["metadata"]))
                arrays = {name: archive[name] for name in archive.files if name != "metadata"}
        elif source.suffix == ".csv":
            metadata, arrays = _read_csv(source)
        else:
            raise ValueError(f"формат {source.suffix!r} не поддерживается: .npz или .csv")
        return cls(**arrays, **metadata)

    def _metadata(self) -> dict[str, object]:
        dumped = self.model_dump(mode="json")
        return {name: dumped[name] for name in CSV_METADATA_FIELDS}


def _save_npz(path: Path, arrays: Mapping[str, FloatArray], metadata: Mapping[str, object]) -> None:
    """Массивы — по именам, метаданные — JSON-строкой под ключом metadata."""
    # Any: заглушки numpy типизируют **kwds у savez как bool (allow_pickle); значения — массивы и строка
    payload: dict[str, Any] = {**arrays, "metadata": json.dumps(dict(metadata))}
    np.savez(path, **payload)


def _profile_arrays(profile: DensityProfile) -> dict[str, FloatArray]:
    arrays = {"coordinate_nm": profile.coordinate_nm, "density_per_nm3": profile.density_per_nm3}
    if profile.density_sem_per_nm3 is not None:
        arrays["density_sem_per_nm3"] = profile.density_sem_per_nm3
    return arrays


def _read_csv(source: Path) -> tuple[dict[str, object], dict[str, FloatArray]]:
    metadata: dict[str, object] = {}
    columns: list[str] = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith(CSV_COMMENT):
                break
            content = line[len(CSV_COMMENT) :].strip()
            key, separator, value = content.partition(": ")
            if separator and key in CSV_METADATA_FIELDS:
                metadata[key] = json.loads(value)
            else:
                columns = content.split(",")
    table = np.loadtxt(source, delimiter=",", comments=CSV_COMMENT.strip(), ndmin=2)
    return metadata, {name: table[:, index] for index, name in enumerate(columns)}


class ProfileSet(PandaModel):
    """Профили по блокам времени: блоки × бины. Среднее по блокам — профиль для фита, разброс — его ошибка."""

    bin_edges_nm: Array1D  # края бинов, нм, в системе frame
    block_densities_per_nm3: Array2D  # блоки × бины
    block_start_ps: Array1D
    block_stop_ps: Array1D
    block_frame_counts: tuple[int, ...]
    block_particle_counts: Array1D  # среднее по кадрам блока число частиц в поре (N)
    frame: Frame
    centering: ProfileCentering
    selection: str
    pore_volume_nm3: float | None = Field(default=None, gt=0.0)
    flipped: bool = False

    @model_validator(mode="after")
    def _check_shapes(self) -> "ProfileSet":
        block_count, bin_count = self.block_densities_per_nm3.shape
        if self.bin_edges_nm.shape != (bin_count + 1,) or not np.all(np.diff(self.bin_edges_nm) > 0):
            raise ValueError("края бинов: по одному больше, чем бинов, строго по возрастанию")
        for name in ("block_start_ps", "block_stop_ps", "block_particle_counts"):
            if getattr(self, name).shape != (block_count,):
                raise ValueError(f"{name}: по одному значению на блок")
        if len(self.block_frame_counts) != block_count or min(self.block_frame_counts, default=0) < 1:
            raise ValueError("в каждом блоке должен быть хотя бы один кадр")
        return self

    @property
    def coordinate_nm(self) -> FloatArray:
        centers: FloatArray = 0.5 * (self.bin_edges_nm[:-1] + self.bin_edges_nm[1:])
        return centers

    @property
    def block_count(self) -> int:
        return int(self.block_densities_per_nm3.shape[0])

    def std(self) -> FloatArray:
        """Стандартное отклонение плотности по блокам (0 при одном блоке)."""
        if self.block_count < 2:
            return np.zeros(self.block_densities_per_nm3.shape[1])
        spread: FloatArray = np.std(self.block_densities_per_nm3, axis=0, ddof=1)
        return spread

    def sem(self) -> FloatArray:
        """Ошибка среднего по блокам: std / √(число блоков)."""
        error: FloatArray = self.std() / np.sqrt(self.block_count)
        return error

    def mean(self) -> DensityProfile:
        """Средний профиль по блокам с ошибкой среднего."""
        return self._profile(
            np.mean(self.block_densities_per_nm3, axis=0),
            float(np.mean(self.block_particle_counts)),
            sum(self.block_frame_counts),
            self.sem() if self.block_count > 1 else None,
        )

    def block(self, index: int) -> DensityProfile:
        return self._profile(
            self.block_densities_per_nm3[index],
            float(self.block_particle_counts[index]),
            self.block_frame_counts[index],
            None,
        )

    def _profile(
        self, density: FloatArray, particle_count: float, frame_count: int, sem: FloatArray | None
    ) -> DensityProfile:
        return DensityProfile(
            coordinate_nm=self.coordinate_nm,
            density_per_nm3=density,
            density_sem_per_nm3=sem,
            frame=self.frame,
            particle_count=particle_count if particle_count > 0 else None,
            pore_volume_nm3=self.pore_volume_nm3,
            flipped=self.flipped,
            frame_count=frame_count,
        )

    def save(self, path: str | Path) -> None:
        """Сохранить в .npz (массивы и метаданные)."""
        arrays = {
            name: getattr(self, name)
            for name in (
                "bin_edges_nm",
                "block_densities_per_nm3",
                "block_start_ps",
                "block_stop_ps",
                "block_particle_counts",
            )
        }
        metadata = self.model_dump(mode="json", exclude=set(arrays))
        _save_npz(Path(path), arrays, metadata)

    @classmethod
    def load(cls, path: str | Path) -> "ProfileSet":
        with np.load(Path(path)) as archive:
            metadata = json.loads(str(archive["metadata"]))
            arrays = {name: archive[name] for name in archive.files if name != "metadata"}
        return cls(**arrays, **metadata)


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

`src/panda_md/trajectory/density.py`:

```python
"""Профиль плотности по траектории: выборка → центр в каждом кадре → координата и бины поры → средние по блокам.

Чтение потоковое: в памяти только текущий кадр и суммы по блокам.
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np

from panda_md.core.enums import Frame, Orientation, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError, PandaWarning
from panda_md.core.pbc import circular_mean, wrap_centered
from panda_md.core.pore import Pore
from panda_md.core.profile import ProfileSet
from panda_md.core.timing import TimeValue, to_picoseconds
from panda_md.core.trajectory import Trajectory, TrajectoryFrame, check_box
from panda_md.core.types import FloatArray, IndexArray

DEFAULT_BIN_COUNT = 200
WALL_GAP_TOLERANCE = 0.02  # промежуток между стенками в кадре отличается от H поры больше чем на 2 % — предупреждение
BOX_AREA_TOLERANCE = 0.01  # площадь основания ящика отличается от основания поры больше чем на 1 % — предупреждение
OUTSIDE_FRACTION_WARNING = 0.01  # больше 1 % выбранных частиц вне поры — предупреждение


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
) -> ProfileSet:
    """Профили плотности выбранных частиц вдоль оси поры, усреднённые по блокам времени длины `block`.

    centering=PORE: 0 — центр поры; он берётся из `pore_center_nm` или в каждом кадре из атомов стенок
    (`pore.wall_selection`); бины выровнены по стенкам поры. centering=LIQUID: 0 — центр масс выборки с учётом PBC;
    бины покрывают весь ящик вдоль оси профиля. Неполный последний блок отбрасывается.
    """
    centering = ProfileCentering(centering)
    orientation = Orientation(orientation)
    liquid = trajectory.select(select)
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
    blocks: list[_BlockAccumulator] = []
    warned: set[str] = set()
    for frame, block_index in zip(
        trajectory.frames(groups, start=start, stop=stop, stride=stride), block_of_frame, strict=True
    ):
        if block_index < 0:
            continue
        check_box(frame.box_nm)
        period = float(frame.box_nm[axis])
        center = _frame_center(frame, pore, centering, pore_center_nm, period, warned)
        coordinate = wrap_centered(frame.positions_nm[0][:, axis] - center, period)
        if edges is None:
            edges = _bin_edges(pore, centering, period, bins, bin_width_nm)
            blocks = [_BlockAccumulator(edges.size - 1) for _ in range(block_count)]
        counts, _ = np.histogram(coordinate, bins=edges)
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
    spacing = float(np.median(np.diff(times))) if times.size > 1 else block_ps
    elapsed = times - times[0]
    block_count = math.floor((elapsed[-1] + spacing) / block_ps + 1e-9)
    if block_count == 0:
        raise InvalidTrajectoryError(
            f"интервал {elapsed[-1] + spacing:g} пс короче одного блока {block_ps:g} пс — уменьши block"
        )
    assignment = [math.floor(value / block_ps + 1e-9) for value in elapsed]
    dropped = sum(1 for index in assignment if index >= block_count)
    if dropped:
        warnings.warn(f"{dropped} последних кадров не составляют полного блока и не учтены", PandaWarning, stacklevel=3)
    return [index if index < block_count else -1 for index in assignment], block_count


def _frame_center(
    frame: TrajectoryFrame,
    pore: Pore,
    centering: ProfileCentering,
    pore_center_nm: float | None,
    period: float,
    warned: set[str],
) -> float:
    axis = pore.normal_axis.column
    if centering is ProfileCentering.LIQUID:
        return circular_mean(frame.positions_nm[0][:, axis], period)
    if pore_center_nm is not None:
        return pore_center_nm
    walls = frame.positions_nm[1][:, axis]
    center = pore.locate_center_nm(walls, period)
    gap = _wall_gap(walls, period, center)
    expected = pore.coordinate_bounds_nm()[1] - pore.coordinate_bounds_nm()[0]
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
    return float(np.min(offsets[offsets > 0]) - np.max(offsets[offsets <= 0]))


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


def _check_area(pore: Pore, area: float, warned: set[str]) -> None:
    base = pore.base_area_nm2()
    if base is None:
        return
    if abs(area - base) > BOX_AREA_TOLERANCE * base and "area" not in warned:
        warned.add("area")
        warnings.warn(
            f"площадь бина {area:.3f} нм² отличается от основания поры {base:.3f} нм² — проверь размеры поры",
            PandaWarning,
            stacklevel=4,
        )
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `384 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/core/profile.py src/panda_md/trajectory/density.py tests/unit/test_density_profile.py
git commit -m "Профиль по траектории: центрирование по поре или жидкости, бины по стенкам, блоки по времени, ProfileSet" \
  -m "Чтение потоковое; неполный последний блок отбрасывается; предупреждения при частицах вне поры и несовпадении размеров; сохранение .npz и .csv." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 5: Публичный API и сквозной путь «частицы → профиль → фит»

**Files:**
- Modify: `src/panda_md/__init__.py`, `tests/test_package.py`
- Create: `tests/synthetic/test_data_path.py`

**Interfaces:**
- Consumes: всё из задач 1–4.
- Produces: в `panda_md`: `ArrayTrajectory`, `Axis`, `Orientation`, `ProfileCentering`, `ProfileSet`, `density_profile`, `load_trajectory`.

- [ ] **Шаг 1: Тесты**

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
```

`tests/synthetic/test_data_path.py`:

```python
"""Сквозной путь: частицы из модели → траектория → центрирование и бины → профиль по блокам → фит.

Координата z частиц разыгрывается по профилю модели (одномерное распределение — всё, что видит профиль плотности),
x и y — равномерно. Стенки — две плоскости атомов, пора пересекает границу ящика и дрейфует между кадрами.
Проверяется путь данных, а не трёхмерная геометрия формы (облака точек по `contains` — фаза 6).
"""

import numpy as np
import pytest

from panda_md import ArrayTrajectory, InterfaceModel, ProfileCentering, SlitPore, density_profile

DENSITY_BULK = 33.0
WIDTH_NM = 4.0
LENGTH_NM = 8.0
ASPECT_RATIO = 0.5
BOX_NM = np.array([LENGTH_NM, ASPECT_RATIO * LENGTH_NM, 5.0])  # подложка — 1 нм
LOWER_WALL_NM = 3.2
FRAME_COUNT = 40
TIME_STEP_PS = 5.0
SAMPLING_POINTS = 20001
SEED = 11
MAX_ANGLE_ERROR_DEG = 1.5


def sample_heights(
    model: InterfaceModel, values: dict[str, float], count: int, generator: np.random.Generator
) -> np.ndarray:
    """z в системе поры (нм) с плотностью ∝ ρ_model(z): обратная функция распределения на мелкой сетке."""
    lower, upper = model.pore.geometry().coordinate_bounds()
    grid = np.linspace(lower, upper, SAMPLING_POINTS)
    density = model.profile(grid, **values)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    heights = np.interp(generator.uniform(0.0, cumulative[-1], count), cumulative, grid)
    return heights * model.pore.characteristic_size_nm()


def particle_trajectory(model: InterfaceModel, values: dict[str, float]) -> ArrayTrajectory:
    generator = np.random.default_rng(SEED)
    pore = model.pore
    count = round(values["phi"] * DENSITY_BULK * pore.volume_nm3())
    grid = np.array([[x, y] for x in np.arange(0.0, BOX_NM[0], 0.4) for y in np.arange(0.0, BOX_NM[1], 0.4)])
    frames = []
    for frame_index in range(FRAME_COUNT):
        drift = 0.02 * frame_index  # стенки и жидкость вместе дрейфуют вдоль z
        lower = LOWER_WALL_NM + drift
        walls = np.vstack(
            [
                np.column_stack([grid, np.full(len(grid), lower)]),
                np.column_stack([grid, np.full(len(grid), lower + WIDTH_NM)]),
            ]
        )
        lateral = generator.uniform(0.0, 1.0, size=(count, 2)) * BOX_NM[:2]
        heights = sample_heights(model, values, count, generator) + lower + WIDTH_NM / 2.0
        frames.append(np.mod(np.vstack([walls, np.column_stack([lateral, heights])]), BOX_NM))
    wall_count = 2 * len(grid)
    return ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.tile(BOX_NM, (FRAME_COUNT, 1)),
        times_ps=TIME_STEP_PS * np.arange(FRAME_COUNT),
        groups={"walls": np.arange(wall_count), "liquid": wall_count + np.arange(count)},
    )


@pytest.mark.parametrize(
    ("shape_name", "truth", "centering"),
    [
        ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}, ProfileCentering.PORE),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}, ProfileCentering.PORE),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}, ProfileCentering.LIQUID),
    ],
    ids=["roll-pore", "droplet-pore", "droplet-liquid"],
)
def test_angle_recovered_from_particles(shape_name: str, truth: dict[str, float], centering: ProfileCentering) -> None:
    pore = SlitPore(width_nm=WIDTH_NM, length_nm=LENGTH_NM, base_aspect_ratio=ASPECT_RATIO, wall_selection="walls")
    model = pore.interface(shape_name)
    trajectory = particle_trajectory(model, truth)
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    assert found.width_nm == pytest.approx(WIDTH_NM)
    profiles = density_profile(trajectory, pore, "liquid", centering=centering, bins=100, block="50 ps")
    assert profiles.block_count == 4
    optimize = ["theta", "delta"] if centering is ProfileCentering.PORE else ["theta"]
    fixed = {} if centering is ProfileCentering.PORE else {"delta": truth["delta"]}
    result = model.fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK, optimize=optimize, fixed=fixed)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)
    assert result.values["phi"] == pytest.approx(truth["phi"], rel=0.02)
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `tests/synthetic/test_data_path.py` — `ImportError: cannot import name 'ArrayTrajectory' from 'panda_md'`.

- [ ] **Шаг 3: Реализация**

Заменить `src/panda_md/__init__.py` целиком:

`src/panda_md/__init__.py`:

```python
"""PANDA: Predicting Angle from Nanoscale Density Analysis."""

from importlib.metadata import PackageNotFoundError, version

from panda_md.core.enums import (
    Axis,
    Frame,
    LossFunction,
    Normalization,
    OptimizerMethod,
    Orientation,
    ParameterName,
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
    "ProfileCentering",
    "ProfileSet",
    "Regime",
    "SlitPore",
    "SquarePore",
    "TrianglePore",
    "Unit",
    "__version__",
    "density_profile",
    "load_trajectory",
]
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `389 passed, 28 skipped`; в `test_data_path` θ восстанавливается с отклонением 0.2–0.8°.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/__init__.py tests/test_package.py tests/synthetic/test_data_path.py
git commit -m "Путь данных в публичном API; сквозной тест: частицы из модели → траектория → профиль → фит" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 6: Сверка со старой PANDA и реальная МД

**Files:**
- Modify: `.gitignore` (эталонные `.npz` тестов в репозитории)
- Create: `tests/regression/generate_density_reference.py`, `tests/regression/data/density_reference.npz`
- Modify: `tests/regression/test_reference_formulas.py`
- Create: `tests/md/generate_md_profiles.py`, `tests/md/data/{graphite,benzene,calcite}_pore.json`, `tests/md/data/graphite_water_profiles.npz`, `tests/md/data/benzene_profiles.npz`, `tests/md/data/calcite_{decane,water}_profiles.npz`, `tests/md/test_real_systems.py`

**Interfaces:**
- Consumes: `ArrayTrajectory`, `density_profile`, `ProfileSet.load`, `SlitPore.from_trajectory`, `load_trajectory` (задачи 2–5).
- Produces: эталон старой PANDA (`positions_nm`, `boxes_nm`, `times_ps`, `bin_count`, `density_per_nm3`, `source`); `ProfileSet` и пора (JSON) для трёх систем: вода в щели графита (8 блоков по 500 кадров), бензол на SOLD (5 блоков по 600 кадров), декан и вода в щели кальцита (один кадр).

- [ ] **Шаг 1: Тесты и генераторы**

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
from panda_md.core.enums import ProfileCentering
from panda_md.pores.polygon import PolygonPore
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile

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


def test_density_profile_matches_old_panda() -> None:
    """Центрирование по жидкости и бины по ящику — как utils.get_center_pbc + get_numerical_density_profile."""
    with np.load(DATA / "density_reference.npz") as reference:
        trajectory = ArrayTrajectory(
            positions_nm=reference["positions_nm"], boxes_nm=reference["boxes_nm"], times_ps=reference["times_ps"]
        )
        box = reference["boxes_nm"][0]
        pore = SlitPore(width_nm=float(box[2]), length_nm=float(box[0]), base_aspect_ratio=float(box[1] / box[0]))
        profiles = density_profile(
            trajectory,
            pore,
            list(range(reference["positions_nm"].shape[1])),
            centering=ProfileCentering.LIQUID,
            bins=int(reference["bin_count"]),
        )
        np.testing.assert_allclose(
            profiles.mean().density_per_nm3, reference["density_per_nm3"], rtol=0.0, atol=TOLERANCE
        )
```

`tests/md/test_real_systems.py`:

```python
"""Реальная МД: профили, посчитанные новым путём данных по трём системам (`generate_md_profiles.py`).

Ожидания — независимые: ручной расчёт стенок и результаты задачи 1 (docs/task1_current_panda.md).
- graphite: вода — несмачивающая перемычка (roll), θ ≈ 172° ± 7° по блокам; по 2D-карте вода отделена от графита
  плёнкой декана, поэтому θ близок к 180°;
- benzene: бензол на подложке SOLD смачивает её (плотность у стенок выше, чем в середине) — θ < 90°, формул
  смачивающего режима нет (O1), фит обязан упереться в границу режима и поднять флаг;
- calcite: стартовый кадр, подложка толстая и из разных атомов — проверяется только поиск стенок и то, что вся
  жидкость в поре.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid

from panda_md import Frame, ProfileSet, SlitPore

DATA = Path(__file__).parent / "data"
WATER_DENSITY_BULK = 33.13  # ρ_bulk кислородов воды в графитовой системе: медиана 2D-карты в сердцевине фазы (задача 1)


def load_pore(name: str) -> SlitPore:
    return SlitPore.model_validate_json((DATA / f"{name}_pore.json").read_text())


def test_walls_are_found_as_computed_by_hand() -> None:
    """H — наибольший промежуток между атомами подложки через PBC (крайние атомы и длины ящиков — из .gro)."""
    graphite = load_pore("graphite")
    assert graphite.width_nm == pytest.approx(11.0 + 0.195 - 2.926, abs=1e-3)  # образ дна графита − его верх
    assert graphite.base_aspect_ratio == pytest.approx(5.12817 / 17.27104, abs=1e-5)
    benzene = load_pore("benzene")
    assert benzene.width_nm == pytest.approx(13.4607 - 3.147, abs=1e-3)  # задача 1: H = Lz − верх SOLD = 10.31
    calcite = load_pore("calcite")
    assert calcite.width_nm == pytest.approx(12.684 + 0.009 - 3.554, abs=1e-3)


@pytest.mark.parametrize(
    ("file_name", "particles"),
    [
        ("graphite_water_profiles.npz", 11807),
        ("benzene_profiles.npz", 16584),
        ("calcite_decane_profiles.npz", 13400),
        ("calcite_water_profiles.npz", 14458),
    ],
)
def test_whole_liquid_is_inside_the_pore(file_name: str, particles: int) -> None:
    profiles = ProfileSet.load(DATA / file_name)
    assert profiles.frame is Frame.PORE
    np.testing.assert_allclose(profiles.block_particle_counts, particles)


def test_graphite_water_is_a_non_wetting_roll_close_to_180_degrees() -> None:
    pore = load_pore("graphite")
    profiles = ProfileSet.load(DATA / "graphite_water_profiles.npz")
    assert profiles.block_frame_counts == (500,) * 8
    mean = profiles.mean()
    phi = trapezoid(mean.density_per_nm3, mean.coordinate_nm) / (WATER_DENSITY_BULK * pore.width_nm)
    assert phi == pytest.approx(0.487, abs=0.005)  # задача 1: φ = 0.487
    roll = pore.interface("roll")
    result = roll.fit(mean, density_bulk_per_nm3=WATER_DENSITY_BULK)
    assert 165.0 < result.values["theta"] <= 180.0
    assert 0.02 < result.values["delta"] < 0.1
    assert result.loss < 0.03
    block_angles = [
        roll.fit(profiles.block(index), density_bulk_per_nm3=WATER_DENSITY_BULK).values["theta"] for index in range(8)
    ]
    assert 165.0 < float(np.mean(block_angles)) < 178.0
    assert float(np.std(block_angles)) < 10.0


@pytest.mark.parametrize("density_bulk", [10.0, 15.0, 20.0, 25.0])
def test_benzene_wets_its_substrate_and_fit_flags_the_regime_boundary(density_bulk: float) -> None:
    """ρ_bulk бензола в этой системе неизвестна, поэтому вывод проверяется на всём разумном диапазоне."""
    pore = load_pore("benzene")
    profiles = ProfileSet.load(DATA / "benzene_profiles.npz")
    mean = profiles.mean()
    near_walls = mean.density_per_nm3[np.abs(mean.coordinate_nm) > 0.4 * pore.width_nm]
    middle = mean.density_per_nm3[np.abs(mean.coordinate_nm) < 0.1 * pore.width_nm]
    assert near_walls[near_walls > 0].mean() > 1.3 * middle.mean()  # вогнутая перемычка: шире у стенок
    result = pore.interface("roll").fit(mean, density_bulk_per_nm3=density_bulk)
    assert result.at_regime_boundary
    assert result.values["theta"] == pytest.approx(90.5, abs=0.1)
```

`tests/regression/generate_density_reference.py`:

```python
"""Эталон профиля плотности по траектории из старой PANDA (`utils.py`, `density_profile.py`, коммит 93c6dbb).

Старый путь: центр жидкости по круговому среднему (`get_center_pbc`), сдвиг в середину ящика и заворот
(`apply_pbc`), бины по всему ящику (`get_numerical_density_profile`). Новый аналог — центрирование по жидкости
(система LIQUID). Вход — случайное облако точек с фиксированным зерном: плотная полоса, пересекающая границу
ящика, и разреженный пар.

Запуск (нужно окружение со старой PANDA, см. ROADMAP «Фаза 0»):
    .venv-legacy/bin/python tests/regression/generate_density_reference.py
Результат: tests/regression/data/density_reference.npz — коммитится, тесты читают только его.
"""

import importlib
from pathlib import Path

import numpy as np

utils = importlib.import_module("panda.utils")
old_density = importlib.import_module("panda.density_profile")

OUTPUT = Path(__file__).parent / "data" / "density_reference.npz"
SEED = 20261016
FRAME_COUNT = 4
BOX_NM = np.array([4.0, 3.0, 10.0])
LIQUID_PARTICLES = 1500
VAPOUR_PARTICLES = 100
SLAB_HALF_WIDTH_NM = 1.5
BIN_COUNT = 80
TIME_STEP_PS = 2.0


def make_positions(generator: np.random.Generator) -> np.ndarray:
    """Кадры: полоса жидкости вокруг z ≈ 9.5 нм (пересекает верхнюю границу ящика) и пар по всему ящику."""
    frames = []
    for frame_index in range(FRAME_COUNT):
        slab_center = 9.5 + 0.1 * frame_index  # жидкость дрейфует — центр в каждом кадре свой
        liquid = generator.uniform(0.0, 1.0, size=(LIQUID_PARTICLES, 3)) * BOX_NM
        liquid[:, 2] = np.mod(
            slab_center + generator.uniform(-1.0, 1.0, LIQUID_PARTICLES) * SLAB_HALF_WIDTH_NM, BOX_NM[2]
        )
        vapour = generator.uniform(0.0, 1.0, size=(VAPOUR_PARTICLES, 3)) * BOX_NM
        frames.append(np.vstack([liquid, vapour]))
    return np.array(frames)


def main() -> None:
    positions = make_positions(np.random.default_rng(SEED))
    boxes = np.tile(BOX_NM, (FRAME_COUNT, 1))
    box = boxes[:, np.newaxis, :]
    centered = utils.apply_pbc(positions - utils.get_center_pbc(positions, box) + box / 2, box)
    profiles = [
        old_density.get_numerical_density_profile(centered[index], box[index], BIN_COUNT, center=True)[1]
        for index in range(FRAME_COUNT)
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        positions_nm=positions,
        boxes_nm=boxes,
        times_ps=TIME_STEP_PS * np.arange(FRAME_COUNT),
        bin_count=BIN_COUNT,
        density_per_nm3=np.mean(profiles, axis=0),
        source="github.com/FlufffyMelon/PANDA@93c6dbb panda/utils.py, panda/density_profile.py",
    )
    print(f"записано в {OUTPUT}")


if __name__ == "__main__":
    main()
```

`tests/md/generate_md_profiles.py`:

```python
"""Профили по реальным траекториям МД — вход для tests/md (сами траектории в репозиторий не входят).

Три системы (найденная пора сохраняется рядом с профилями, чтобы тесты сверили её с ручным расчётом):
- graphite: щель графита с деканом и водой TIP4P (PANDA_Reborn graphite_decane_tip4p_1.0), 10 нс, кадр — 2 пс;
  профиль воды (кислороды) после 2 нс, блоки по 1 нс;
- benzene: бензол в щели подложки SOLD (roughness_2/angles/base), 10 нс, кадр — 2 пс; после 4 нс, блоки по 1.2 нс;
- calcite: щель кальцита с деканом и водой (система примера старой PANDA), один стартовый кадр.

Запуск (нужен panda-md[md] и доступ к данным; корни — переменные PANDA_REBORN_DIR и PANDA_ROUGHNESS_DIR):
    uv run python tests/md/generate_md_profiles.py
Результат: tests/md/data/*.npz (ProfileSet) и *_pore.json — коммитятся, тесты читают только их.
"""

import os
import time
from pathlib import Path

from panda_md import SlitPore, density_profile, load_trajectory

REBORN = Path(os.environ.get("PANDA_REBORN_DIR", "/mnt/c/Phystech/Computing_Physics_Center/Projects/PANDA_Reborn"))
ROUGHNESS = Path(os.environ.get("PANDA_ROUGHNESS_DIR", "/mnt/c/Phystech/Computing_Physics_Center/Projects/roughness_2"))
OUTPUT = Path(__file__).parent / "data"
BIN_COUNT = 200


def save_pore(name: str, pore: SlitPore) -> None:
    (OUTPUT / f"{name}_pore.json").write_text(pore.model_dump_json(indent=2) + "\n")


def graphite() -> None:
    system = REBORN / "graphite_decane_tip4p_1.0"
    trajectory = load_trajectory(system / "graphite_decane_tip4p.tpr", system / "graphite_decane_tip4p.xtc")
    pore = SlitPore.from_trajectory(trajectory, walls="resname GRAPHITE")
    save_pore("graphite", pore)
    water = density_profile(trajectory, pore, "resname WATER and name OW", bins=BIN_COUNT, start="2 ns", block="1 ns")
    water.save(OUTPUT / "graphite_water_profiles.npz")


def benzene() -> None:
    system = ROUGHNESS / "angles" / "base"
    trajectory = load_trajectory(system / "nvt_prod.gro", system / "nvt_prod.xtc")
    pore = SlitPore.from_trajectory(trajectory, walls="resname SOLD")
    save_pore("benzene", pore)
    benzene_profiles = density_profile(trajectory, pore, "resname BENZ", bins=BIN_COUNT, start="4 ns", block="1.2 ns")
    benzene_profiles.save(OUTPUT / "benzene_profiles.npz")


def calcite() -> None:
    system = ROUGHNESS / "angles" / "Dev_PANDA" / "Dev_PANDA" / "systems" / "calcite_decane_tip4p_sc_2.5_offset"
    trajectory = load_trajectory(system / "cal_dec_tip4p.gro", system / "cal_dec_tip4p.xtc")
    pore = SlitPore.from_trajectory(trajectory, walls="resname CAL")
    save_pore("calcite", pore)
    density_profile(trajectory, pore, "resname DECAN", bins=BIN_COUNT).save(OUTPUT / "calcite_decane_profiles.npz")
    density_profile(trajectory, pore, "resname WATER and name OW", bins=BIN_COUNT).save(
        OUTPUT / "calcite_water_profiles.npz"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for system in (graphite, benzene, calcite):
        started = time.perf_counter()
        system()
        print(f"{system.__name__}: {time.perf_counter() - started:.0f} с")


if __name__ == "__main__":
    main()
```

В `.gitignore` после строки `*.npz` добавить строку `!tests/*/data/*.npz`.

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: `11 failed` — `FileNotFoundError` на данных `tests/md/data/` (10 тестов) и на `tests/regression/data/density_reference.npz`.

- [ ] **Шаг 3: Эталоны**

Run: `.venv-legacy/bin/python tests/regression/generate_density_reference.py`
Expected: `записано в .../density_reference.npz` (156 КБ).

Run: `uv run python tests/md/generate_md_profiles.py` (корни данных — переменные `PANDA_REBORN_DIR` и `PANDA_ROUGHNESS_DIR`, по умолчанию `/mnt/c/Phystech/Computing_Physics_Center/Projects/{PANDA_Reborn,roughness_2}`)
Expected: `graphite: ~45 с`, `benzene: ~20 с`, `calcite: ~1 с`; в `*_pore.json` ширины 8.269, 10.314 и 9.139 нм; предупреждения «последних кадров не составляют полного блока» ожидаемы.

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `400 passed, 28 skipped`; регрессия со старой PANDA совпадает точно; вода — θ ≈ 171°; бензол — флаг границы режима.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок; `git status` показывает оба `.npz` как новые файлы.

```bash
git add .gitignore tests/regression tests/md
git commit -m "Путь данных: совпадение со старой PANDA и реальная МД — графит, бензол/SOLD, кальцит" \
  -m "Профиль по траектории (центрирование по жидкости, бины по ящику) совпадает со старой PANDA точно. Вода в щели графита: θ = 171° по среднему профилю, 171° ± 8° по блокам — как в задаче 1. Бензол смачивает SOLD — фит поднимает флаг границы режима. Стенки всех трёх подложек — как в ручном расчёте." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 7: Документация стадии

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
| траектория | `unit/test_trajectory.py` | данные | `ArrayTrajectory`: формы массивов, порядок времён, группы и индексы; выбор кадров по времени и шагу; `MDATrajectory` на Universe в памяти: Å → нм, время, выборка, непрямоугольный ящик — ошибка; нет файла или MDAnalysis — понятная ошибка; `SlitPore.from_trajectory`: H, L, k по атомам стенок через границу ящика, неверные оси — ошибка | решётки атомов, Universe из MDAnalysis | точные значения | 1e-12 |
| профиль по траектории | `unit/test_density_profile.py` | данные | центрирование по поре (стенки в каждом кадре или заданный центр) и по жидкости; бины выровнены по стенкам; блоки по времени, неполный блок отброшен; среднее, std, sem; отражение профиля жидкости у верхней стенки; N и V для нормировки mean; предупреждения (частицы вне поры, промежуток стенок ≠ H, основание ящика ≠ kL²); ошибки (нет стенок и центра, пора шире ящика, треугольник без центра); сохранение .npz и .csv | равномерный слой, дрейфующая пора через границу ящика | плотность N / (A · ширина) | 1 % (статистика) |
| путь данных | `synthetic/test_data_path.py` | данные | частицы, разыгранные по профилю модели (z по обратной функции распределения, x, y равномерно), стенки через границу ящика с дрейфом → `density_profile` → фит: валик и капля в системе поры, капля в системе жидкости | щель 4 × 8 нм, k = 0.5, 40 кадров, 4 блока | θ восстановлен | 1.5° |
| МД | `md/test_real_systems.py` | физическая модель | профили по реальным траекториям, посчитанные новым путём данных (`generate_md_profiles.py`, нужны данные и `panda-md[md]`): стенки найдены как в ручном расчёте по .gro; вся жидкость в поре; вода в щели графита — несмачивающий валик; бензол на подложке SOLD смачивает её — фит упирается в границу режима и поднимает флаг (формул θ < 90° нет, O1) | графит (вода, 8 блоков × 1 нс), бензол/SOLD (5 блоков × 1.2 нс), кальцит (стартовый кадр) | вода: θ 165°–180° (задача 1: 171.8° ± 7.1°), φ = 0.487; бензол: у стенок плотнее, чем в середине, `at_regime_boundary` при ρ_bulk 10–25 нм⁻³ | H — 1e-3 нм |
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
````

- [ ] **Шаг 2: Спецификация** (точечные замены через Edit)

В таблице решений после строки D20 добавить:

```markdown
| D21 | Траектория: интерфейс `Trajectory` в ядре (кадры — время, ящик, позиции запрошенных групп; нм, пс), реализации `ArrayTrajectory` (массивы) и `MDATrajectory` (MDAnalysis, extra `md`; импорт только внутри `load_trajectory`) (стадия 4) |
| D22 | Пора в ящике: `normal_axis`, `wall_selection`, `locate_center_nm`; щель — середина наибольшего пустого промежутка между атомами стенок через PBC, `SlitPore.from_trajectory`; у многоугольных пор центр пока задаётся явно (`pore_center_nm`) — данных МД для них нет (стадия 4) |
| D23 | Центрирование профиля — `pore` / `liquid`; «без сдвига» = `pore` с заданным центром. Осевое центрирование (`+axis`), `per="molecule"`, `wall_offset`, статистическая неэффективность — фаза 6: на 1D-профиль не влияют (стадия 4) |
```

§3.3: строку, начинающуюся с `` `SlitPore(width_nm, length_nm, base_aspect_ratio=1.0)` (стенки и ось — фаза 5 ``, заменить на:

```markdown
`SlitPore(width_nm, length_nm, base_aspect_ratio=1.0, normal_axis=Axis.Z, wall_selection=None)`; `SlitPore.from_trajectory(traj, walls=<выборка>, length_axis=Axis.X)`.
```

§5, строку таблицы стратегий `` | `none` | без сдвига | задаёт пользователь | `` заменить на:

```markdown
| `pore` + `pore_center_nm` | заданный центр → 0 (данные уже отцентрированы — 0) | PORE |
```

§5, в блоке API строку `traj = panda.load_trajectory("md.xtc", "md.gro")` заменить на `traj = panda.load_trajectory("md.gro", "md.xtc")   # топология, затем траектория`, а `L_axis="x"` — на `length_axis=panda.Axis.X`.

- [ ] **Шаг 3: ROADMAP и README**

`ROADMAP.md`, «Фаза 5. Путь данных»: первой строкой списка добавить

```markdown
- **Статус: выполнено** — план `docs/superpowers/plans/2026-10-16-stage4-data-path.md`; 400 тестов; щель по траектории (стенки через PBC), центрирование по поре и жидкости, блоки, сохранение; совпадение со старой PANDA и реальная МД: вода в щели графита (θ ≈ 171°, как в задаче 1), бензол на SOLD (смачивает — флаг границы режима, O1), кальцит (только стартовый кадр — поиск стенок). Поиск стенок треугольника и квадрата — нет данных МД (центр задаётся явно); осевое центрирование, `per="molecule"`, `wall_offset` — фаза 6 (D23).
```

`README.md`, блок «Status: pre-alpha»: фрагмент `Trajectory processing and CLI are\n> not implemented yet` заменить на `Density profiles from MD trajectories (MDAnalysis, extra \`md\`) work for the slit pore. The CLI is\n> not implemented yet`. После первого примера кода добавить:

````markdown
From an MD trajectory (`pip install 'panda-md[md]'`):

```python
import panda_md as panda
traj = panda.load_trajectory("system.tpr", "system.xtc")
pore = panda.SlitPore.from_trajectory(traj, walls="resname GRAPHITE")
profiles = panda.density_profile(traj, pore, "resname WATER and name OW", start="2 ns", block="1 ns")
result = pore.interface("roll").fit(profiles.mean(), density_bulk_per_nm3=33.1)
```
````

- [ ] **Шаг 4: Полная проверка и коммит**

Run: `make check`, `make test-env`, `make wheel-check`; на Python 3.10: `UV_PROJECT_ENVIRONMENT=.venv-py310 uv sync --python 3.10 --all-extras` и `UV_PROJECT_ENVIRONMENT=.venv-py310 uv run --no-sync pytest -q`. Всё зелёное, `400 passed, 28 skipped`.

```bash
git add tests/README.md docs ROADMAP.md README.md
git commit -m "Стадия 4 закрыта: описание тестов, решения D21–D23 в спецификации, ROADMAP, README" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
