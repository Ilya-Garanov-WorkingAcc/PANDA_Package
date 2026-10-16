# Стадия 6 — трёхмерные формы (фаза 6б): план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Трёхмерная форма жидкости (`contains`) у форм щели, оси симметрии в классах форм, осевое центрирование и 2D-карты плотности, синтетические траектории из облаков точек — сквозная проверка «форма → траектория → профиль → фит».

**Architecture:** Профиль щели — доля площади сечения ячейки, занятая жидкостью, поэтому `contains` выводится из профиля по виду сечения (`CrossSection`: круг, отверстие, полоса, всё) — без новых формул. Ось симметрии — `ClassVar` формы (`SymmetryAxis(orientation, locate)`); её положение в кадре — `core.centering.axis_position` (круговое среднее; у перфорации — центр пустоты). `density_map` строит карту «длинная сторона × ось профиля» с осевым центрированием; `synthetic_trajectory` заполняет форму частицами с ρ_bulk.

**Tech Stack:** Python ≥ 3.10, numpy, scipy, pydantic v2, pytest, uv, black, ruff, mypy (strict + плагин pydantic).

**Spec:** `docs/superpowers/specs/2026-10-01-panda-architecture-design.md` (§5 «Ось», §9 contract «облако из contains даёт тот же профиль», synthetic «облако точек → полный путь данных → фит |Δθ| < 2°»); `docs/architecture/03_centering_data.md` §3 (оси форм — по замечанию пользователя: валик — горизонтальная, пончик и перфорация — вертикальные, у остальных нет).

**Решения** (в спецификацию — задача 4, D26): `contains` у треугольника и квадрата не выводится — модели PANDA_Polygon ещё меняются (O9); 2D-карта — только для щели (плоскость «длинная сторона × ось профиля», поэтому у `SlitPore` появляется поле `length_axis`).

**Проверено до написания плана.** Код собран и прогнан по задачам в чистом клоне (коммит `5bf2a3d`): тесты задач падают до реализации, после — `black --check`, `ruff check`, `mypy`, `pytest` (467 тестов, 28 пропусков). На реальной МД (вода в щели графита, каждый 50-й кадр после 2 нс): карта с осевым центрированием по валику симметрична (корреляция с отражением 0.995), в сердцевине 33–35 нм⁻³. Код переносить дословно.

## Global Constraints

- Ограничения стадий 1–5 в силе: Python ≥ 3.10; pydantic, без датаклассов (D14); `Enum`, типы, говорящие имена (D17); black 120, ruff, mypy strict; сообщения по-русски; слои (§2): `core` не импортирует `pores`, `trajectory`, `synthetic`.
- Разработка локальная, в ветке `stage1-core-prototype`; в GitHub не отправлять.
- Многострочный текст — через Write/Edit; коммиты — отдельными `-m`, последний — `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Проверка перед коммитом: `make check`.

## Review Focus

1. Профиль ↔ форма: облако точек по `contains` должно давать тот же профиль при k ≠ 1 (осесимметричные формы видят короткую сторону). Тест — `test_point_cloud_reproduces_the_profile[*-0.4]` (задача 1).
2. Ось формы у границы ящика: положение — круговое среднее, у перфорации — плюс половина периода. Тест — `test_axis_position_through_box_boundary` (задача 1).
3. Шум выборки малой капли: критерий |Δθ| < 2° — на 100 кадрах (за 20 кадров ошибка до 3.4°). Тест — `test_angle_recovered_from_point_clouds[droplet]` (задача 2).
4. Осевое центрирование формы без оси: ожидание — понятная ошибка. Тест — `test_map_needs_a_shape_with_an_axis_and_survives_saving` (задача 3).
5. Рефакторинг пути данных (помощники стали публичными): профиль не должен измениться — md-тесты на реальных кадрах и регрессия со старой PANDA остаются зелёными (задача 3).

---

### Задача 1: Трёхмерные формы щели и оси симметрии

**Files:**
- Modify: `src/panda_md/core/enums.py` (`AxisOrientation`, `AxisLocator`), `src/panda_md/core/shape.py` (`contains`, `has_contains`, `symmetry_axis`), `src/panda_md/pores/slit/shape.py` (`CrossSection`, `contains`), формы щели `droplet.py`, `doughnut.py`, `perforation.py`, `worm.py`, `roll.py`, `layer.py` (вид сечения, оси)
- Create: `src/panda_md/core/centering.py`, `tests/unit/test_shapes_3d.py`

**Interfaces:**
- Produces: `SymmetryAxis(orientation: AxisOrientation, locate: AxisLocator)`; `axis_position(lateral_positions, lateral_periods, axis) -> FloatArray` (по длинной и короткой стороне, в [0, L)); `InterfaceShape.contains(points, geometry, values) -> BoolArray`, `InterfaceShape.has_contains()`, `InterfaceShape.symmetry_axis: ClassVar[SymmetryAxis | None] = None`; `CrossSection.DISK/HOLE/BAND/FULL`, `SlitShape.cross_section`.

- [ ] **Шаг 1: Тесты**

`tests/unit/test_shapes_3d.py`:

```python
"""Трёхмерные формы щели (contains) и оси симметрии."""

import itertools
import math

import numpy as np
import pytest

from panda_md import SlitPore, SquarePore
from panda_md.core.centering import SymmetryAxis, axis_position
from panda_md.core.enums import AxisLocator, AxisOrientation
from panda_md.core.errors import NotDerivedError
from panda_md.core.types import FloatArray
from panda_md.pores.slit import SlitGeometry
from panda_md.pores.slit.droplet import SlitDroplet
from panda_md.pores.slit.worm import SlitWorm

SEED = 17
POINTS = 400_000


def cell_points(geometry: SlitGeometry, generator: np.random.Generator) -> FloatArray:
    """Точки, равномерно заполняющие ячейку щели (безразмерные, ось формы в начале координат)."""
    uniform = generator.uniform(-0.5, 0.5, size=(POINTS, 3))
    points: FloatArray = uniform * np.array([geometry.pore_length_ratio, geometry.short_side_ratio(), 1.0])
    return points


def test_droplet_is_a_spherical_cap_on_the_lower_wall() -> None:
    pore = SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.5)
    values = pore.interface("droplet").internal_values({"theta": 130.0, "delta": 0.03, "phi": 0.02})
    shape, geometry = SlitDroplet(), pore.geometry()
    square = geometry.equivalent_square()
    square_values = shape.square_values(geometry, values)
    radius, center = shape.sphere_radius(square, square_values), shape.sphere_center(square, square_values)
    points = cell_points(geometry, np.random.default_rng(SEED))
    sphere = (points[:, 0] ** 2 + points[:, 1] ** 2 + (points[:, 2] - center) ** 2 < radius**2) & (
        points[:, 2] > -0.5 + 0.03
    )
    agreement = np.mean(shape.contains(points, geometry, values) == sphere)
    assert agreement > 0.9995  # расхождение — только точки на самой границе (дискретизация не участвует)


def test_worm_is_a_cylindrical_segment_along_the_short_side() -> None:
    pore = SlitPore(width_nm=8.0, length_nm=32.0, base_aspect_ratio=0.5)
    values = pore.interface("worm").internal_values({"theta": 130.0, "delta": 0.03, "phi": 0.1})
    shape, geometry = SlitWorm(), pore.geometry()
    radius, center = shape.cylinder_radius(geometry, values), shape.cylinder_center(geometry, values)
    points = cell_points(geometry, np.random.default_rng(SEED))
    cylinder = (points[:, 0] ** 2 + (points[:, 2] - center) ** 2 < radius**2) & (points[:, 2] > -0.5 + 0.03)
    assert np.mean(shape.contains(points, geometry, values) == cylinder) > 0.9995


CASES = [
    ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.02}),
    ("doughnut", {"theta": 130.0, "delta": 0.03, "phi": 0.15}),
    ("worm", {"theta": 130.0, "delta": 0.03, "phi": 0.1}),
    ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
    ("perforation", {"theta": 130.0, "delta": 0.03, "phi": 0.8}),
    ("layer", {"delta": 0.03, "phi": 0.4}),
]


@pytest.mark.parametrize("aspect_ratio", [1.0, 0.4])
@pytest.mark.parametrize(("shape_name", "values"), CASES, ids=[name for name, _ in CASES])
def test_point_cloud_reproduces_the_profile(shape_name: str, values: dict[str, float], aspect_ratio: float) -> None:
    """Доля точек в жидкости по слоям = профиль (контракт §9: облако даёт тот же профиль; допуск — статистика)."""
    pore = SlitPore(width_nm=8.0, length_nm=16.0, base_aspect_ratio=aspect_ratio)
    model = pore.interface(shape_name)
    assert model.is_feasible(**values), values
    internal = model.internal_values(values)
    points = cell_points(pore.geometry(), np.random.default_rng(SEED))
    inside = model.shape.contains(points, pore.geometry(), internal)
    phi = internal["phi"]
    assert np.mean(inside) == pytest.approx(phi, abs=4.0 * math.sqrt(phi * (1.0 - phi) / POINTS))  # 4σ выборки
    edges = np.linspace(-0.5, 0.5, 21)
    counts, _ = np.histogram(points[inside, 2], bins=edges)
    totals, _ = np.histogram(points[:, 2], bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fine = np.linspace(-0.5, 0.5, 20001)
    expected = [
        np.mean(model.shape.profile(fine[(fine >= low) & (fine < high)], pore.geometry(), internal))
        for low, high in itertools.pairwise(edges)
    ]
    np.testing.assert_allclose(counts / totals, expected, atol=0.02, err_msg=str(centers))


def test_shapes_declare_their_symmetry_axes() -> None:
    pore = SlitPore(width_nm=8.0, length_nm=16.0)
    axes = {name: pore.interface(name).shape.symmetry_axis for name in pore.shape_names()}
    assert axes["roll"] == SymmetryAxis(orientation=AxisOrientation.HORIZONTAL, locate=AxisLocator.LIQUID_CENTER)
    assert axes["doughnut"] == SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.LIQUID_CENTER)
    assert axes["perforation"] == SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.VOID_CENTER)
    assert axes["droplet"] is None and axes["worm"] is None and axes["layer"] is None


def test_axis_position_through_box_boundary() -> None:
    periods = np.array([10.0, 4.0])
    around_edge = np.array([[9.5, 1.0], [0.5, 3.0], [9.8, 1.0], [0.2, 3.0]])
    vertical = SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.LIQUID_CENTER)
    position = axis_position(around_edge, periods, vertical)
    assert min(position[0], 10.0 - position[0]) == pytest.approx(0.0, abs=1e-9)
    assert position[1] == pytest.approx(2.0)
    horizontal = SymmetryAxis(orientation=AxisOrientation.HORIZONTAL, locate=AxisLocator.LIQUID_CENTER)
    assert axis_position(around_edge, periods, horizontal)[1] == 0.0
    void = SymmetryAxis(orientation=AxisOrientation.VERTICAL, locate=AxisLocator.VOID_CENTER)
    np.testing.assert_allclose(axis_position(around_edge, periods, void), [5.0, 0.0], atol=1e-9)


def test_polygon_shapes_have_no_3d_form_yet() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    droplet = square.interface("droplet_w1c0")
    assert not droplet.shape.has_contains()
    with pytest.raises(NotDerivedError, match="contains"):
        droplet.shape.contains(
            np.zeros((1, 3)), square.geometry(), droplet.internal_values({"theta": 140.0, "delta": 0.0, "phi": 0.05})
        )
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.core.centering'`.

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


class AxisOrientation(str, Enum):
    """Направление оси симметрии формы относительно стенок."""

    VERTICAL = "vertical"  # перпендикулярно стенкам (пончик, перфорация): центр ищется по обеим осям плоскости
    HORIZONTAL = "horizontal"  # вдоль короткой стороны ячейки (валик): центр ищется только поперёк оси


class AxisLocator(str, Enum):
    """Как найти положение оси в кадре."""

    LIQUID_CENTER = "liquid_center"  # центр жидкости (круговое среднее с учётом PBC)
    VOID_CENTER = "void_center"  # центр пустоты: жидкость занимает всё, кроме отверстия, — центр жидкости + L/2
```

`src/panda_md/core/centering.py`:

```python
"""Осевое центрирование: ось симметрии формы в плоскости стенок и как найти её положение в кадре.

Центрирование вдоль оси профиля (по поре или по жидкости) — `trajectory/density.py`. Здесь — вторая, независимая
часть: сдвиг в плоскости стенок, чтобы ось формы стояла в начале координат. На одномерный профиль щели он не влияет,
нужен для 2D-карт и облаков точек. Ось объявляется в классе формы (`InterfaceShape.symmetry_axis`).
"""

import numpy as np

from panda_md.core.enums import AxisLocator, AxisOrientation
from panda_md.core.pbc import circular_mean
from panda_md.core.types import FloatArray, PandaModel


class SymmetryAxis(PandaModel):
    orientation: AxisOrientation
    locate: AxisLocator


def axis_position(lateral_positions: FloatArray, lateral_periods: FloatArray, axis: SymmetryAxis) -> FloatArray:
    """Положение оси в плоскости стенок: (по длинной стороне, по короткой стороне), в [0, L) по каждой.

    lateral_positions — координаты частиц (n × 2): вдоль длинной стороны ячейки и вдоль короткой. У горизонтальной
    оси (вдоль короткой стороны) положение по короткой стороне не определено и возвращается как 0.
    """
    along_length = circular_mean(lateral_positions[:, 0], float(lateral_periods[0]))
    along_width = 0.0
    if axis.orientation is AxisOrientation.VERTICAL:
        along_width = circular_mean(lateral_positions[:, 1], float(lateral_periods[1]))
    position = np.array([along_length, along_width])
    if axis.locate is AxisLocator.VOID_CENTER:
        shift = 0.5 * lateral_periods
        if axis.orientation is AxisOrientation.HORIZONTAL:
            shift = np.array([shift[0], 0.0])
        position = np.mod(position + shift, lateral_periods)
    centered: FloatArray = position
    return centered
```

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

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import Frame, ParameterName, Regime
from panda_md.core.errors import NotDerivedError, UnsupportedRegimeError
from panda_md.core.parameters import Parameter, ParameterValues
from panda_md.core.types import BoolArray, FloatArray, PandaModel

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
    symmetry_axis: ClassVar[SymmetryAxis | None] = None  # ось для осевого центрирования; None — нет
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

    def contains(self, points: FloatArray, geometry: GeometryT, values: ParameterValues) -> BoolArray:
        """Какие точки (n × 3, безразмерные: вдоль длинной стороны, вдоль короткой, вдоль оси профиля) — в жидкости.

        Ось симметрии формы (если есть) — в начале координат плоскости стенок; координаты в плоскости стенок
        периодичны с периодом ячейки. Вызывается только для допустимых параметров.
        """
        raise NotDerivedError(f"{self.name}: трёхмерная форма (contains) не выведена")

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

    def has_contains(self) -> bool:
        return type(self).contains is not InterfaceShape.contains

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

Заменить целиком формы щели:

`src/panda_md/pores/slit/shape.py`:

```python
"""Общая база форм щели: система отсчёта поры, доступная ширина 1 − 2δ, существование по интервалу φ,
трёхмерная форма по профилю.

Профиль щели — доля площади сечения ячейки, занятая жидкостью на высоте z. Поэтому сечение восстанавливается
по профилю: круг (капля, пончик) площади ρ · S, круглое отверстие (перфорация) площади (1 − ρ) · S, полоса
вдоль короткой стороны (валик, червь) ширины ρ · l, всё сечение (слой); S = k l² — основание ячейки.
"""

import math
from enum import Enum
from typing import ClassVar

import numpy as np

from panda_md.core.enums import Frame, ParameterName
from panda_md.core.parameters import ParameterValues
from panda_md.core.shape import InterfaceShape
from panda_md.core.types import BoolArray, FloatArray
from panda_md.pores.slit.pore import SlitGeometry

FULL_DENSITY_THRESHOLD = 0.5  # профиль слоя — ступенька 0/1


class CrossSection(str, Enum):
    """Сечение формы щели плоскостью z = const."""

    DISK = "disk"  # круг с центром на вертикальной оси
    HOLE = "hole"  # всё доступное сечение, кроме круга на вертикальной оси
    BAND = "band"  # полоса вдоль короткой стороны, симметричная относительно x = 0
    FULL = "full"  # всё сечение


class SlitShape(InterfaceShape[SlitGeometry]):
    """Форма щели. Существует, если доступная ширина положительна и φ строго внутри (φ_min, φ_max)."""

    frame: ClassVar[Frame] = Frame.PORE
    cross_section: ClassVar[CrossSection]

    @staticmethod
    def accessible_width(values: ParameterValues) -> float:
        """1 − 2δ — ширина щели, доступная анализируемой жидкости."""
        return 1.0 - 2.0 * values[ParameterName.WALL_LAYER_THICKNESS]

    @staticmethod
    def inside_accessible(coordinate: FloatArray, values: ParameterValues) -> BoolArray:
        """|z| < 1/2 − δ — точки, доступные анализируемой жидкости."""
        half = 0.5 * SlitShape.accessible_width(values)
        return np.abs(coordinate) < half

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        if self.accessible_width(values) <= 0.0:
            return False
        lower, upper = self.phi_limits(geometry, values)
        return bool(lower < values[ParameterName.VOLUME_FRACTION] < upper)

    def contains(self, points: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> BoolArray:
        """Точка в жидкости, если она внутри сечения на своей высоте (сечение — по профилю, см. модуль)."""
        length = geometry.pore_length_ratio
        width = geometry.short_side_ratio()
        along_length = points[:, 0] - length * np.round(points[:, 0] / length)
        along_width = points[:, 1] - width * np.round(points[:, 1] / width)
        density = self.profile(points[:, 2], geometry, values)
        base = geometry.base_area()
        if self.cross_section is CrossSection.BAND:
            inside: BoolArray = np.abs(along_length) < 0.5 * density * length
            return inside
        radius_squared = along_length**2 + along_width**2
        if self.cross_section is CrossSection.DISK:
            disk: BoolArray = radius_squared < density * base / math.pi
            return disk
        if self.cross_section is CrossSection.HOLE:
            hole_radius_squared = (1.0 - density) * base / math.pi
            liquid: BoolArray = self.inside_accessible(points[:, 2], values) & (radius_squared >= hole_radius_squared)
            return liquid
        full: BoolArray = density >= FULL_DENSITY_THRESHOLD
        return full
```

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
from panda_md.pores.slit.shape import CrossSection


@SlitPore.register
class SlitDroplet(AxisymmetricSlitShape):
    name: ClassVar[str] = "droplet"
    cross_section: ClassVar[CrossSection] = CrossSection.DISK
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

`src/panda_md/pores/slit/doughnut.py`:

```python
"""Пончик (doughnut) в щели: перемычка между стенками, осесимметричная вокруг вертикальной оси.

Формулы: Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (6)–(9), с поправкой на слой δ
по старой PANDA (`interface/alpha.py`). Ячейка L × kL — через эквивалентную квадратную ячейку (axisymmetric.py).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import AxisLocator, AxisOrientation, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.types import FloatArray
from panda_md.pores.slit.axisymmetric import AxisymmetricSlitShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection


@SlitPore.register
class SlitDoughnut(AxisymmetricSlitShape):
    name: ClassVar[str] = "doughnut"
    cross_section: ClassVar[CrossSection] = CrossSection.DISK
    symmetry_axis: ClassVar[SymmetryAxis | None] = SymmetryAxis(
        orientation=AxisOrientation.VERTICAL, locate=AxisLocator.LIQUID_CENTER
    )
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

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import AxisLocator, AxisOrientation, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.types import FloatArray
from panda_md.pores.slit.axisymmetric import AxisymmetricSlitShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection


@SlitPore.register
class SlitPerforation(AxisymmetricSlitShape):
    name: ClassVar[str] = "perforation"
    cross_section: ClassVar[CrossSection] = CrossSection.HOLE
    symmetry_axis: ClassVar[SymmetryAxis | None] = SymmetryAxis(
        orientation=AxisOrientation.VERTICAL, locate=AxisLocator.VOID_CENTER
    )
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

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.common import circular_segment_centroid_offset, circular_segment_factor
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection, SlitShape


@SlitPore.register
class SlitWorm(SlitShape):
    name: ClassVar[str] = "worm"
    cross_section: ClassVar[CrossSection] = CrossSection.BAND
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

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
        accessible_width = self.accessible_width(values)
        length_ratio = geometry.pore_length_ratio
        double_segment = 2.0 * contact_angle - math.sin(2.0 * contact_angle)
        by_height = accessible_width**2 * double_segment / (2.0 * length_ratio * (1.0 - math.cos(contact_angle)) ** 2)
        by_length = length_ratio * double_segment / 8.0
        return 0.0, min(by_height, by_length)

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

`src/panda_md/pores/slit/roll.py`:

```python
"""Валик (roll) в щели: перемычка между стенками, однородная вдоль одной оси.

Формулы: Semenchuk et al., Colloids Surf. A 725 (2025) 137586, уравнения (1)–(4);
при δ = 0 совпадают с Semenchuk et al., Colloids Surf. A 708 (2025) 135994, уравнения (15)–(18).
"""

import math
from typing import ClassVar

import numpy as np

from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import AxisLocator, AxisOrientation, ParameterName, Regime
from panda_md.core.parameters import CONTACT_ANGLE, VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SLIT_HALF_WIDTH, SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection, SlitShape


@SlitPore.register
class SlitRoll(SlitShape):
    name: ClassVar[str] = "roll"
    cross_section: ClassVar[CrossSection] = CrossSection.BAND
    symmetry_axis: ClassVar[SymmetryAxis | None] = SymmetryAxis(
        orientation=AxisOrientation.HORIZONTAL, locate=AxisLocator.LIQUID_CENTER
    )
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE, WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}

    def contact_spot_width(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        """d_roll — ширина пятна контакта, уравнение (1)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
        length_ratio = geometry.pore_length_ratio
        return (
            values[ParameterName.VOLUME_FRACTION] * length_ratio / accessible_width
            + accessible_width * (math.pi - 2.0 * contact_angle) / (4.0 * math.cos(contact_angle) ** 2)
            - 0.5 * accessible_width * math.tan(contact_angle)
        )

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        """φ_min и φ_max, уравнения (3) и (4)."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
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

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # валик симметричен относительно середины щели

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        """Две дуги мениска R(2θ − π) · 2 = (1−2δ)(π − 2θ)/cos θ и два пятна контакта 2d — на длину валика k l.
        При θ = 180°, δ = 0 сумма — kl (2d + π), уравнение (21) Shistko et al. (2025).
        """
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        accessible_width = self.accessible_width(values)
        length = geometry.short_side_ratio()
        return SurfaceTerms(
            liquid_liquid_area=length * accessible_width * (math.pi - 2.0 * contact_angle) / math.cos(contact_angle),
            liquid_solid_area=2.0 * self.contact_spot_width(geometry, values) * length,
        )

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk, уравнение (2): (d + √((1−2δ)²/cos²θ − 4z²) + (1−2δ) tan θ) / l при |z| < 1/2 − δ."""
        contact_angle = values[ParameterName.CONTACT_ANGLE]
        wall_layer = values[ParameterName.WALL_LAYER_THICKNESS]
        accessible_width = self.accessible_width(values)
        inside = np.abs(coordinate) < SLIT_HALF_WIDTH - wall_layer
        meniscus = np.sqrt(np.maximum(accessible_width**2 / math.cos(contact_angle) ** 2 - 4.0 * coordinate**2, 0.0))
        density = (
            self.contact_spot_width(geometry, values) + meniscus + accessible_width * math.tan(contact_angle)
        ) / geometry.pore_length_ratio
        return np.where(inside, density, 0.0)
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

from panda_md.core.enums import ParameterName, Regime
from panda_md.core.parameters import VOLUME_FRACTION, WALL_LAYER_THICKNESS, Parameter, ParameterValues
from panda_md.core.shape import SurfaceTerms
from panda_md.core.types import FloatArray
from panda_md.pores.slit.pore import SlitGeometry, SlitPore
from panda_md.pores.slit.shape import CrossSection, SlitShape


@SlitPore.register
class SlitLayer(SlitShape):
    name: ClassVar[str] = "layer"
    cross_section: ClassVar[CrossSection] = CrossSection.FULL
    parameters: ClassVar[tuple[Parameter, ...]] = (WALL_LAYER_THICKNESS, VOLUME_FRACTION)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {}  # контактного угла нет
    default_optimize: ClassVar[tuple[str, ...]] = (ParameterName.VOLUME_FRACTION.value,)

    def phi_limits(self, geometry: SlitGeometry, values: ParameterValues) -> tuple[float, float]:
        return 0.0, self.accessible_width(values)

    def center_of_mass(self, geometry: SlitGeometry, values: ParameterValues) -> float:
        return 0.0  # слой посередине щели

    def surface_terms(self, geometry: SlitGeometry, values: ParameterValues) -> SurfaceTerms:
        return SurfaceTerms(liquid_liquid_area=2.0 * geometry.base_area(), liquid_solid_area=0.0)

    def profile(self, coordinate: FloatArray, geometry: SlitGeometry, values: ParameterValues) -> FloatArray:
        """ρ/ρ_bulk = 1 при |z| < φ/2, иначе 0."""
        half_thickness = 0.5 * values[ParameterName.VOLUME_FRACTION]
        return np.where(np.abs(coordinate) < half_thickness, 1.0, 0.0)
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `459 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests/unit/test_shapes_3d.py
git commit -m "Трёхмерные формы щели (contains по профилю) и оси симметрии в классах форм" \
  -m "Сечение восстанавливается по профилю: круг, отверстие, полоса, всё сечение; капля и червь совпадают с прямой геометрией шара и цилиндра. Оси: валик — горизонтальная, пончик и перфорация — вертикальные." \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 2: Облака точек — синтетическая траектория

**Files:**
- Create: `src/panda_md/synthetic/points.py`, `tests/synthetic/test_point_clouds.py` (без 2D-карт — задача 3)

**Interfaces:**
- Consumes: `contains` (задача 1), `ArrayTrajectory`, `SlitPore`.
- Produces: `synthetic_trajectory(model, values, *, density_bulk_per_nm3, frame_count=10, time_step_ps=10.0, substrate_nm=1.0, seed=None) -> ArrayTrajectory` (группы `liquid`, `walls`).

- [ ] **Шаг 1: Тесты**

`tests/synthetic/test_point_clouds.py`:

```python
"""Облака точек: форма в щели → траектория с блуждающей формой → стенки, центрирование, профиль, фит.

Критерий спецификации (§9): «облако точек → полный путь данных → фит» — |Δθ| < 2°.
"""

import pytest

from panda_md import SlitPore, SquarePore, density_profile
from panda_md.core.errors import NotDerivedError
from panda_md.synthetic.points import synthetic_trajectory

DENSITY_BULK = 33.0
PORE = SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5)
MAX_ANGLE_ERROR_DEG = 2.0


@pytest.mark.parametrize(
    ("shape_name", "truth"),
    [
        ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
        ("doughnut", {"theta": 130.0, "delta": 0.03, "phi": 0.2}),
    ],
)
def test_angle_recovered_from_point_clouds(shape_name: str, truth: dict[str, float]) -> None:
    model = PORE.interface(shape_name)
    trajectory = synthetic_trajectory(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame_count=100, seed=1)
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    assert found.width_nm == pytest.approx(PORE.width_nm, abs=1e-9)
    assert found.base_aspect_ratio == pytest.approx(PORE.base_aspect_ratio)
    profiles = density_profile(trajectory, found, "liquid", bins=100)
    result = found.interface(shape_name).fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)
    assert result.values["phi"] == pytest.approx(truth["phi"], rel=0.03)


def test_point_clouds_exist_only_for_slit_shapes() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    with pytest.raises(NotDerivedError, match="щели"):
        synthetic_trajectory(
            square.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.0, "phi": 0.05}, density_bulk_per_nm3=30.0
        )
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.synthetic.points'`.

- [ ] **Шаг 3: Реализация**

`src/panda_md/synthetic/points.py`:

```python
"""Синтетическая траектория из облаков точек: форма в щели → частицы и атомы стенок по кадрам.

Частицы жидкости равномерно заполняют трёхмерную форму (`contains`) с плотностью ρ_bulk; ось формы в каждом кадре
сдвинута случайно в плоскости стенок (как блуждает капля или валик в МД), щель пересекает границу ящика по оси
профиля. Это вход для сквозной проверки «форма → траектория → центрирование → профиль → фит» и 2D-карт.
"""

from collections.abc import Mapping

import numpy as np

from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.pores.slit.pore import SlitPore
from panda_md.trajectory.arrays import ArrayTrajectory

WALL_SPACING_NM = 0.25  # шаг решётки атомов подложки
OVERSAMPLING = 1.3  # кандидатов на одну принятую частицу больше 1/φ — меньше повторных попыток
MAX_ATTEMPTS = 100


def synthetic_trajectory(
    model: InterfaceModel,
    values: Mapping[str, float],
    *,
    density_bulk_per_nm3: float,
    frame_count: int = 10,
    time_step_ps: float = 10.0,
    substrate_nm: float = 1.0,
    seed: int | None = None,
) -> ArrayTrajectory:
    """Траектория щели с формой `model` (единицы пользователя в values): группы «liquid» и «walls».

    Ящик — L × kL × (H + substrate_nm); подложка занимает z ∈ [0, substrate_nm), её образ сверху замыкает щель.
    Число частиц жидкости N = φ ρ_bulk V постоянно; в каждом кадре ось формы — в случайной точке плоскости стенок.
    """
    pore = model.pore
    if not isinstance(pore, SlitPore) or not model.shape.has_contains():
        raise NotDerivedError(f"{model}: облако точек выведено только для форм щели (нужен contains)")
    internal = model.supported_values(values)
    geometry = pore.geometry()
    if not model.shape.feasible(geometry, internal):
        raise NotDerivedError(f"{model}: форма не существует при {dict(values)}")
    generator = np.random.default_rng(seed)
    width = pore.width_nm
    box = np.array([pore.length_nm, pore.base_aspect_ratio * pore.length_nm, width + substrate_nm])
    walls = _substrate(box, substrate_nm)
    volume_fraction = internal["phi"]
    count = round(volume_fraction * density_bulk_per_nm3 * pore.volume_nm3())
    cell = np.array([geometry.pore_length_ratio, geometry.short_side_ratio(), 1.0])
    frames = []
    for _ in range(frame_count):
        points = _sample_inside(model, internal, cell, count, volume_fraction, generator)
        axis = generator.uniform(0.0, 1.0, size=2) * box[:2]
        liquid = points * width + np.array([axis[0], axis[1], substrate_nm + width / 2.0])
        frames.append(np.mod(np.vstack([walls, liquid]), box))
    return ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.tile(box, (frame_count, 1)),
        times_ps=time_step_ps * np.arange(frame_count),
        groups={"walls": np.arange(len(walls)), "liquid": len(walls) + np.arange(count)},
    )


def _substrate(box: np.ndarray, substrate_nm: float) -> np.ndarray:
    """Решётка атомов подложки: слои через WALL_SPACING_NM от z = 0 до z = substrate_nm (обе поверхности)."""
    xs = np.arange(0.0, box[0], WALL_SPACING_NM)
    ys = np.arange(0.0, box[1], WALL_SPACING_NM)
    zs = np.linspace(0.0, substrate_nm, max(2, round(substrate_nm / WALL_SPACING_NM) + 1))
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    walls: np.ndarray = grid
    return walls


def _sample_inside(
    model: InterfaceModel,
    internal: Mapping[str, float],
    cell: np.ndarray,
    count: int,
    volume_fraction: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """count точек, равномерно распределённых в жидкости (безразмерные, ось формы в начале координат)."""
    geometry = model.pore.geometry()
    accepted: list[np.ndarray] = []
    total = 0
    for _ in range(MAX_ATTEMPTS):
        batch = max(1000, int(OVERSAMPLING * (count - total) / max(volume_fraction, 1e-6)))
        candidates = generator.uniform(-0.5, 0.5, size=(batch, 3)) * cell
        inside = candidates[model.shape.contains(candidates, geometry, dict(internal))]
        accepted.append(inside)
        total += len(inside)
        if total >= count:
            points: np.ndarray = np.vstack(accepted)[:count]
            return points
    raise NotDerivedError(f"{model}: не удалось набрать {count} точек в форме за {MAX_ATTEMPTS} попыток")
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `463 passed, 28 skipped`.

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md/synthetic/points.py tests/synthetic/test_point_clouds.py
git commit -m "Облака точек: синтетическая траектория формы в щели — сквозной путь до фита, |Δθ| < 2°" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Задача 3: Осевое центрирование и 2D-карты плотности

**Files:**
- Modify: `src/panda_md/core/profile.py` (`save_npz_arrays`, `load_npz_arrays`), `src/panda_md/trajectory/density.py` (публичные `particle_positions`, `profile_frame_center`, `profile_bin_edges`), `src/panda_md/pores/slit/pore.py` (`length_axis`)
- Create: `src/panda_md/trajectory/maps.py`
- Modify: `tests/synthetic/test_point_clouds.py`

**Interfaces:**
- Consumes: `axis_position`, `symmetry_axis` (задача 1), путь данных стадий 4–5.
- Produces: `DensityMap(length_edges_nm, height_edges_nm, density_per_nm3, frame, axis_centered, frame_count, selection)` с `save`/`load`; `density_map(trajectory, pore, select, *, shape=None, centering="pore", bins=(100, 100), start, stop, stride, pore_center_nm=None, per="atom") -> DensityMap`; `SlitPore.length_axis: Axis = Axis.X`.

- [ ] **Шаг 1: Тесты**

Заменить `tests/synthetic/test_point_clouds.py` целиком:

`tests/synthetic/test_point_clouds.py`:

```python
"""Облака точек: форма в щели → траектория с блуждающей формой → стенки, центрирование, профиль, фит и 2D-карты.

Критерий спецификации (§9): «облако точек → полный путь данных → фит» — |Δθ| < 2°.
"""

from pathlib import Path

import numpy as np
import pytest

from panda_md import SlitPore, SquarePore, density_profile
from panda_md.core.errors import InvalidTrajectoryError, NotDerivedError
from panda_md.synthetic.points import synthetic_trajectory
from panda_md.trajectory.maps import DensityMap, density_map

DENSITY_BULK = 33.0
PORE = SlitPore(width_nm=4.0, length_nm=8.0, base_aspect_ratio=0.5)
MAX_ANGLE_ERROR_DEG = 2.0


@pytest.mark.parametrize(
    ("shape_name", "truth"),
    [
        ("roll", {"theta": 120.0, "delta": 0.03, "phi": 0.4}),
        ("droplet", {"theta": 130.0, "delta": 0.03, "phi": 0.05}),
        ("doughnut", {"theta": 130.0, "delta": 0.03, "phi": 0.2}),
    ],
)
def test_angle_recovered_from_point_clouds(shape_name: str, truth: dict[str, float]) -> None:
    model = PORE.interface(shape_name)
    trajectory = synthetic_trajectory(model, truth, density_bulk_per_nm3=DENSITY_BULK, frame_count=100, seed=1)
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    assert found.width_nm == pytest.approx(PORE.width_nm, abs=1e-9)
    assert found.base_aspect_ratio == pytest.approx(PORE.base_aspect_ratio)
    profiles = density_profile(trajectory, found, "liquid", bins=100)
    result = found.interface(shape_name).fit(profiles.mean(), density_bulk_per_nm3=DENSITY_BULK)
    assert result.values["theta"] == pytest.approx(truth["theta"], abs=MAX_ANGLE_ERROR_DEG)
    assert result.values["phi"] == pytest.approx(truth["phi"], rel=0.03)


def mirror_agreement(density: np.ndarray) -> float:
    """Корреляция карты с её отражением x → −x: 1 — форма симметрична относительно оси в x = 0."""
    return float(np.corrcoef(density.ravel(), density[::-1].ravel())[0, 1])


def test_axis_centering_sharpens_the_map_of_a_wandering_roll() -> None:
    roll = PORE.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.3}, density_bulk_per_nm3=DENSITY_BULK, frame_count=20, seed=2
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    centered = density_map(trajectory, found, "liquid", shape="roll", bins=(40, 40))
    smeared = density_map(trajectory, found, "liquid", bins=(40, 40))
    assert centered.axis_centered and not smeared.axis_centered
    assert mirror_agreement(centered.density_per_nm3) > 0.95
    middle = centered.density_per_nm3[:, 20]
    assert middle[20] == pytest.approx(DENSITY_BULK, rel=0.15)  # в сердцевине валика — объёмная плотность
    assert middle[0] == pytest.approx(0.0, abs=1.0)  # на краю ячейки (за валиком) жидкости нет
    assert np.max(smeared.density_per_nm3) < 0.8 * np.max(centered.density_per_nm3)


def region_mean(density_map_: DensityMap, length_window: tuple[float, float], height_limit: float) -> float:
    """Средняя плотность карты в полосе |z| < height_limit и length_window[0] ≤ |x| < length_window[1] (нм)."""
    along = 0.5 * (density_map_.length_edges_nm[:-1] + density_map_.length_edges_nm[1:])
    height = 0.5 * (density_map_.height_edges_nm[:-1] + density_map_.height_edges_nm[1:])
    rows = (np.abs(along) >= length_window[0]) & (np.abs(along) < length_window[1])
    columns = np.abs(height) < height_limit
    return float(np.mean(density_map_.density_per_nm3[np.ix_(rows, columns)]))


def test_perforation_axis_is_the_center_of_the_hole() -> None:
    perforation = PORE.interface("perforation")
    trajectory = synthetic_trajectory(
        perforation,
        {"theta": 130.0, "delta": 0.03, "phi": 0.8},
        density_bulk_per_nm3=DENSITY_BULK,
        frame_count=50,
        seed=3,
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    centered = density_map(trajectory, found, "liquid", shape="perforation", bins=(40, 40))
    hole = region_mean(centered, (0.0, 0.5), 1.0)
    bulk = region_mean(centered, (3.0, 4.0), 1.0)
    assert bulk == pytest.approx(DENSITY_BULK, rel=0.1)
    assert hole < 0.6 * bulk


def test_map_needs_a_shape_with_an_axis_and_survives_saving(tmp_path: Path) -> None:
    roll = PORE.interface("roll")
    trajectory = synthetic_trajectory(
        roll, {"theta": 120.0, "delta": 0.03, "phi": 0.3}, density_bulk_per_nm3=DENSITY_BULK, frame_count=2, seed=4
    )
    found = SlitPore.from_trajectory(trajectory, walls="walls")
    with pytest.raises(InvalidTrajectoryError, match="оси симметрии"):
        density_map(trajectory, found, "liquid", shape="droplet")
    density = density_map(trajectory, found, "liquid", shape="roll", bins=(10, 12))
    density.save(tmp_path / "map.npz")
    assert DensityMap.load(tmp_path / "map.npz").model_dump_json() == density.model_dump_json()


def test_point_clouds_exist_only_for_slit_shapes() -> None:
    square = SquarePore(side_nm=6.0, length_nm=12.0, layer_width_nm=6.0)
    with pytest.raises(NotDerivedError, match="щели"):
        synthetic_trajectory(
            square.interface("droplet_w1c0"), {"theta": 140.0, "delta": 0.0, "phi": 0.05}, density_bulk_per_nm3=30.0
        )
```

- [ ] **Шаг 2: Убедиться, что тесты падают**

Run: `uv run pytest -q -m "not slow"`
Expected: ошибка сборки `ModuleNotFoundError: No module named 'panda_md.trajectory.maps'`.

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
            save_npz_arrays(target, _profile_arrays(self), self._metadata())
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


def load_npz_arrays(path: Path) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Массивы и метаданные, сохранённые save_npz_arrays."""
    with np.load(path) as archive:
        metadata: dict[str, Any] = json.loads(str(archive["metadata"]))
        arrays = {name: archive[name] for name in archive.files if name != "metadata"}
    return arrays, metadata


def save_npz_arrays(path: Path, arrays: Mapping[str, FloatArray], metadata: Mapping[str, object]) -> None:
    """Массивы — по именам, метаданные — JSON-строкой под ключом metadata."""
    # Any: заглушки numpy типизируют **kwds у savez как bool (allow_pickle); значения — массивы и строка
    payload: dict[str, Any] = {**arrays, "metadata": json.dumps(dict(metadata))}
    np.savez(path, **payload)


def _require_npz(path: Path) -> None:
    """ProfileSet хранится только в .npz: иначе numpy молча дописывает суффикс, и load по тому же имени падает."""
    if path.suffix != ".npz":
        raise ValueError(f"набор профилей хранится в .npz, а не в {path.name!r}")


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
        _require_npz(Path(path))
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
        save_npz_arrays(Path(path), arrays, metadata)

    @classmethod
    def load(cls, path: str | Path) -> "ProfileSet":
        _require_npz(Path(path))
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
    отбрасывается. per=MOLECULE: частица — центр масс выбранных атомов молекулы (с учётом PBC; «name OW» даст
    кислород, а не центр масс воды), N — число молекул.
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
        heights = particle_positions(frame.positions_nm[0], frame.box_nm, molecules)[:, axis]
        center = profile_frame_center(frame, heights, pore, centering, pore_center_nm, period, warned)
        coordinate = wrap_centered(heights - center, period)
        if edges is None:
            first_period = period
            edges = profile_bin_edges(pore, centering, period, bins, bin_width_nm)
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
    empty = sum(1 for accumulator in blocks if accumulator.frame_count == 0)
    if empty:
        warnings.warn(
            f"{empty} блоков без кадров (разрыв во времени траектории) — пропущены", PandaWarning, stacklevel=2
        )
        blocks = [accumulator for accumulator in blocks if accumulator.frame_count > 0]
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


def particle_positions(
    positions: FloatArray, box: FloatArray, molecules: tuple[IndexArray, FloatArray] | None
) -> FloatArray:
    """Позиции частиц (n × 3): атомы или центры масс молекул (molecules — номер молекулы и масса каждого атома)."""
    if molecules is None:
        return positions
    molecule_indices, masses = molecules
    centers: FloatArray = molecule_centers(positions, box, molecule_indices, masses)
    return centers


def profile_frame_center(
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


def profile_bin_edges(
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
    length_axis: Axis = Axis.X  # ось ящика вдоль длинной стороны основания (для 2D-карт и осевого центрирования)
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
            length_axis=length_axis,
            wall_selection=walls,
        )
```

`src/panda_md/trajectory/maps.py`:

```python
"""2D-карта плотности щели: длинная сторона ячейки × ось профиля, усреднение по короткой стороне.

Карта показывает саму форму (валик, каплю, отверстие) — для проверки, что выбранная форма та, и для рисунков. Без
осевого центрирования форма, блуждающая по ящику, размазывается; с ним ось формы в каждом кадре ставится в x = 0.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import Field, model_validator

from panda_md.core.centering import axis_position
from panda_md.core.enums import Frame, ParticleGrouping, ProfileCentering
from panda_md.core.errors import InvalidTrajectoryError
from panda_md.core.pbc import wrap_centered
from panda_md.core.profile import load_npz_arrays, save_npz_arrays
from panda_md.core.timing import TimeValue
from panda_md.core.trajectory import Trajectory, check_box
from panda_md.core.types import Array1D, Array2D, FloatArray, IndexArray, PandaModel
from panda_md.pores.slit.pore import SlitPore
from panda_md.trajectory.density import particle_positions, profile_bin_edges, profile_frame_center

DEFAULT_MAP_BINS = (100, 100)


class DensityMap(PandaModel):
    """ρ (нм⁻³) на сетке: вдоль длинной стороны × вдоль оси профиля, среднее по кадрам и по короткой стороне."""

    length_edges_nm: Array1D
    height_edges_nm: Array1D
    density_per_nm3: Array2D  # бины по длинной стороне × бины по оси профиля
    frame: Frame
    axis_centered: bool  # ось формы в каждом кадре поставлена в x = 0
    frame_count: int = Field(ge=1)
    selection: str

    @model_validator(mode="after")
    def _check_shapes(self) -> "DensityMap":
        expected = (self.length_edges_nm.size - 1, self.height_edges_nm.size - 1)
        if self.density_per_nm3.shape != expected:
            raise ValueError(f"карта формы {self.density_per_nm3.shape}, а по краям бинов — {expected}")
        return self

    def save(self, path: str | Path) -> None:
        arrays = {name: getattr(self, name) for name in ("length_edges_nm", "height_edges_nm", "density_per_nm3")}
        save_npz_arrays(Path(path), arrays, self.model_dump(mode="json", exclude=set(arrays)))

    @classmethod
    def load(cls, path: str | Path) -> "DensityMap":
        arrays, metadata = load_npz_arrays(Path(path))
        return cls(**arrays, **metadata)


def density_map(
    trajectory: Trajectory,
    pore: SlitPore,
    select: str | Sequence[int],
    *,
    shape: str | None = None,
    centering: ProfileCentering | str = ProfileCentering.PORE,
    bins: tuple[int, int] = DEFAULT_MAP_BINS,
    start: TimeValue = 0.0,
    stop: TimeValue | None = None,
    stride: int = 1,
    pore_center_nm: float | None = None,
    per: ParticleGrouping | str = ParticleGrouping.ATOM,
) -> DensityMap:
    """Карта плотности выборки в плоскости «длинная сторона × ось профиля».

    shape — имя формы поры с осью симметрии (`roll`, `doughnut`, `perforation`): тогда в каждом кадре ось формы
    ставится в x = 0 (осевое центрирование); без shape — координата вдоль длинной стороны как в траектории.
    Ось профиля центрируется так же, как в `density_profile`.
    """
    centering = ProfileCentering(centering)
    grouping = ParticleGrouping(per)
    axis = None
    if shape is not None:
        axis = pore.interface(shape).shape.symmetry_axis
        if axis is None:
            raise InvalidTrajectoryError(f"у формы {shape} нет оси симметрии — осевое центрирование не определено")
    liquid = trajectory.select(select)
    groups: list[IndexArray] = [liquid]
    if centering is ProfileCentering.PORE and pore_center_nm is None:
        if pore.wall_selection is None:
            raise InvalidTrajectoryError("центрирование по поре: задай pore_center_nm или wall_selection у поры")
        groups.append(trajectory.select(pore.wall_selection))
    molecules = trajectory.molecules(liquid) if grouping is ParticleGrouping.MOLECULE else None
    normal = pore.normal_axis.column
    length_column = pore.length_axis.column
    (width_column,) = (column for column in range(3) if column not in (normal, length_column))
    length_edges: FloatArray | None = None
    height_edges: FloatArray | None = None
    total: FloatArray | None = None
    frame_count = 0
    warned: set[str] = set()
    for frame in trajectory.frames(groups, start=start, stop=stop, stride=stride):
        check_box(frame.box_nm)
        positions = particle_positions(frame.positions_nm[0], frame.box_nm, molecules)
        period = float(frame.box_nm[normal])
        center = profile_frame_center(frame, positions[:, normal], pore, centering, pore_center_nm, period, warned)
        heights = wrap_centered(positions[:, normal] - center, period)
        length = float(frame.box_nm[length_column])
        along = positions[:, length_column]
        if axis is not None:
            lateral = positions[:, [length_column, width_column]]
            periods = frame.box_nm[[length_column, width_column]]
            along = along - axis_position(lateral, periods, axis)[0]
        along = wrap_centered(along, length)
        if length_edges is None or height_edges is None or total is None:
            length_edges = np.linspace(-0.5 * length, 0.5 * length, bins[0] + 1)
            height_edges = profile_bin_edges(pore, centering, period, bins[1], None)
            total = np.zeros((bins[0], height_edges.size - 1))
        counts, _, _ = np.histogram2d(along, heights, bins=(length_edges, height_edges))
        cell = np.outer(np.diff(length_edges), np.diff(height_edges)) * float(frame.box_nm[width_column])
        total += counts / cell
        frame_count += 1
    assert length_edges is not None and height_edges is not None and total is not None  # кадры есть
    return DensityMap(
        length_edges_nm=length_edges,
        height_edges_nm=height_edges,
        density_per_nm3=total / frame_count,
        frame=Frame.PORE if centering is ProfileCentering.PORE else Frame.LIQUID,
        axis_centered=axis is not None,
        frame_count=frame_count,
        selection=select if isinstance(select, str) else f"{len(liquid)} индексов",
    )
```

- [ ] **Шаг 4: Тесты проходят**

Run: `uv run pytest -q`
Expected: `466 passed, 28 skipped`; md-тесты и регрессия со старой PANDA зелёные (путь данных не изменился).

- [ ] **Шаг 5: Проверка и коммит**

Run: `make check` — без ошибок.

```bash
git add src/panda_md tests/synthetic/test_point_clouds.py
git commit -m "Осевое центрирование и 2D-карты плотности щели (density_map)" \
  -m "Ось формы в каждом кадре — в x = 0; у валика карта симметрична, без центрирования блуждающая форма размазывается. Помощники пути данных стали общими для профиля и карты." \
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


def test_three_dimensional_shapes_are_public() -> None:
    expected = {"AxisLocator", "AxisOrientation", "DensityMap", "SymmetryAxis", "density_map", "synthetic_trajectory"}
    assert expected <= set(panda_md.__all__)
```

- [ ] **Шаг 2: Убедиться, что тест падает**

Run: `uv run pytest -q tests/test_package.py`
Expected: `1 failed` — `test_three_dimensional_shapes_are_public`.

- [ ] **Шаг 3: Реализация и документация**

Заменить `src/panda_md/__init__.py` целиком:

`src/panda_md/__init__.py`:

```python
"""PANDA: Predicting Angle from Nanoscale Density Analysis."""

from importlib.metadata import PackageNotFoundError, version

from panda_md.analysis.blocks import FitResultSet, StatisticalInefficiency, fit_blocks, statistical_inefficiency
from panda_md.analysis.compare import ShapeComparison, ShapeFit, compare_shapes
from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import (
    Axis,
    AxisLocator,
    AxisOrientation,
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
from panda_md.synthetic.points import synthetic_trajectory
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile
from panda_md.trajectory.maps import DensityMap, density_map
from panda_md.trajectory.reader import load_trajectory

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = [
    "ArrayTrajectory",
    "Axis",
    "AxisLocator",
    "AxisOrientation",
    "DensityMap",
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
    "SymmetryAxis",
    "TrianglePore",
    "Unit",
    "__version__",
    "compare_shapes",
    "density_map",
    "density_profile",
    "fit_blocks",
    "load_trajectory",
    "statistical_inefficiency",
    "synthetic_trajectory",
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
| трёхмерные формы | `unit/test_shapes_3d.py` | математика | `contains` щели: капля — сферический сегмент на слое δ, червь — цилиндрический сегмент (прямая геометрия, совпадение > 99.95 % точек); для всех шести форм при k = 1 и 0.4 доля точек облака по слоям = профиль, всего — φ (контракт §9); оси симметрии объявлены у валика (горизонтальная), пончика и перфорации (вертикальные, у перфорации — центр отверстия); положение оси через границу ящика; у форм треугольника и квадрата `contains` нет (`NotDerivedError`) | 400 000 случайных точек ячейки | доли по слоям ± 0.02, φ ± 4σ выборки | — |
| облака точек | `synthetic/test_point_clouds.py` | физическая модель | `synthetic_trajectory`: частицы равномерно в форме (ρ_bulk), ось формы блуждает по ячейке, подложка — решётка, щель через границу ящика → `from_trajectory` → `density_profile` → фит (валик, капля, пончик); 2D-карта: с осевым центрированием валик симметричен и в сердцевине ρ_bulk, без него размазан; у перфорации провал плотности в x = 0; форма без оси — ошибка; сохранение карты | щель 4 × 8 нм, k = 0.5, 100 кадров | |Δθ| < 2° (§9), φ ± 3 % | — |
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
````

Спецификация: после строки D25 добавить:

```markdown
| D26 | Трёхмерная форма — `InterfaceShape.contains(points, geometry, values)` (безразмерные точки, ось формы в начале координат); у щели выводится из профиля (сечение: круг, отверстие, полоса, всё), у многоугольных пор пока нет (O9). Ось симметрии — `symmetry_axis: SymmetryAxis(orientation, locate)` в классе формы: валик — горизонтальная, пончик и перфорация — вертикальные (у перфорации — центр отверстия), у остальных нет. `density_map` — 2D-карта щели с осевым центрированием; `synthetic_trajectory` — облака точек (стадия 6) |
```

`ROADMAP.md`, «Фаза 6»: строку `- **6б, трёхмерные формы:** …` заменить на:

```markdown
- **6б, трёхмерные формы — выполнено** (план `docs/superpowers/plans/2026-10-16-stage6-shapes-3d.md`): `contains` у форм щели (из профиля), оси симметрии в классах форм, осевое центрирование и 2D-карты (`density_map`), облака точек (`synthetic_trajectory`) — сквозной путь |Δθ| < 2°. У треугольника и квадрата `contains` нет — модели ещё меняются (O9).
```

`README.md`: в примере «From an MD trajectory» последней строкой добавить
`water_map = panda.density_map(traj, pore, "resname WATER and name OW", shape="roll")  # ось валика в x = 0`.

- [ ] **Шаг 4: Полная проверка и коммит**

Run: `make check`, `make test-env`, `make wheel-check`; Python 3.10: `UV_PROJECT_ENVIRONMENT=.venv-py310 uv run --no-sync pytest -q`. Всё зелёное, `467 passed, 28 skipped`.

```bash
git add src/panda_md/__init__.py tests docs ROADMAP.md README.md
git commit -m "Стадия 6 (фаза 6б) закрыта: трёхмерные формы в публичном API, описание тестов, D26, ROADMAP, README" \
  -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
