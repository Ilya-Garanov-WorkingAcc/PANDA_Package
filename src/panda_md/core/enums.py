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
