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
                source = "задано в fixed"
            elif name in values_from_data:
                user_value = parameter.to_user(values_from_data[name])
                source = "оценено из данных — проверь ρ_bulk, единицы профиля и размер поры"
            elif parameter.default is not None:
                user_value = parameter.default
                source = "значение по умолчанию"
            else:
                raise MissingParameterValueError(f"задай значение {name} в fixed или освободи его в optimize")
            lower, upper = parameter.bounds
            if not lower <= user_value <= upper:
                raise InvalidBoundsError(
                    f"{name} = {user_value:.4g} вне допустимых границ [{lower}, {upper}] ({source})"
                )
            fixed_values[name] = parameter.to_internal(user_value)

        bounds_not_free = sorted(set(bounds) - set(free_names))
        if bounds_not_free:
            raise ParameterConflictError(f"границы заданы для несвободных параметров {bounds_not_free}")
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        user_bounds: dict[str, tuple[float, float]] = {}
        for name in free_names:
            parameter = self.parameters[name]
            lower, upper = parameter.bounds
            if name in bounds:
                requested_lower, requested_upper = bounds[name]
                if not (math.isfinite(requested_lower) and math.isfinite(requested_upper)):
                    raise InvalidBoundsError(f"{name}: границы {bounds[name]} должны быть конечными числами")
                lower, upper = max(lower, requested_lower), min(upper, requested_upper)
                if not lower < upper:
                    raise InvalidBoundsError(f"{name}: границы {bounds[name]} не пересекаются с {parameter.bounds}")
            user_bounds[name] = (lower, upper)
            lower_bounds.append(parameter.to_internal(lower))
            upper_bounds.append(parameter.to_internal(upper))

        not_free = sorted(set(initial) - set(free_names))
        if not_free:
            raise ParameterConflictError(f"стартовые значения заданы для несвободных параметров {not_free}")
        for name, value in initial.items():
            lower, upper = user_bounds[name]
            if not lower <= value <= upper:
                raise InvalidBoundsError(f"{name}: стартовое значение {value} вне границ фита [{lower}, {upper}]")
        initial_values = {name: self.parameters[name].to_internal(value) for name, value in initial.items()}
        return FreeSpec(
            self.parameters,
            free_names,
            fixed_values,
            np.array(lower_bounds, dtype=np.float64),
            np.array(upper_bounds, dtype=np.float64),
            initial_values,
        )
