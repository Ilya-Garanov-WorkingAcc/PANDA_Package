"""Общая базовая модель pydantic, тип одномерного массива numpy и неизменяемый словарь."""

from collections.abc import Iterator, Mapping
from typing import Annotated, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, PlainValidator

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

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
