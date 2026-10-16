"""Общая базовая модель pydantic, типы массивов numpy с проверкой и неизменяемый словарь."""

from collections.abc import Callable, Iterator, Mapping
from typing import Annotated, Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator, BaseModel, ConfigDict, PlainSerializer, PlainValidator
from typing_extensions import Self

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
        protected_namespaces=(),  # поля model_curve и т. п.: pydantic < 2.10 предупреждает о префиксе model_
    )

    def __eq__(self, other: object) -> bool:
        """Сравнение по значениям; массивы — поэлементно (иначе == падает: truth value of an array is ambiguous)."""
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, PandaModel)
        return _values_equal(self.__dict__, other.__dict__)

    def __hash__(self) -> int:
        return hash((type(self), _hashable(self.__dict__)))

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        duplicate = super().__deepcopy__(memo)
        _freeze_arrays(duplicate.__dict__)
        return duplicate

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        _freeze_arrays(self.__dict__)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Копия; с update — заново через проверку модели (pydantic по умолчанию её пропускает)."""
        if not update:
            return super().model_copy(deep=deep)
        values = {field.alias or name: getattr(self, name) for name, field in type(self).model_fields.items()}
        update = {
            (type(self).model_fields[name].alias if name in type(self).model_fields else None) or name: value
            for name, value in update.items()
        }
        return self.model_validate({**values, **update})


def _values_equal(first: object, second: object) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (
            isinstance(first, np.ndarray)
            and isinstance(second, np.ndarray)
            and first.shape == second.shape
            and bool(np.array_equal(first, second))
        )
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(_values_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(_values_equal(a, b) for a, b in zip(first, second, strict=True))
    return bool(first == second)


def _hashable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return (value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_hashable(item) for item in value)
    return value


def _freeze_arrays(value: object) -> None:
    """Массивы внутри копии — снова только для чтения (deepcopy и pickle создают изменяемые)."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, Mapping):
        for item in value.values():
            _freeze_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _freeze_arrays(item)
