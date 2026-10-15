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
