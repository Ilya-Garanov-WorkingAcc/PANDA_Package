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
