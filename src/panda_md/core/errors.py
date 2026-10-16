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
