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
