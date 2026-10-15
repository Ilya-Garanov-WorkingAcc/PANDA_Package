"""Тип поры: геометрия, приведение профиля к безразмерному виду, реестр форм."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from panda_md.core.enums import Normalization
from panda_md.core.profile import DensityProfile, ReducedProfile, reduce_profile
from panda_md.core.shape import InterfaceShape, PoreGeometry
from panda_md.core.types import PandaModel

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel

# Any: реестр хранит формы с разными подклассами геометрии; конкретный тип известен только самой поре.
ShapeClassT = TypeVar("ShapeClassT", bound=type[InterfaceShape[Any]])


class Pore(PandaModel, ABC):
    """Конкретная пора с размерами. Каждый подкласс имеет свой реестр форм."""

    name: ClassVar[str]
    shape_registry: ClassVar[dict[str, type[InterfaceShape[Any]]]]

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

    def reduce(
        self, profile: DensityProfile, normalization: Normalization, density_bulk_per_nm3: float | None
    ) -> ReducedProfile:
        return reduce_profile(profile, self.characteristic_size_nm(), normalization, density_bulk_per_nm3)

    def interface(self, shape_name: str) -> "InterfaceModel":
        from panda_md.core.model import InterfaceModel

        registry = self.own_registry()
        if shape_name not in registry:
            raise KeyError(f"в поре {self.name} нет формы {shape_name}; доступны: {list(registry)}")
        return InterfaceModel(self, registry[shape_name]())

    def available_interfaces(self) -> list["InterfaceModel"]:
        return [self.interface(shape_name) for shape_name in self.own_registry()]
