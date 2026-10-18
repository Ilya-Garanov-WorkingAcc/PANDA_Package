"""Pore type: geometry, position in the simulation box, reduction of the profile to dimensionless form, shapes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from panda_md.core.enums import Axis, Normalization
from panda_md.core.errors import NotDerivedError, UnknownShapeError
from panda_md.core.profile import DensityProfile, ReducedProfile, reduce_profile
from panda_md.core.shape import InterfaceShape, PoreGeometry
from panda_md.core.types import FloatArray, PandaModel

if TYPE_CHECKING:
    from panda_md.core.model import InterfaceModel

# Any: the registry holds shapes with different geometry subclasses; the concrete type is known only to the pore.
ShapeClassT = TypeVar("ShapeClassT", bound=type[InterfaceShape[Any]])


class Pore(PandaModel, ABC):
    """A concrete pore with dimensions. Each subclass has its own shape registry."""

    name: ClassVar[str]
    shape_registry: ClassVar[dict[str, type[InterfaceShape[Any]]]]
    normal_axis: Axis = Axis.Z  # box axis along which the profile is built
    wall_selection: str | None = None  # wall atoms: used to find the pore centre in each frame

    @classmethod
    def own_registry(cls) -> dict[str, type[InterfaceShape[Any]]]:
        """Registry of exactly this pore class: subclasses do not share the registry with the parent."""
        if "shape_registry" not in cls.__dict__:
            cls.shape_registry = {}
        return cls.shape_registry

    @classmethod
    def register(cls, shape_class: ShapeClassT) -> ShapeClassT:
        """Decorator: attach a shape to this pore type."""
        registry = cls.own_registry()
        if shape_class.name in registry:
            raise ValueError(f"shape '{shape_class.name}' is already registered in pore '{cls.name}'")
        registry[shape_class.name] = shape_class
        return shape_class

    @classmethod
    def shape_names(cls) -> tuple[str, ...]:
        return tuple(cls.own_registry())

    @abstractmethod
    def geometry(self) -> PoreGeometry:
        """Dimensionless geometry for the shape formulas."""

    @abstractmethod
    def characteristic_size_nm(self) -> float:
        """Size to which the coordinate and δ are reduced: H for the slit, a for polygons."""

    @abstractmethod
    def volume_nm3(self) -> float:
        """Pore volume relative to which φ is defined."""

    @abstractmethod
    def bin_cross_section_nm2(self, box_nm: FloatArray) -> float:
        """Cross-section area of a bin layer (a bin is a layer of thickness dz along the profile axis)."""

    def base_area_nm2(self) -> float | None:
        """Box base area the pore expects (for checking dimensions); None — does not depend on the box."""
        return None

    def coordinate_bounds_nm(self) -> tuple[float, float]:
        """Pore walls along the profile axis in the pore frame, nm."""
        lower, upper = self.geometry().coordinate_bounds()
        size = self.characteristic_size_nm()
        return lower * size, upper * size

    def expected_wall_gap_nm(self) -> float:
        """Empty gap between the centres of the outermost wall atoms that the pore expects."""
        lower, upper = self.coordinate_bounds_nm()
        return upper - lower

    def locate_center_nm(self, wall_coordinates_nm: FloatArray, period_nm: float) -> float:
        """Position of the pore centre along the profile axis from the wall atom coordinates (with PBC)."""
        raise NotDerivedError(
            f"locating the centre of pore '{self.name}' from wall atoms is not implemented; "
            "give the centre explicitly (pore_center_nm)"
        )

    def reduce(
        self, profile: DensityProfile, normalization: Normalization, density_bulk_per_nm3: float | None
    ) -> ReducedProfile:
        return reduce_profile(profile, self.characteristic_size_nm(), normalization, density_bulk_per_nm3)

    def interface(self, shape_name: str) -> "InterfaceModel":
        from panda_md.core.model import InterfaceModel

        registry = self.own_registry()
        if shape_name not in registry:
            raise UnknownShapeError(
                f"unknown shape '{shape_name}' in pore '{self.name}'; available: {', '.join(registry)}"
            )
        return InterfaceModel(self, registry[shape_name]())

    def available_interfaces(self) -> list["InterfaceModel"]:
        return [self.interface(shape_name) for shape_name in self.own_registry()]
