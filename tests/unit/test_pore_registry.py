"""Pore shape registry: a shape is plugged in without editing the core, each pore type has its own registry."""

from collections.abc import Iterator
from typing import ClassVar

import pytest
from pydantic import ValidationError

from panda_md.core.enums import Frame, Regime
from panda_md.core.parameters import CONTACT_ANGLE, Parameter, ParameterValues
from panda_md.core.pore import Pore
from panda_md.core.shape import InterfaceShape
from panda_md.pores.slit.pore import SlitGeometry, SlitPore

SLIT = SlitPore(width_nm=8.0, length_nm=16.0)


class FlatLayer(InterfaceShape[SlitGeometry]):
    """A shape declared right in the test: a user could declare it the same way in a script."""

    name: ClassVar[str] = "test_layer"
    parameters: ClassVar[tuple[Parameter, ...]] = (CONTACT_ANGLE,)
    regimes: ClassVar[dict[Regime, tuple[float, float]]] = {Regime.NON_WETTING: (90.0, 180.0)}
    frame: ClassVar[Frame] = Frame.PORE

    def feasible(self, geometry: SlitGeometry, values: ParameterValues) -> bool:
        return True


class OtherPore(Pore):
    name: ClassVar[str] = "test_other"
    size_nm: float = 1.0

    def geometry(self) -> SlitGeometry:
        return SlitGeometry(pore_length_ratio=1.0)

    def characteristic_size_nm(self) -> float:
        return self.size_nm

    def volume_nm3(self) -> float:
        return self.size_nm**3


@pytest.fixture
def registered_layer() -> Iterator[type[FlatLayer]]:
    SlitPore.register(FlatLayer)
    yield FlatLayer
    del SlitPore.own_registry()["test_layer"]


def test_new_shape_registers_without_core_changes(registered_layer: type[FlatLayer]) -> None:
    assert "test_layer" in [model.shape.name for model in SLIT.available_interfaces()]
    model = SLIT.interface("test_layer")
    assert isinstance(model.shape, registered_layer)
    assert model.parameters.names == ("theta", "shift")


def test_duplicate_registration_is_rejected(registered_layer: type[FlatLayer]) -> None:
    with pytest.raises(ValueError, match="already registered"):
        SlitPore.register(registered_layer)


def test_registry_belongs_to_pore_type(registered_layer: type[FlatLayer]) -> None:
    assert "test_layer" in SlitPore.shape_names()
    assert "test_layer" not in OtherPore.shape_names()


def test_unknown_shape_lists_available() -> None:
    with pytest.raises(KeyError, match="available:"):
        SLIT.interface("no_such_shape")


def test_pore_rejects_non_positive_size() -> None:
    with pytest.raises(ValidationError):
        SlitPore(width_nm=0.0, length_nm=10.0)


def test_pore_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SlitPore(width_nm=8.0, length_nm=16.0, height_nm=3.0)  # type: ignore[call-arg]


def test_pore_prints_its_dimensions_readably() -> None:
    pore = SlitPore(width_nm=8.269, length_nm=17.271, base_aspect_ratio=0.297, wall_selection="resname GRAPHITE")
    text = str(pore)
    assert text.startswith("slit pore: ")
    assert "width_nm = 8.269" in text and "length_nm = 17.271" in text and "base_aspect_ratio = 0.297" in text
    assert "wall_selection = 'resname GRAPHITE'" in text and "<Axis" not in text
    assert repr(pore) == text  # the notebook shows the same readable form
