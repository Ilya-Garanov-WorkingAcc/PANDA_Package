"""Pore types by name: for the command line and settings files."""

from panda_md.core.errors import PandaError, UnknownPoreError
from panda_md.core.pore import Pore
from panda_md.pores.cylinder import CylinderPore
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore

PORE_TYPES: tuple[type[Pore], ...] = (SlitPore, SquarePore, TrianglePore, CylinderPore)  # a new pore type is added here


def pore_types() -> dict[str, type[Pore]]:
    types: dict[str, type[Pore]] = {}
    for pore_type in PORE_TYPES:
        if pore_type.name in types:
            raise PandaError(f"two pore types named {pore_type.name!r}: {types[pore_type.name]} and {pore_type}")
        types[pore_type.name] = pore_type
    return types


def pore_class(name: str) -> type[Pore]:
    types = pore_types()
    if name not in types:
        raise UnknownPoreError(f"unknown pore type {name!r}; available: {', '.join(types)}")
    return types[name]
