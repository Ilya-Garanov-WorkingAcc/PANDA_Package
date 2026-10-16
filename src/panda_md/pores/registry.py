"""Типы пор по имени: для командной строки и файлов настроек."""

from panda_md.core.errors import UnknownPoreError
from panda_md.core.pore import Pore
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore

PORE_TYPES: tuple[type[Pore], ...] = (SlitPore, SquarePore, TrianglePore)  # новый тип поры добавляется сюда


def pore_types() -> dict[str, type[Pore]]:
    return {pore_class.name: pore_class for pore_class in PORE_TYPES}


def pore_class(name: str) -> type[Pore]:
    types = pore_types()
    if name not in types:
        raise UnknownPoreError(f"тип поры {name!r} неизвестен; доступны: {list(types)}")
    return types[name]
