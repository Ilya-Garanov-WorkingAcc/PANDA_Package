"""Разбор значений опций командной строки: «имя=значение», «имя=нижняя:верхняя», стенки, пора."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from panda_md.core.errors import PandaError
from panda_md.core.pore import Pore
from panda_md.pores.registry import pore_class

DEFAULT_PORE = "slit"
PORE_TYPE_KEY = "type"  # тип поры в JSON поры и в разделе [pore] конфига
WALL_COORDINATE_COUNT = 2  # «--walls z1,z2»: координаты двух стенок


class CommandLineError(PandaError):
    """Неверное значение опции или аргумента командной строки."""


def split_names(items: Sequence[str]) -> tuple[str, ...]:
    """«--optimize theta --optimize delta» и «--optimize theta,delta» — одно и то же."""
    return tuple(name.strip() for item in items for name in item.split(",") if name.strip())


def parse_assignments(items: Sequence[str], option: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise CommandLineError(f"{option} {item!r}: ожидается имя=значение")
        assignments[name.strip()] = value.strip()
    return assignments


def parse_numbers(items: Sequence[str], option: str) -> dict[str, float]:
    numbers: dict[str, float] = {}
    for name, value in parse_assignments(items, option).items():
        numbers[name] = _number(value, f"{option} {name}")
    return numbers


def parse_bounds(items: Sequence[str], option: str = "--bounds") -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for item in items:
        name, separator, interval = item.partition("=")
        lower, colon, upper = interval.partition(":")
        if not separator or not colon or not name.strip():
            raise CommandLineError(f"{option} {item!r}: ожидается имя=нижняя:верхняя")
        bounds[name.strip()] = (_number(lower, f"{option} {name}"), _number(upper, f"{option} {name}"))
    return bounds


def parse_walls(text: str) -> tuple[str | None, float | None]:
    """«--walls»: выборка атомов стенок или две координаты стенок «z1,z2» (нм) → (выборка, центр поры)."""
    parts = text.split(",")
    if len(parts) == WALL_COORDINATE_COUNT:
        try:
            first, second = (float(part) for part in parts)
        except ValueError:
            return text, None
        return None, 0.5 * (first + second)
    return text, None


def pore_request(
    pore_name: str | None, pore_file: Path | None, fields: Sequence[str]
) -> tuple[type[Pore], dict[str, object]]:
    """Тип поры и её поля из JSON-файла (--pore-file) и «-g имя=значение»; поля из -g заменяют поля файла."""
    values: dict[str, object] = {}
    if pore_file is not None:
        values = _read_pore_file(pore_file)
    file_type = values.pop(PORE_TYPE_KEY, None)
    if file_type is not None and pore_name is not None and file_type != pore_name:
        raise CommandLineError(f"в {pore_file} пора {file_type!r}, а в --pore {pore_name!r}")
    values.update(parse_assignments(fields, "-g"))
    return pore_class(str(file_type or pore_name or DEFAULT_PORE)), values


def missing_fields(pore_type: type[Pore], values: Mapping[str, object]) -> list[str]:
    """Обязательные поля поры (размеры), которых нет в values."""
    return [name for name, field in pore_type.model_fields.items() if field.is_required() and name not in values]


def require_dimensions(pore_type: type[Pore], values: Mapping[str, object]) -> Pore:
    missing = missing_fields(pore_type, values)
    if missing:
        raise CommandLineError(
            f"задай размеры поры {pore_type.name}: --pore-file pore.json или -g поле=значение; "
            f"не хватает: {', '.join(missing)}"
        )
    return pore_type.model_validate(dict(values))


def load_pore(pore_name: str | None, pore_file: Path | None, fields: Sequence[str]) -> Pore:
    """Пора с заданными размерами: из --pore-file и/или -g."""
    pore_type, values = pore_request(pore_name, pore_file, fields)
    return require_dimensions(pore_type, values)


def pore_json(pore: Pore, indent: int | None = 2) -> str:
    """JSON поры с её типом — читается обратно через --pore-file и раздел [pore] конфига."""
    return json.dumps({PORE_TYPE_KEY: pore.name, **pore.model_dump(mode="json")}, ensure_ascii=False, indent=indent)


def _read_pore_file(path: Path) -> dict[str, object]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CommandLineError(f"{path}: не JSON ({error})") from None
    if not isinstance(content, dict):
        raise CommandLineError(f"{path}: ожидается объект JSON с полями поры")
    return content


def _number(text: str, context: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise CommandLineError(f"{context}: {text!r} — не число") from None
