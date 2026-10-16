"""Время в пользовательских единицах: «2 ns», «500 ps» или число пикосекунд."""

import re

from panda_md.core.enums import TimeUnit

TimeValue = float | str  # число — пикосекунды; строка — «<число> <единица>»
TIME_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*([a-zA-Zµ]+)\s*$")


def to_picoseconds(value: TimeValue) -> float:
    """Перевести время в пикосекунды: 2000.0 и «2000» → 2000.0, «2 ns» → 2000.0, «1.5ns» → 1500.0."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # строка с одним числом — тоже пикосекунды
    except ValueError:
        pass
    match = TIME_PATTERN.match(value)
    if match is None:
        raise ValueError(f"время {value!r} не разобрано: ожидается число пс или строка вида «2 ns»")
    number, unit_text = match.groups()
    unit_text = unit_text.lower().replace("µ", "u")
    try:
        unit = TimeUnit(unit_text)
    except ValueError:
        known = ", ".join(unit.value for unit in TimeUnit)
        raise ValueError(f"единица времени {unit_text!r} неизвестна; доступны: {known}") from None
    return float(number) * unit.picoseconds
