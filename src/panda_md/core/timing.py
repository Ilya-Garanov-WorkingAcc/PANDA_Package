"""Время в пользовательских единицах: «2 ns», «500 ps» или число пикосекунд."""

import math
import re

from panda_md.core.enums import TimeUnit

TimeValue = float | str  # число — пикосекунды; строка — «<число> <единица>»
TIME_PATTERN = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Zµμ]+)\s*$")
MICRO_SIGNS = ("µ", "μ")  # знак микро и греческая мю — обе пишутся как «us»


def to_picoseconds(value: TimeValue) -> float:
    """Перевести время в пикосекунды: 2000.0 и «2000» → 2000.0, «2 ns» → 2000.0, «.5 ns» → 500.0.
    Время — конечное неотрицательное число; логическое значение, NaN и бесконечность — ошибка.
    """
    if isinstance(value, bool):
        raise ValueError(f"время {value!r}: ожидается число пс или строка вида «2 ns», а не логическое значение")
    picoseconds = _parse(value)
    if not math.isfinite(picoseconds) or picoseconds < 0.0:
        raise ValueError(f"время {value!r}: нужно конечное неотрицательное значение")
    return picoseconds


def _parse(value: TimeValue) -> float:
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
    unit_text = unit_text.lower()
    for sign in MICRO_SIGNS:
        unit_text = unit_text.replace(sign, "u")
    try:
        unit = TimeUnit(unit_text)
    except ValueError:
        known = ", ".join(unit.value for unit in TimeUnit)
        raise ValueError(f"единица времени {unit_text!r} неизвестна; доступны: {known}") from None
    return float(number) * unit.picoseconds
