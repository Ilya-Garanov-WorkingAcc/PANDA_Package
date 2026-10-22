"""Time in user units: '2 ns', '500 ps' or a number of picoseconds."""

import math
import re

from panda_md.core.enums import TimeUnit

TimeValue = float | str  # number — picoseconds; string — '<number> <unit>'
TIME_PATTERN = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Zµμ]+)\s*$")
MICRO_SIGNS = ("µ", "μ")  # micro sign and Greek mu — both are written as 'us'


def to_picoseconds(value: TimeValue) -> float:
    """Convert a time to picoseconds: 2000.0 and '2000' → 2000.0, '2 ns' → 2000.0, '.5 ns' → 500.0.
    The time is a finite non-negative number; a boolean, NaN and infinity are errors.
    """
    if isinstance(value, bool):
        raise ValueError(f"time {value!r} is a boolean; give a number of ps or a string like '2 ns'")
    picoseconds = _parse(value)
    if not math.isfinite(picoseconds) or picoseconds < 0.0:
        raise ValueError(f"time {value!r} is not a finite non-negative value; give a time ≥ 0")
    return picoseconds


def _parse(value: TimeValue) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # a string with a single number is also picoseconds
    except ValueError:
        pass
    match = TIME_PATTERN.match(value)
    if match is None:
        raise ValueError(f"cannot parse time {value!r}; give a number of ps or a string like '2 ns'")
    number, unit_text = match.groups()
    unit_text = unit_text.lower()
    for sign in MICRO_SIGNS:
        unit_text = unit_text.replace(sign, "u")
    try:
        unit = TimeUnit(unit_text)
    except ValueError:
        known = ", ".join(unit.value for unit in TimeUnit)
        raise ValueError(f"unknown time unit {unit_text!r}; available: {known}") from None
    return float(number) * unit.picoseconds
