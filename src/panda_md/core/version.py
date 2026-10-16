"""Версия установленной библиотеки: пишется в результаты и сохраняемые файлы (воспроизводимость расчёта)."""

from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION_NAME = "panda-md"
UNKNOWN_VERSION = "0.0.0"  # пакет не установлен (запуск из исходников без установки)


def library_version() -> str:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VERSION
