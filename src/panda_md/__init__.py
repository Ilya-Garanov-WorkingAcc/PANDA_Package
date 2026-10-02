"""PANDA: Predicting ANgle from nanoscale Density Analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = ["__version__"]
