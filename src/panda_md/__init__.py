"""PANDA: Predicting Angle from Nanoscale Density Analysis."""

from importlib.metadata import PackageNotFoundError, version

from panda_md.core.enums import Frame, LossFunction, Normalization, OptimizerMethod, ParameterName, Regime, Unit
from panda_md.core.errors import PandaError, PandaWarning
from panda_md.core.model import InterfaceModel
from panda_md.core.parameters import Parameter
from panda_md.core.profile import DensityProfile
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = [
    "DensityProfile",
    "FitResult",
    "FitSpec",
    "Frame",
    "InterfaceModel",
    "LossFunction",
    "Normalization",
    "OptimizerMethod",
    "PandaError",
    "PandaWarning",
    "Parameter",
    "ParameterName",
    "Regime",
    "SlitPore",
    "SquarePore",
    "TrianglePore",
    "Unit",
    "__version__",
]
