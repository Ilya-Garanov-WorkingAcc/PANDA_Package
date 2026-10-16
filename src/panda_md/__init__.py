"""PANDA: Predicting Angle from Nanoscale Density Analysis."""

from importlib.metadata import PackageNotFoundError, version

from panda_md.analysis.blocks import FitResultSet, StatisticalInefficiency, fit_blocks, statistical_inefficiency
from panda_md.analysis.compare import ShapeComparison, ShapeFit, compare_shapes
from panda_md.core.enums import (
    Axis,
    Frame,
    LossFunction,
    Normalization,
    OptimizerMethod,
    Orientation,
    ParameterName,
    ParticleGrouping,
    ProfileCentering,
    Regime,
    Unit,
)
from panda_md.core.errors import PandaError, PandaWarning
from panda_md.core.model import InterfaceModel
from panda_md.core.parameters import Parameter
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile
from panda_md.trajectory.reader import load_trajectory

try:
    __version__ = version("panda-md")
except PackageNotFoundError:  # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0"

__all__ = [
    "ArrayTrajectory",
    "Axis",
    "DensityProfile",
    "FitResult",
    "FitResultSet",
    "FitSpec",
    "Frame",
    "InterfaceModel",
    "LossFunction",
    "Normalization",
    "OptimizerMethod",
    "Orientation",
    "PandaError",
    "PandaWarning",
    "Parameter",
    "ParameterName",
    "ParticleGrouping",
    "ProfileCentering",
    "ProfileSet",
    "Regime",
    "ShapeComparison",
    "ShapeFit",
    "SlitPore",
    "SquarePore",
    "StatisticalInefficiency",
    "TrianglePore",
    "Unit",
    "__version__",
    "compare_shapes",
    "density_profile",
    "fit_blocks",
    "load_trajectory",
    "statistical_inefficiency",
]
