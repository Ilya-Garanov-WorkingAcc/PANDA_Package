"""PANDA: Predicting Angle from Nanoscale Density Analysis."""

from panda_md.analysis.blocks import FitResultSet, StatisticalInefficiency, fit_blocks, statistical_inefficiency
from panda_md.analysis.compare import ShapeComparison, ShapeFit, compare_shapes
from panda_md.analysis.stability import StabilityDiagram, stability_diagram
from panda_md.core.centering import SymmetryAxis
from panda_md.core.enums import (
    Axis,
    AxisLocator,
    AxisOrientation,
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
from panda_md.core.errors import (
    InfeasibleParametersError,
    InvalidProfileError,
    InvalidTrajectoryError,
    MissingDependencyError,
    NotDerivedError,
    PandaError,
    PandaWarning,
    ProfileMismatchError,
    UnsupportedRegimeError,
)
from panda_md.core.model import InterfaceModel
from panda_md.core.parameters import Parameter
from panda_md.core.profile import DensityProfile, ProfileSet
from panda_md.core.result import FitResult
from panda_md.core.spec import FitSpec
from panda_md.core.version import library_version
from panda_md.pores.slit import SlitPore
from panda_md.pores.square import SquarePore
from panda_md.pores.triangle import TrianglePore
from panda_md.synthetic.points import synthetic_trajectory
from panda_md.synthetic.profiles import synthetic_profile
from panda_md.trajectory.arrays import ArrayTrajectory
from panda_md.trajectory.density import density_profile
from panda_md.trajectory.maps import DensityMap, density_map
from panda_md.trajectory.reader import load_trajectory

__version__ = library_version()

__all__ = [
    "ArrayTrajectory",
    "Axis",
    "AxisLocator",
    "AxisOrientation",
    "DensityMap",
    "DensityProfile",
    "FitResult",
    "FitResultSet",
    "FitSpec",
    "Frame",
    "InfeasibleParametersError",
    "InterfaceModel",
    "InvalidProfileError",
    "InvalidTrajectoryError",
    "LossFunction",
    "MissingDependencyError",
    "Normalization",
    "NotDerivedError",
    "OptimizerMethod",
    "Orientation",
    "PandaError",
    "PandaWarning",
    "Parameter",
    "ParameterName",
    "ParticleGrouping",
    "ProfileCentering",
    "ProfileMismatchError",
    "ProfileSet",
    "Regime",
    "ShapeComparison",
    "ShapeFit",
    "SlitPore",
    "SquarePore",
    "StabilityDiagram",
    "StatisticalInefficiency",
    "SymmetryAxis",
    "TrianglePore",
    "Unit",
    "UnsupportedRegimeError",
    "__version__",
    "compare_shapes",
    "density_map",
    "density_profile",
    "fit_blocks",
    "load_trajectory",
    "stability_diagram",
    "statistical_inefficiency",
    "synthetic_profile",
    "synthetic_trajectory",
]
