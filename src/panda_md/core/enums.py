"""Enumerations of all fixed choices. The string values are used in the CLI and JSON."""

from enum import Enum


class ParameterName(str, Enum):
    """Names of the standard model parameters (keys in optimize / fixed / bounds)."""

    CONTACT_ANGLE = "theta"
    WALL_LAYER_THICKNESS = "delta"
    VOLUME_FRACTION = "phi"
    PROFILE_SHIFT = "shift"


class Unit(str, Enum):
    """Units of a parameter: what the user gives it in."""

    DEGREE = "degree"  # radians inside the formulas
    PORE_SIZE = "pore_size"  # fractions of the characteristic pore size: H for the slit, a for polygons
    DIMENSIONLESS = "dimensionless"


class Regime(str, Enum):
    """Wetting regime of the analysed liquid."""

    WETTING = "wetting"  # θ < 90°
    NON_WETTING = "non_wetting"  # θ > 90°


class Frame(str, Enum):
    """Frame of reference of the profile coordinate."""

    PORE = "pore"  # 0 — pore centre (centroid for polygons)
    LIQUID = "liquid"  # 0 — centre of the analysed liquid (mean over the selected atoms)


class Normalization(str, Enum):
    """How the data are brought to a dimensionless density."""

    AUTO = "auto"  # BULK if the bulk density is given, otherwise MEAN
    BULK = "bulk"  # ρ / ρ_bulk versus ρ_model
    MEAN = "mean"  # ρ / n̄ versus ρ_model / φ, n̄ = N / V


class LossFunction(str, Enum):
    L1 = "l1"  # mean absolute deviation (PANDA papers)
    L2 = "l2"  # mean squared deviation


class OptimizerMethod(str, Enum):
    """Local method of scipy.optimize.minimize. Nelder–Mead by default: the L1 loss is non-smooth,
    gradient-based L-BFGS-B gets stuck on synthetic data (comparison — tests/README.md, section 'Optimizer')."""

    NELDER_MEAD = "Nelder-Mead"
    POWELL = "Powell"
    LBFGSB = "L-BFGS-B"


class Axis(str, Enum):
    """Axis of the rectangular simulation box."""

    X = "x"
    Y = "y"
    Z = "z"

    @property
    def column(self) -> int:
        """Column index of the coordinate in the position array (x — 0, y — 1, z — 2)."""
        return AXIS_ORDER.index(self)


AXIS_ORDER = (Axis.X, Axis.Y, Axis.Z)


class ProfileCentering(str, Enum):
    """What is placed at 0 of the profile coordinate in each frame."""

    PORE = "pore"  # midpoint between the walls (or the given pore centre) — PORE frame
    LIQUID = "liquid"  # centre of the selected atoms (unweighted) with PBC taken into account — LIQUID frame


class ParticleGrouping(str, Enum):
    """What counts as a profile particle."""

    ATOM = "atom"  # every selected atom
    MOLECULE = "molecule"  # centre of mass of the molecule (less molecular 'ripple' near the walls)


class Orientation(str, Enum):
    """Whether to reflect the profile z → −z in the pore frame."""

    AUTO = "auto"  # reflect if the liquid is at the upper wall (first moment of the mean profile > 0)
    KEEP = "keep"  # as in the trajectory


class TimeUnit(str, Enum):
    """Time units in strings like '2 ns'; everything is in picoseconds internally."""

    FEMTOSECOND = "fs"
    PICOSECOND = "ps"
    NANOSECOND = "ns"
    MICROSECOND = "us"

    @property
    def picoseconds(self) -> float:
        return TIME_UNIT_PICOSECONDS[self]


TIME_UNIT_PICOSECONDS = {
    TimeUnit.FEMTOSECOND: 1e-3,
    TimeUnit.PICOSECOND: 1.0,
    TimeUnit.NANOSECOND: 1e3,
    TimeUnit.MICROSECOND: 1e6,
}


class AxisOrientation(str, Enum):
    """Direction of the shape's symmetry axis relative to the walls."""

    VERTICAL = (
        "vertical"  # perpendicular to the walls (doughnut, perforation): centre searched along both in-plane axes
    )
    HORIZONTAL = "horizontal"  # along the short side of the cell (roll): centre is searched only across the axis


class AxisLocator(str, Enum):
    """How to find the axis position in a frame."""

    LIQUID_CENTER = "liquid_center"  # liquid centre (circular mean with PBC taken into account)
    VOID_CENTER = "void_center"  # void centre: the liquid fills everything except the hole — liquid centre + L/2


class NoiseModel(str, Enum):
    """Noise of a synthetic profile."""

    CONSTANT = "constant"  # σ = noise_std · ρ_bulk in every bin; negative values are clipped to 0
    POISSON = "poisson"  # σ ∝ √ρ, as for the particle count in a bin; no noise in empty bins


class ProfileSampling(str, Enum):
    """How a synthetic profile represents a bin."""

    CENTER = "center"  # model value at the bin centre
    BIN_AVERAGE = "bin_average"  # model average over the bin — like an MD histogram
