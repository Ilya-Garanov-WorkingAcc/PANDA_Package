"""Library exceptions and warnings. All exceptions inherit from PandaError."""


class PandaError(Exception):
    """Base PANDA exception."""


class NotDerivedError(PandaError):
    """The quantity is not derived for this shape (no formula)."""


class UnknownPoreError(PandaError, KeyError):
    """No pore type is registered under this name."""

    def __str__(self) -> str:  # KeyError wraps the message in quotes
        return str(self.args[0]) if self.args else ""


class UnknownParameterError(PandaError):
    """The parameter name does not belong to the model."""


class UnknownShapeError(PandaError, KeyError):
    """The pore has no shape with this name (inherits KeyError: registry lookup)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class ParameterConflictError(PandaError):
    """A parameter is both free and fixed."""


class InvalidBoundsError(PandaError):
    """Bounds given for the fit do not intersect the parameter's feasible bounds."""


class MissingParameterValueError(PandaError):
    """A fixed parameter has no value."""


class UnidentifiableParameterError(PandaError):
    """A free parameter does not affect the model profile."""


class InfeasibleParametersError(PandaError):
    """No starting point lies in the existence domain of the shape."""


class EmptyWindowError(PandaError):
    """The fit window contains too few profile points."""


class UnsupportedRegimeError(PandaError):
    """The wetting regime is not declared by the shape (formulas are not derived)."""


class UnsupportedFrameError(PandaError):
    """The data are in a frame of reference into which the shape formulas are not converted."""


class InvalidProfileError(PandaError):
    """The density profile is not suitable for analysis."""


class ProfileMismatchError(InvalidProfileError):
    """The profile does not match the pore: different size, different units or a different frame of reference."""


class InvalidTrajectoryError(PandaError):
    """The trajectory or selection is unsuitable: empty selection, no walls, invalid box, empty time interval."""


class MissingDependencyError(PandaError, ImportError):
    """An optional package is required (for example, MDAnalysis from panda-md[md])."""


class PandaWarning(UserWarning):
    """PANDA warning: the computation finished, but the result should not be trusted without a check."""
