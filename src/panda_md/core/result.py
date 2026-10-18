"""Fit outcome: parameter values, quality, check flags, data for the plot."""

from panda_md.core.enums import Frame, Normalization, Regime
from panda_md.core.spec import FitSpec
from panda_md.core.types import Array1D, FrozenFloats, PandaModel


class FitResult(PandaModel):
    pore_name: str
    shape_name: str
    regime: Regime | None  # None — the shape has no θ parameter
    frame: Frame
    normalization: Normalization
    values: FrozenFloats  # all parameters, user units (θ in degrees, δ as a fraction of the pore size)
    free_parameters: tuple[str, ...]
    fixed_parameters: tuple[str, ...]
    loss: float
    success: bool
    message: str
    start_count: int
    evaluation_count: int
    feasible: bool
    at_bounds: tuple[str, ...]
    at_regime_boundary: bool
    wall_layer_thickness_nm: float | None
    density_bulk_derived_per_nm3: float | None  # mean normalization: ρ_bulk = n̄ / φ
    volume_fraction_from_data: float | None  # bulk normalization: φ = ∫ρ/ρ_bulk when φ is not given
    coordinate: Array1D  # dimensionless coordinate of the fit window
    data: Array1D  # dimensionless data density
    model_curve: Array1D  # model in the frame of reference of the data
    alternatives: tuple["FitResult", ...] = ()  # best results in the other wetting regimes
    spec: FitSpec
    library_version: str | None = None  # PANDA version that made the fit; None — result from an old JSON

    def summary(self) -> str:
        regime_label = self.regime.value if self.regime is not None else "—"
        lines = [f"{self.pore_name}/{self.shape_name}  regime: {regime_label}  loss: {self.loss:.4g}"]
        for name, value in self.values.items():
            marker = "free" if name in self.free_parameters else "fixed"
            lines.append(f"  {name:>6} = {value:.6g}  ({marker})")
        lines.extend(f"  ! {problem}" for problem in self.problems())
        return "\n".join(lines)

    def problems(self) -> tuple[str, ...]:
        """Check flags: empty — the fit converged inside the existence domain and did not hit the bounds."""
        flags = []
        if not self.success:
            flags.append(f"optimizer: {self.message}")
        if not self.feasible:
            flags.append("outside the existence domain")
        if self.at_bounds:
            flags.append(f"at bound: {', '.join(self.at_bounds)}")
        if self.at_regime_boundary:
            flags.append("θ at 90° (regime boundary)")
        return tuple(flags)

    def __str__(self) -> str:
        return self.summary()
