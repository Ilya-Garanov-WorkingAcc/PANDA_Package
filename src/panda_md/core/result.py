"""Итог фита: значения параметров, качество, флаги проверок, данные для графика."""

from panda_md.core.enums import Frame, Normalization, Regime
from panda_md.core.spec import FitSpec
from panda_md.core.types import Array1D, FrozenFloats, PandaModel


class FitResult(PandaModel):
    pore_name: str
    shape_name: str
    regime: Regime | None  # None — у формы нет параметра θ
    frame: Frame
    normalization: Normalization
    values: FrozenFloats  # все параметры, единицы пользователя (θ в градусах, δ в долях размера поры)
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
    density_bulk_derived_per_nm3: float | None  # нормировка mean: ρ_bulk = n̄ / φ
    volume_fraction_from_data: float | None  # нормировка bulk: φ = ∫ρ/ρ_bulk, если φ не задана
    coordinate: Array1D  # безразмерная координата окна фита
    data: Array1D  # безразмерная плотность данных
    model_curve: Array1D  # модель в системе отсчёта данных
    alternatives: tuple["FitResult", ...] = ()  # лучшие результаты в других режимах смачивания
    spec: FitSpec

    def summary(self) -> str:
        regime_label = self.regime.value if self.regime is not None else "—"
        lines = [f"{self.pore_name}/{self.shape_name}  режим: {regime_label}  loss: {self.loss:.4g}"]
        for name, value in self.values.items():
            marker = "свободный" if name in self.free_parameters else "фиксирован"
            lines.append(f"  {name:>6} = {value:.6g}  ({marker})")
        flags = []
        if not self.success:
            flags.append(f"оптимизатор: {self.message}")
        if not self.feasible:
            flags.append("результат вне области существования формы")
        if self.at_bounds:
            flags.append(f"у границы: {', '.join(self.at_bounds)}")
        if self.at_regime_boundary:
            flags.append("θ прижат к 90° (граница режимов)")
        lines.extend(f"  ! {flag}" for flag in flags)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
