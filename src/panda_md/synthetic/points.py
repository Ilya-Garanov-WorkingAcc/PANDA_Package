"""Синтетическая траектория из облаков точек: форма в щели → частицы и атомы стенок по кадрам.

Частицы жидкости равномерно заполняют трёхмерную форму (`contains`) с плотностью ρ_bulk; ось формы в каждом кадре
сдвинута случайно в плоскости стенок (как блуждает капля или валик в МД), щель пересекает границу ящика по оси
профиля. Это вход для сквозной проверки «форма → траектория → центрирование → профиль → фит» и 2D-карт.
"""

from collections.abc import Mapping

import numpy as np

from panda_md.core.errors import NotDerivedError
from panda_md.core.model import InterfaceModel
from panda_md.pores.slit.pore import SlitPore
from panda_md.trajectory.arrays import ArrayTrajectory

WALL_SPACING_NM = 0.25  # шаг решётки атомов подложки
OVERSAMPLING = 1.3  # кандидатов на одну принятую частицу больше 1/φ — меньше повторных попыток
MAX_ATTEMPTS = 100


def synthetic_trajectory(
    model: InterfaceModel,
    values: Mapping[str, float],
    *,
    density_bulk_per_nm3: float,
    frame_count: int = 10,
    time_step_ps: float = 10.0,
    substrate_nm: float = 1.0,
    seed: int | None = None,
) -> ArrayTrajectory:
    """Траектория щели с формой `model` (единицы пользователя в values): группы «liquid» и «walls».

    Ящик — L × kL × (H + substrate_nm); подложка занимает z ∈ [0, substrate_nm), её образ сверху замыкает щель.
    Число частиц жидкости N = φ ρ_bulk V постоянно; в каждом кадре ось формы — в случайной точке плоскости стенок.
    """
    pore = model.pore
    if not isinstance(pore, SlitPore) or not model.shape.has_contains():
        raise NotDerivedError(f"{model}: облако точек выведено только для форм щели (нужен contains)")
    internal = model.supported_values(values)
    geometry = pore.geometry()
    if not model.shape.feasible(geometry, internal):
        raise NotDerivedError(f"{model}: форма не существует при {dict(values)}")
    generator = np.random.default_rng(seed)
    width = pore.width_nm
    box = np.array([pore.length_nm, pore.base_aspect_ratio * pore.length_nm, width + substrate_nm])
    walls = _substrate(box, substrate_nm)
    volume_fraction = internal["phi"]
    count = round(volume_fraction * density_bulk_per_nm3 * pore.volume_nm3())
    cell = np.array([geometry.pore_length_ratio, geometry.short_side_ratio(), 1.0])
    frames = []
    for _ in range(frame_count):
        points = _sample_inside(model, internal, cell, count, volume_fraction, generator)
        axis = generator.uniform(0.0, 1.0, size=2) * box[:2]
        liquid = points * width + np.array([axis[0], axis[1], substrate_nm + width / 2.0])
        frames.append(np.mod(np.vstack([walls, liquid]), box))
    return ArrayTrajectory(
        positions_nm=np.array(frames),
        boxes_nm=np.tile(box, (frame_count, 1)),
        times_ps=time_step_ps * np.arange(frame_count),
        groups={"walls": np.arange(len(walls)), "liquid": len(walls) + np.arange(count)},
    )


def _substrate(box: np.ndarray, substrate_nm: float) -> np.ndarray:
    """Решётка атомов подложки: слои через WALL_SPACING_NM от z = 0 до z = substrate_nm (обе поверхности)."""
    xs = np.arange(0.0, box[0], WALL_SPACING_NM)
    ys = np.arange(0.0, box[1], WALL_SPACING_NM)
    zs = np.linspace(0.0, substrate_nm, max(2, round(substrate_nm / WALL_SPACING_NM) + 1))
    grid = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    walls: np.ndarray = grid
    return walls


def _sample_inside(
    model: InterfaceModel,
    internal: Mapping[str, float],
    cell: np.ndarray,
    count: int,
    volume_fraction: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """count точек, равномерно распределённых в жидкости (безразмерные, ось формы в начале координат)."""
    geometry = model.pore.geometry()
    accepted: list[np.ndarray] = []
    total = 0
    for _ in range(MAX_ATTEMPTS):
        batch = max(1000, int(OVERSAMPLING * (count - total) / max(volume_fraction, 1e-6)))
        candidates = generator.uniform(-0.5, 0.5, size=(batch, 3)) * cell
        inside = candidates[model.shape.contains(candidates, geometry, dict(internal))]
        accepted.append(inside)
        total += len(inside)
        if total >= count:
            points: np.ndarray = np.vstack(accepted)[:count]
            return points
    raise NotDerivedError(f"{model}: не удалось набрать {count} точек в форме за {MAX_ATTEMPTS} попыток")
