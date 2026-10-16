"""Fit a synthetic density profile: the shortest path from a model to a contact angle.

A roll (cylindrical-segment liquid bridge) in an 8 × 16 nm slit pore is turned into the density profile an MD
simulation would give (bin-averaged, Poisson noise), and the profile is fitted back.

    python examples/01_synthetic_fit.py
"""

import panda_md as panda
from panda_md.core.enums import NoiseModel, ProfileSampling
from panda_md.synthetic.profiles import synthetic_profile

DENSITY_BULK = 30.0  # particles per nm³ in the bulk liquid
TRUTH = {"theta": 120.0, "delta": 0.03, "phi": 0.4}  # contact angle (°), wall layer (in H), volume fraction


def main() -> None:
    pore = panda.SlitPore(width_nm=8.0, length_nm=16.0)
    roll = pore.interface("roll")
    profile = synthetic_profile(
        roll,
        TRUTH,
        density_bulk_per_nm3=DENSITY_BULK,
        noise_std=0.005,
        seed=0,
        noise_model=NoiseModel.POISSON,
        sampling=ProfileSampling.BIN_AVERAGE,
    )
    result = roll.fit(profile, density_bulk_per_nm3=DENSITY_BULK)  # θ and δ free, φ from ∫ρ
    print(result)
    print(f"θ: true {TRUTH['theta']:g}°, fitted {result.values['theta']:.2f}°")
    print(panda.compare_shapes(pore, profile, panda.FitSpec(density_bulk_per_nm3=DENSITY_BULK)))


if __name__ == "__main__":
    main()
