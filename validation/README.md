# Accuracy of the method

`run_accuracy.py` builds synthetic profiles the way an MD simulation gives them:

- the density in a bin is the mean of the model over the bin (`ProfileSampling.BIN_AVERAGE`);
- the noise is Poisson-like (`NoiseModel.POISSON`): σ ∝ √ρ, given as a fraction of ρ_bulk at ρ = ρ_bulk; empty bins
  stay empty.

The profile is then fitted with the same shape (the model is evaluated at the bin centres). This measures the error
of the method itself — fit, discretization and noise. The error of choosing the shape is not included, so these
numbers are a lower bound of the error on real data. Tables and figures: `REPORT.md` and `figures/`. Rebuild with
`uv run python validation/run_accuracy.py` (about a minute, needs `panda-md[plot]`); the conclusions below are
written by hand and must be checked against the tables after a rerun.

Conditions: an 8 × 16 nm slit, ρ_bulk = 30 nm⁻³, 10 noise seeds per point. Default fit settings: θ and δ free, φ
from ∫ρ. True parameters: roll θ = 120°, δ = 0.03, φ = 0.4; droplet θ = 130°, φ = 0.05; worm θ = 130°, φ = 0.1.

## Conclusions

1. **Noise.** The error of θ grows linearly with the noise. At 0.5 % of ρ_bulk per bin (200 bins) the mean |Δθ| is
   0.21° for the roll, 0.19° for the droplet and 0.17° for the worm, the maximum up to 0.46°. At 2 % the mean error
   is 0.6–0.85°, the maximum up to 1.9°. The criterion “mean |Δθ| < 1°” holds over the whole tested range.
2. **Number of bins.** With the same trajectory statistics (noise scaled as √(bins / 200)), 50 to 400 bins give the
   same error: mean 0.16–0.21°, bias below 0.15°. Discretization barely affects the angle.
3. **Angle.** Up to 140° the mean |Δθ| is at most 0.31°. Towards 180° the profile depends weakly on θ and the error
   grows:
   - droplet: 0.9° at 160°, 2.0° at 170°, 3.7° at 175° (maximum 4.8°);
   - worm: 0.5° at 160°, 1.9° at 170°, 4.1° at 175°; at 175° two fits of ten stop at the bound 180°, so the worm
     error there is a lower estimate;
   - the roll holds longest: 0.4° at 170°, 0.9° at 175°.
   A positive bias appears near 180° (droplet +1.9°, worm +3.0° at 175°). Angles above about 165° for droplets and
   worms come with an error of a degree or more; report them with the spread over time blocks (`fit_blocks`).
4. **Centering.** Wall positions off by 5 % of the slit width:
   - roll, pore centering: 4.7° on average, up to 7°;
   - roll, pore centering with `shift` free: 0.3°;
   - roll, liquid centering with δ known: 2.7°;
   - droplet: 0.2–0.25° with every method — it sits on one wall, and the shift of the walls is absorbed by δ.
   Recommendation: centre on the pore by default; for shapes touching both walls with uncertain wall positions add
   `shift` to the free parameters.

## Limitations

- The synthetic profile and the fit use the same shape model: the error of choosing the shape and the difference
  between a real interface and the model are not included.
- All rows of one study use the same seeds 0…9, so rows compare better than the spread over 10 seeds suggests; the
  maximum over 10 seeds is a noisy estimate, and differences below about 0.1° are not significant.
