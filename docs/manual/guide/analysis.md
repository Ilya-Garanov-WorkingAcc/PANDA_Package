# Analysis and accuracy

## θ ± error over time blocks

```python
blocks = panda.fit_blocks(pore.interface("roll"), profiles, spec)
print(blocks)                       # table of blocks, mean ± error of the mean
blocks.mean("theta"), blocks.sem("theta")
```

`panda.statistical_inefficiency(series)` estimates how many consecutive values are correlated (block-variance
method); a value s means that blocks shorter than s samples underestimate the error.

## Stability diagrams

```python
from panda_md.analysis.stability import stability_diagram

diagram = stability_diagram(
    pore, contact_angle_deg=130.0, wall_layer=0.0,
    volume_fractions=[0.1, 0.2, 0.3], length_ratios=[1.0, 2.0, 3.0],
)
```

For every (L/H, φ) the feasible shape with the smallest capillary energy S_LV − cos θ·S_SL wins. Shapes without
derived surface areas (doughnut and perforation of the slit) do not take part.

## Accuracy of the method

`validation/run_accuracy.py` turns a shape into the profile an MD histogram would give (bin-averaged, Poisson
noise) and fits it back. Summary (8 × 16 nm slit, ρ_bulk = 30 nm⁻³, 10 noise seeds):

| Condition | Mean \|Δθ\| |
|-----------|-------------|
| noise 0.5 % of ρ_bulk per bin | 0.2° (roll, droplet, worm) |
| noise 2 % | 0.6–0.85° |
| 50…400 bins | no effect |
| θ ≤ 140° | ≤ 0.3° |
| θ = 170° / 175° | droplet 2.0° / 3.7°, worm 1.9° / 4.1°, roll 0.4° / 0.9° |
| walls off by 5 % of H (roll) | 4.7°; 0.3° with `shift` free |

The full tables and figures are in `validation/REPORT.md` of the repository. These numbers are a lower bound of
the error on real data: the synthetic profile has exactly the model shape.
