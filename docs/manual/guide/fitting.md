# Fitting

```python
result = pore.interface("roll").fit(
    profile,
    optimize=["theta", "delta"],          # free parameters (default: the shape's own list, usually θ and δ)
    fixed={"phi": 0.45},                   # fixed values in user units
    bounds={"theta": (120.0, 180.0)},      # narrower bounds for free parameters
    density_bulk_per_nm3=33.1,             # bulk density of the analysed liquid
)
```

All settings form one `FitSpec`; it is stored in the result and is what the command line and `panda run` build.

## Rules for parameters

- every parameter is either free (`optimize`) or fixed; a parameter in both is an error;
- a fixed value comes from `fixed`, else from the data (φ = ∫ρ / ρ_bulk), else from the parameter's default;
- `bounds` narrow the bounds of free parameters only; `initial` adds a starting point;
- θ is given and returned in degrees, δ and `shift` in units of the pore size.

## Normalization

| `normalization` | Data | Model | Needs |
|-----------------|------|-------|-------|
| `bulk` | ρ / ρ_bulk | ρ_model | `density_bulk_per_nm3` |
| `mean` | ρ / n̄, n̄ = N / V | ρ_model / φ | the particle count and pore volume of the profile |
| `auto` (default) | `bulk` if ρ_bulk is given, else `mean` | | |

## Optimizer

The loss is the mean absolute deviation (`l1`, as in the PANDA papers) or the mean squared deviation (`l2`).
Nelder–Mead runs from a grid of starting points (`starts_per_parameter` per free parameter) inside the feasible
domain; infeasible points get a barrier above any feasible loss. Wetting regimes are fitted separately and the
best results of the other regimes are kept in `result.alternatives`.

## Problem flags

`result.problems()` lists what makes the value suspicious; the command line exits with code 2 if it is not empty:

| Flag | Meaning |
|------|---------|
| `success = False` | the optimizer did not converge |
| `feasible = False` | the best point is outside the shape's existence domain |
| `at_bounds` | a free parameter is at its bound — the true value may lie beyond (δ = 0 is a physical bound, not flagged) |
| `at_regime_boundary` | θ is pressed to 90°: the liquid probably wets the walls (no formulas for θ < 90°) |

## Choosing the shape

`panda.compare_shapes(pore, profile, spec)` fits every shape with the same settings and sorts them by loss;
shapes that cannot hold the data (no feasible start, no profile formula) are listed with the reason. Shapes
whose losses differ by less than 2 % are marked with «≈» (`comparison.indistinguishable`) and a warning is
issued: the profile alone cannot tell them apart (a roll and a perforation at the same φ give nearly the same
profile) — check the 2D density map (`density_map`) and the stability diagram before trusting the ranking.

With `normalization="mean"` the volume fraction cannot be taken from the data (that needs ρ_bulk): give
`density_bulk_per_nm3`, fix φ or free it.
