# Convergence & Parameter Selection

This guide documents how interpolation and differential-operator accuracy depend on
the point cloud, basis function, polynomial augmentation degree, and stencil size —
with convergence and work-precision plots generated from offline sweeps across every
basis (PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian) and polynomial degree combination.

Use it as a reference when picking parameters for your own problems.

## Pages

- **[Scalar Operators](scalar-operators.md)** — convergence for `Interpolator`, `laplacian`, `gradient`, `partial`, and mixed partials.
- **[Vector Operators](vector-operators.md)** — convergence for `hessian`, `jacobian`, `divergence`, `curl`.
- **[Refinement Sweeps](refinement-sweeps.md)** — polynomial degree (p-refinement), stencil size (k-refinement), and shape parameter (ε-refinement) behavior across all basis functions.
- **[Work-Precision](work-precision.md)** — the computational cost of every (basis, polynomial degree) combination: build time, apply time, and memory vs accuracy.
- **[3D Extensions](3d.md)** — convergence plots for 3D problems.

## Notation & definitions

- **h-refinement**: refine the point cloud (increase `N`). In 2D, typical spacing `h ∼ 1/√N`. Primary convergence test.
- **p-refinement**: vary `poly_deg` at fixed `N`.
- **k-refinement**: vary stencil size at fixed `N` and `poly_deg`.
- **ε-refinement**: vary the shape parameter of `IMQ` or `Gaussian` at fixed everything else.
- **NRMSE**: normalized RMSE, `√(Σ(computed − exact)² / Σ exact²)`. Dimensionless; comparable across problems.

## Expected convergence rates

With polynomial augmentation of degree `m` and a sufficiently smooth target, local-stencil
RBF-FD approximations converge at:

- **Interpolation**: `O(h^{m+1})`
- **First derivatives**: `O(h^m)`
- **Second derivatives**: `O(h^{m-1})`

For PHS, the standard rule is `poly_deg = ⌈n/2⌉` for PHS(n) as the minimum that achieves
the expected rate: PHS1 → p=1, PHS3 → p=2, PHS5 → p=3, PHS7 → p=4. Higher `poly_deg` adds
more monomials and can further boost the observed rate up to a point of diminishing returns.

For IMQ and Gaussian, the shape parameter `ε` controls the basis width; smaller `ε` is
flatter (more accurate in principle) but conditioning degrades — see the [ε-refinement section](refinement-sweeps.md#shape-parameter-epsilon-refinement).

## How to read the plots

All convergence plots are log-log NRMSE vs `N` (number of points). Dashed gray lines show
reference slopes `O(h^p)` for visual comparison. Each colored curve is one (basis,
polynomial degree) configuration.

Work-precision plots show NRMSE vs wall-clock time — lower-left is better. A point cloud
that's "cheaper per error" is closer to the bottom-left corner.

!!! note "Unusable combinations are omitted"
    Some `(basis, poly_deg)` pairs do not converge for certain operators — most notably
    PHS1 with any second-derivative operator, and PHS3/p=2 with mixed partials or the
    full Hessian. These combinations are **omitted from plots** rather than shown as
    flat lines, with a warning admonition at the top of each affected operator section
    listing exactly what's excluded and why.

## Methodology

- **Test function for interpolation**: Franke (1979), a standard test with multiple length scales.
- **Test function for differential operators**: `g(x) = 1 + sin(4x₁) + cos(3x₁) + sin(2x₂)` with analytic derivatives.
- **Point cloud**: jittered grid on `[0,1]²`, seed=42 for reproducibility.
- **Sweep**: `n_side ∈ {10, 15, 20, 30, 45, 70}`, giving `N ∈ {100, 225, 400, 900, 2025, 4900}`.
- **Error metric**: NRMSE on the training points (collocation residual) for differential operators; NRMSE on 500 held-out points for interpolation.
- **Regeneration**: all plots come from committed CSVs under `docs/src/assets/convergence/data/`. To refresh, run `julia --project=docs docs/src/assets/convergence/generate_data.jl` followed by `generate_plots.jl`.
