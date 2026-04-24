# Convergence & Parameter Selection

This guide documents how interpolation and differential-operator accuracy depend on
the point cloud, basis function, polynomial augmentation degree, and stencil size —
with convergence and work-precision plots generated from offline sweeps across every
basis (PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian) and polynomial degree combination.

Use it as a reference when picking parameters for your own problems.

## Pages

- **[Scalar Operators](scalar-operators.md)** — convergence for `Interpolator`, `laplacian`, `gradient`, `partial`, and mixed partials (PHS only).
- **[Vector Operators](vector-operators.md)** — convergence for `hessian`, `jacobian`, `divergence`, `curl` (PHS only).
- **[Shape-Parameter Bases](shape-parameter-bases.md)** — h-refinement for `IMQ` and `Gaussian`: why fixed-ε sweeps diverge at large `N` (the RBF uncertainty principle) and how stationary scaling `ε = c/h` restores clean convergence.
- **[Work-Precision](work-precision.md)** — the computational cost of every (basis, polynomial degree) combination: build time, apply time, and memory vs accuracy.

The same convergence story transfers to 3D (stencil neighborhoods become spheres instead of disks, polynomial basis gains more monomials — `binomial(3+m, m)` vs `binomial(2+m, m)` — so 3D stencils should be larger). All plots below are 2D for clarity.

## Notation & definitions

- **h-refinement**: refine the point cloud (increase `N`). In 2D, typical spacing `h ∼ 1/√N`; in 3D, `h ∼ 1/∛N`. Primary convergence test.
- **p-refinement**: vary `poly_deg` at fixed `N`.
- **k-refinement**: vary stencil size at fixed `N` and `poly_deg`.
- **ε-refinement**: vary the shape parameter of `IMQ` or `Gaussian` at fixed everything else.
- **NRMSE**: normalized RMSE, `√(Σ(computed − exact)² / Σ exact²)`. Dimensionless; comparable across problems.

## Background: RBF-FD and polynomial augmentation

A radial basis function (RBF) approximation expresses an unknown function `u(x)` as a
linear combination of radial kernels centered at the data points, optionally augmented
with a polynomial term to guarantee well-posedness and polynomial reproduction:

```math
s(x) = \sum_{j=1}^{k} w_j \, \phi(\|x - x_j\|) + \sum_{|\alpha| \le m} c_\alpha x^\alpha
```

Weights `wⱼ` and polynomial coefficients `c_α` are fixed by collocation plus moment
conditions. For **global** interpolation (used by `Interpolator`) the sum runs over all
`N` data points. For **local-stencil RBF-FD** (used by `laplacian`, `gradient`, etc.) the
sum runs over the `k` nearest neighbors of each evaluation point, producing sparse weight
matrices [1, 2, 3].

The package supports three RBF families:

- **Polyharmonic splines (PHS)** — `ϕ(r) = r^n` for odd `n`; scale-free, no shape
  parameter to tune. Not strictly positive definite, so polynomial augmentation of
  sufficient degree is **required** for the linear system to be solvable [4].
- **Inverse multiquadric (IMQ)** — `ϕ(r) = 1/√(1+(εr)²)`; strictly positive definite,
  with a shape parameter `ε` controlling basis width [5].
- **Gaussian** — `ϕ(r) = exp(-(εr)²)`; also strictly positive definite, same `ε`
  interpretation.

## Expected convergence rates

For smooth target functions and sufficiently regular point distributions, RBF-FD with
polynomial augmentation of degree `m` converges as follows in 2D [1, 6]:

- **Interpolation**: `O(h^{m+1})`
- **First derivatives** (gradient, partials, divergence, curl, Jacobian): `O(h^m)`
- **Second derivatives** (Laplacian, pure partials, mixed partials, Hessian): `O(h^{m-1})`

In 2D the observed convergence in `N` is `O(N^{-p/2})` for rate `p` since `h ∼ 1/√N`;
in 3D it is `O(N^{-p/3})`.

### Matched polynomial degree for PHS

Flyer, Fornberg, Bayona, and Barnett [1] established that for polyharmonic splines of
order `n`, the **minimum** polynomial degree needed to realize the expected convergence
rate is `m = ⌈n/2⌉`:

| Basis | Matched `poly_deg` | Interp. rate | 1st-deriv rate | 2nd-deriv rate |
|---|---|---|---|---|
| PHS1 (`r`)    | 1 | `O(h²)` | `O(h)`  | `O(h⁰)` (non-convergent) |
| PHS3 (`r³`)   | 2 | `O(h³)` | `O(h²)` | `O(h)` |
| PHS5 (`r⁵`)   | 3 | `O(h⁴)` | `O(h³)` | `O(h²)` |
| PHS7 (`r⁷`)   | 4 | `O(h⁵)` | `O(h⁴)` | `O(h³)` |

Because PHS1/p=1 predicts `O(h⁰)` for second derivatives, this combination is formally
non-convergent for the Laplacian, Hessian, and second partials. In practice it is also
numerically unstable because PHS1's second derivative is singular at `r=0`, so the local
systems are ill-conditioned — see the [Laplacian section](scalar-operators.md#laplacian)
for how badly this manifests.

Higher `poly_deg` than the matched minimum can improve the observed rate on sufficiently
smooth targets (superconvergence) [6, 7], but the gain saturates once the stencil size
becomes the limiting factor.

### Stencil size and the "matched-degree + a few extra points" rule

Local stencils need at least as many points as monomials in the polynomial basis. For
polynomial degree `m` in `d` dimensions the polynomial block has `binomial(d+m, m)` terms,
so `k ≥ binomial(d+m, m) + δ` with `δ ≈ 5–15` is a common rule of thumb [1, 2]. The
`autoselect_k(data, basis)` helper applies this automatically. Empirically, `k` exhibits
a U-shape — too few neighbors under-resolves the local polynomial; too many introduces
extraneous distant points that hurt local fit quality.

### Shape-parameter bases and the stagnation / ill-conditioning tradeoff

For IMQ and Gaussian, `ε` controls the kernel width. In the **flat limit** `ε → 0` the
interpolant converges to a polynomial of degree determined by the point cloud and
approaches spectral accuracy, but the collocation matrix becomes arbitrarily
ill-conditioned [8, 9]. Conversely, `ε → ∞` (very peaked kernels) produces nearly
diagonal, well-conditioned systems but poor interpolation. The result is a
**stagnation error** curve: accuracy improves with decreasing `ε` until conditioning
takes over, producing a characteristic U-shape or V-shape — see
[Shape-Parameter Bases](shape-parameter-bases.md) for the h-refinement consequence
of this and how scaled-ε restores convergence.

Stable algorithms such as RBF-QR [9] partially circumvent this limit but are not
currently used by this package.

## How to read the plots

All h-refinement plots are log-log NRMSE vs `N` (number of points). Dashed gray lines
show reference slopes `O(h^p)` for visual comparison. Each colored curve is one
(basis, polynomial degree) configuration.

Work-precision plots show NRMSE vs wall-clock time — lower-left is better. A
configuration that's "cheaper per error" is closer to the bottom-left corner.

!!! note "Unusable combinations are omitted"
    Some `(basis, poly_deg)` pairs do not converge for certain operators — most notably
    PHS1 with any second-derivative operator, and PHS3/p=2 with mixed partials or the
    full Hessian. These combinations are **omitted from plots** rather than shown as
    flat lines, with a warning admonition at the top of each affected operator section
    listing exactly what's excluded and why.

## Methodology

- **Test function for interpolation**: Franke's function [10], a standard benchmark with
  multiple length scales and both positive and negative features:
  ```math
  F(x,y) = \tfrac{3}{4}e^{-\frac{(9x-2)^2}{4} - \frac{(9y-2)^2}{4}} + \tfrac{3}{4}e^{-\frac{(9x+1)^2}{49} - \frac{9y+1}{10}} + \tfrac{1}{2}e^{-\frac{(9x-7)^2}{4} - \frac{(9y-3)^2}{4}} - \tfrac{1}{5}e^{-(9x-4)^2 - (9y-7)^2}
  ```
  Franke's function is the de facto scattered-data interpolation benchmark in the RBF
  literature [10, 11].
- **Test function for differential operators**: `g(x) = 1 + sin(4x₁) + cos(3x₁) + sin(2x₂) + sin(x₁ + x₂)`,
  with analytic gradient, Hessian, and Laplacian. The `sin(x₁+x₂)` term ensures nonzero
  mixed partials so Frobenius NRMSE on the Hessian remains well-defined.
- **Test vector field** (for `jacobian`, `divergence`, `curl`): `u(x) = [sin(πx₁) + ½ cos(2πx₂), exp(x₁x₂)]`
  with analytic partials, divergence, and curl.
- **Point cloud**: jittered grid on `[0,1]²`, seed=42 for reproducibility. Jittering
  breaks grid alignment to approximate truly scattered points while keeping density
  uniform — a standard protocol in RBF-FD benchmarks [1, 6].
- **Sweep**: `n_side ∈ {10, 15, 20, 30, 45, 70}`, giving `N ∈ {100, 225, 400, 900, 2025, 4900}`.
- **Error metric**: NRMSE on the training points (collocation residual) for differential
  operators; NRMSE on 500 held-out points for interpolation.
- **Regeneration**: all plots come from committed CSVs under
  `docs/src/assets/convergence/data/`. To refresh, run
  `julia --project=docs docs/src/assets/convergence/generate_data.jl` followed by
  `generate_plots.jl`.

## References

1. **Flyer, N., Fornberg, B., Bayona, V., and Barnett, G. A.** (2016). *On the role of
   polynomials in RBF-FD approximations: I. Interpolation and accuracy.* Journal of
   Computational Physics, **321**, 21–38. DOI: [10.1016/j.jcp.2016.05.026](https://doi.org/10.1016/j.jcp.2016.05.026).
   *Establishes the matched-degree rule `m = ⌈n/2⌉` for polyharmonic splines.*
2. **Bayona, V., Flyer, N., Fornberg, B., and Barnett, G. A.** (2017). *On the role of
   polynomials in RBF-FD approximations: II. Numerical solution of elliptic PDEs.*
   Journal of Computational Physics, **332**, 257–273. DOI:
   [10.1016/j.jcp.2016.12.008](https://doi.org/10.1016/j.jcp.2016.12.008).
3. **Wright, G. B. and Fornberg, B.** (2006). *Scattered node compact finite
   difference-type formulas generated from radial basis functions.* Journal of
   Computational Physics, **212**(1), 99–123. DOI:
   [10.1016/j.jcp.2005.05.030](https://doi.org/10.1016/j.jcp.2005.05.030).
   *Foundational RBF-FD paper introducing local-stencil weight generation.*
4. **Iske, A.** (2004). *Multiresolution Methods in Scattered Data Modelling.*
   Lecture Notes in Computational Science and Engineering, vol. 37, Springer.
   *Conditional positive definiteness and polynomial augmentation for PHS.*
5. **Hardy, R. L.** (1971). *Multiquadric equations of topography and other irregular
   surfaces.* Journal of Geophysical Research, **76**(8), 1905–1915. DOI:
   [10.1029/JB076i008p01905](https://doi.org/10.1029/JB076i008p01905).
   *Introduces the multiquadric family; IMQ is its reciprocal variant.*
6. **Bayona, V.** (2019). *An insight into RBF-FD approximations augmented with
   polynomials.* Computers & Mathematics with Applications, **77**(9), 2337–2353. DOI:
   [10.1016/j.camwa.2018.12.029](https://doi.org/10.1016/j.camwa.2018.12.029).
   *Detailed rate analysis; discusses superconvergence beyond the matched degree.*
7. **Fornberg, B. and Flyer, N.** (2015). *A Primer on Radial Basis Functions with
   Applications to the Geosciences.* CBMS-NSF Regional Conference Series in Applied
   Mathematics, SIAM. ISBN: 978-1-611974-02-7.
   *Monograph covering PHS, shape-parameter bases, and RBF-FD.*
8. **Fornberg, B., Wright, G., and Larsson, E.** (2004). *Some observations regarding
   interpolants in the limit of flat radial basis functions.* Computers & Mathematics
   with Applications, **47**(1), 37–55. DOI:
   [10.1016/S0898-1221(04)90004-1](https://doi.org/10.1016/S0898-1221(04)90004-1).
   *Analyzes the flat-limit `ε → 0` and polynomial interpretation.*
9. **Fornberg, B. and Wright, G.** (2004). *Stable computation of multiquadric
   interpolants for all values of the shape parameter.* Computers & Mathematics with
   Applications, **48**(5–6), 853–867. DOI:
   [10.1016/j.camwa.2003.08.010](https://doi.org/10.1016/j.camwa.2003.08.010).
   *Introduces RBF-QR, which sidesteps the conditioning cliff.*
10. **Franke, R.** (1982). *Scattered data interpolation: tests of some methods.*
    Mathematics of Computation, **38**(157), 181–200. DOI:
    [10.2307/2007474](https://doi.org/10.2307/2007474).
    *Original publication of Franke's function and comparison protocol used here.*
11. **Fasshauer, G. E. and McCourt, M. J.** (2015). *Kernel-based Approximation Methods
    using MATLAB.* Interdisciplinary Mathematical Sciences, vol. 19, World Scientific.
    ISBN: 978-981-4630-13-6.
    *Comprehensive reference on kernel methods including RBF benchmark protocols.*
