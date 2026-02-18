<table align="center">
  <tr>
    <td><img src="docs/src/assets/logo.svg" alt="RadialBasisFunctions.jl" width="300"></td>
    <td>
      <h1>RadialBasisFunctions.jl</h1>
      <p>High-performance radial basis function interpolation<br>and differential operators for Julia.</p>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://JuliaMeshless.github.io/RadialBasisFunctions.jl/stable"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable Docs"></a>
  <a href="https://JuliaMeshless.github.io/RadialBasisFunctions.jl/dev"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev Docs"></a>
  <a href="https://github.com/JuliaMeshless/RadialBasisFunctions.jl/actions/workflows/CI.yml?query=branch%3Amain"><img src="https://github.com/JuliaMeshless/RadialBasisFunctions.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"></a>
  <a href="https://codecov.io/gh/JuliaMeshless/RadialBasisFunctions.jl"><img src="https://codecov.io/gh/JuliaMeshless/RadialBasisFunctions.jl/graph/badge.svg?token=S3BQ5FIULZ" alt="codecov"></a>
  <a href="https://github.com/JuliaMeshless/RadialBasisFunctions.jl/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="https://zenodo.org/badge/latestdoi/634682663"><img src="https://zenodo.org/badge/634682663.svg" alt="DOI"></a>
</p>

---

## Why RadialBasisFunctions.jl?

RBF methods approximate functions from scattered data without requiring a mesh. This package focuses on **local collocation** — building stencils from k-nearest neighbors instead of coupling all N points — so it scales to large problems without dense N×N solves.

Beyond interpolation, it provides differential operators (Laplacian, gradient, partials, custom) that act directly on scattered data, with support for Hermite interpolation to enforce boundary conditions in PDE applications.

Other things that might matter to you:
- GPU-accelerated weight computation via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
- Native autodiff rules for [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) and [Mooncake.jl](https://github.com/compintell/Mooncake.jl) (no generic fallbacks)
- Operator algebra — combine operators with `+`, `-`, `*`

<p align="center">
  <img src="docs/src/assets/interpolation_demo.png" alt="RBF Interpolation Demo" width="800">
</p>

<p align="center"><em>Scattered data points (left) reconstructed as a smooth surface using RBF interpolation (right)</em></p>

## Quick Start

```julia
using RadialBasisFunctions, StaticArrays

# Scattered 2D points — no mesh needed
points = [SVector{2}(rand(2)) for _ in 1:500]
f(x) = sin(4x[1]) * cos(3x[2])
values = f.(points)

# Interpolation
interp = Interpolator(points, values)
interp(SVector(0.5, 0.5))

# Differential operators on scattered data
∇²  = laplacian(points)
∇   = gradient(points)
∂x  = partial(points, 1, 1)       # ∂/∂x₁
∂²y = partial(points, 2, 2)       # ∂²/∂x₂²

∇²(values)                         # apply to data
∇(values)                          # Nx2 matrix

# Combine operators
mixed = ∂x + ∂²y                   # operator algebra

# Transfer data between point sets
target = [SVector{2}(rand(2)) for _ in 1:1000]
rg = regrid(points, target)
rg(values)                         # interpolated onto target
```

## Supported Radial Basis Functions

| Type | Formula | Best For |
|------|---------|----------|
| Polyharmonic Spline (PHS) | $r^n$ where $n = 1, 3, 5, 7$ | General purpose, no shape parameter tuning |
| Inverse Multiquadric (IMQ) | $1 / \sqrt{(\varepsilon r)^2 + 1}$ | Smooth interpolation with tunable accuracy |
| Gaussian | $e^{-(\varepsilon r)^2}$ | Infinitely smooth functions |

## Installation

```julia
using Pkg
Pkg.add("RadialBasisFunctions")
```

Requires Julia 1.10 or later.

> [!TIP]
> The examples above use sensible defaults (`PHS(3)` basis, quadratic polynomial augmentation, auto-selected stencil size). All of these are configurable — see the **[Getting Started guide](https://JuliaMeshless.github.io/RadialBasisFunctions.jl/stable/getting_started)** for details on choosing a basis function, stencil size (`k`), polynomial degree, Hermite boundary conditions, and more.

## Documentation

- **[Getting Started](https://JuliaMeshless.github.io/RadialBasisFunctions.jl/stable/getting_started)** — tutorials covering interpolation, operators, boundary conditions, and GPU usage
- **[Autodiff](https://JuliaMeshless.github.io/RadialBasisFunctions.jl/stable/autodiff)** — differentiating through operators with Enzyme and Mooncake
- **[Theory](https://JuliaMeshless.github.io/RadialBasisFunctions.jl/stable/theory)** — mathematical background on RBF methods
- **[API Reference](https://JuliaMeshless.github.io/RadialBasisFunctions.jl/stable/api)** — full function documentation

## Citation

If you use RadialBasisFunctions.jl in your research, please cite:

```bibtex
@software{RadialBasisFunctions_jl,
  author = {Beggs, Kyle},
  title = {RadialBasisFunctions.jl: Meshless RBF interpolation and differential operators for Julia},
  url = {https://github.com/JuliaMeshless/RadialBasisFunctions.jl},
  doi = {10.5281/zenodo.7941390}
}
```

## Contributing

Contributions are welcome! Please feel free to:

- Report bugs or request features on [GitHub Issues](https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues)
- Start a discussion on [GitHub Discussions](https://github.com/JuliaMeshless/RadialBasisFunctions.jl/discussions)
- Submit pull requests

Part of the [JuliaMeshless](https://github.com/JuliaMeshless) organization.
