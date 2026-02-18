```@raw html
---
layout: home

hero:
  name: RadialBasisFunctions.jl
  text: Differential Operators on Scattered Data
  tagline: Build interpolators, Laplacians, and gradients directly from point clouds — no mesh required
  actions:
    - theme: brand
      text: Get Started
      link: /getting_started
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaMeshless/RadialBasisFunctions.jl

features:
  - icon: 📐
    title: No Mesh Required
    details: Interpolate and differentiate directly on scattered point clouds. Local stencils from k-nearest neighbors scale to large problems.
  - icon: ∇
    title: Differential Operators
    details: "Laplacian, gradient, partials, directional derivatives, and custom operators — with operator algebra to combine them."
  - icon: 🚀
    title: GPU Ready
    details: Weight computation parallelizes over stencils via KernelAbstractions.jl. Same code runs on CPU and GPU.
  - icon: 🔬
    title: Fully Differentiable
    details: Native AD rules for Enzyme.jl and Mooncake.jl — differentiate through operators, interpolators, and weight construction.
---
```

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
