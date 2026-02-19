```@raw html
---
layout: home

hero:
  name: RadialBasisFunctions.jl
  text: Meshless Computing in Julia
  tagline: Radial basis functions for operators, machine learning, and beyond.
  image:
    src: /assets/logo.svg
    alt: RadialBasisFunctions.jl
  actions:
    - theme: brand
      text: Get Started
      link: /getting_started
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaMeshless/RadialBasisFunctions.jl

features:
  - icon: '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"><circle cx="3" cy="7" r="1.5" fill="#7c6cf0"/><circle cx="10" cy="3" r="1.5" fill="#7c6cf0"/><circle cx="19" cy="6" r="1.5" fill="#7c6cf0"/><circle cx="7" cy="14" r="1.5" fill="#7c6cf0"/><circle cx="15" cy="12" r="1.5" fill="#7c6cf0"/><circle cx="4" cy="21" r="1.5" fill="#7c6cf0"/><circle cx="20" cy="19" r="1.5" fill="#7c6cf0"/></svg>'
    title: Scattered Data as a First Class Citizen
    details: Interpolate and differentiate directly on scattered point clouds. Local stencils from k-nearest neighbors scale to large problems.
  - icon: ℒ
    title: API for Operators
    details: "Laplacian, gradient, partials, directional derivatives, and custom operators — with operator algebra to combine them."
  - icon: ∂
    title: Fully Differentiable
    details: Native AD rules for Enzyme.jl and Mooncake.jl — differentiate through operators, interpolators, and weight construction.
---
```

## Quick Start

```julia
using RadialBasisFunctions, StaticArrays

# Scattered data
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
