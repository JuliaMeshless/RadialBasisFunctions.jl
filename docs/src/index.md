```@raw html
---
layout: home

hero:
  name: RadialBasisFunctions.jl
  text: Meshless Methods Made Easy
  tagline: RBF interpolation, differential operators, and PDE tools for Julia
  actions:
    - theme: brand
      text: Get Started
      link: /getting_started
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaMeshless/RadialBasisFunctions.jl

features:
  - icon: 📐
    title: Interpolation
    details: Scattered data interpolation with polynomial augmentation
  - icon: ∇
    title: Differential Operators
    details: Convenience constructors for common operators such as partial derivatives, laplacian, and gradient
  - icon: 🚀
    title: GPU Ready
    details: KernelAbstractions.jl enables seamless CPU and GPU execution
  - icon: 🔧
    title: Extensible
    details: Custom operators and Hermite boundary conditions for PDE applications
---
```

## Quick Example

```julia
using RadialBasisFunctions, StaticArrays

# Create scattered data points
x = [SVector{2}(rand(2)) for _ in 1:1000]
y = sin.(norm.(x))

# Build an interpolator
interp = Interpolator(x, y)
interp(SVector{2}(rand(2)))  # Evaluate at points

# Build differential operators
lap = laplacian(x)
grad = gradient(x)
∂x = partial(x, 1, 1)  # ∂/∂x₁
```

## Supported Radial Basis Functions

| Type                 | Function                              |
| -------------------- | ------------------------------------- |
| Polyharmonic Spline  | $r^n$ where $n = 1, 3, 5, 7$          |
| Inverse Multiquadric | $1 / \sqrt{(\varepsilon r)^2 + 1}$    |
| Gaussian             | $e^{-(\varepsilon r)^2}$              |

All basis functions support optional polynomial augmentation for enhanced accuracy near boundaries.

## Installation

```julia
using Pkg
Pkg.add("RadialBasisFunctions")
```

Or in the REPL:

```julia
] add RadialBasisFunctions
```
