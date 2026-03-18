# PDE Operators Cookbook

Recipes for assembling PDE-specific differential operators via [`@operator`](@ref) and [`custom`](@ref), producing a
**single weight matrix** that applies the full PDE operator in one matrix-vector multiply.

```@example pde
using RadialBasisFunctions
using StaticArrays

x = [SVector{2}(rand(2)) for _ in 1:100]
f(p) = sin(p[1]) * cos(p[2])
u = f.(x)
```

## Helmholtz Operator

The Helmholtz equation ``\nabla^2 f + k^2 f = 0`` appears in acoustics, electromagnetics,
and quantum mechanics. The operator combines a Laplacian with a scaled identity.

```@example pde
k² = 4.0

helm_op = custom(x, @operator(∇² + k² * f); rank=0)

# Verify against separate built-in operators
expected = laplacian(x)(u) .+ k² .* u
maximum(abs, helm_op(u) .- expected)
```

## Anisotropic Diffusion

Anisotropic diffusion ``\kappa_x \, \partial^2 f / \partial x^2 + \kappa_y \, \partial^2 f / \partial y^2``
models heat conduction in materials with direction-dependent conductivity.

```@example pde
κ_x = 2.0
κ_y = 0.5

aniso_op = custom(x, @operator(κ_x * ∂²(1) + κ_y * ∂²(2)); rank=0)

# Verify against separate built-in operators
expected = κ_x .* partial(x, 2, 1)(u) .+ κ_y .* partial(x, 2, 2)(u)
maximum(abs, aniso_op(u) .- expected)
```

When ``\kappa_x = \kappa_y``, this reduces to a scaled Laplacian.

## Advection-Diffusion

The steady advection-diffusion equation ``\nu \nabla^2 f - \mathbf{c} \cdot \nabla f = 0``
balances viscous diffusion against transport by a velocity field. It appears in fluid dynamics,
pollutant transport, and thermal convection.

```@example pde
ν = 0.01
c = SVector(1.0, 0.5)

advdiff_op = custom(x, @operator(ν * ∇² - c[1] * ∂(1) - c[2] * ∂(2)); rank=0)

# Verify against separate built-in operators
expected = ν .* laplacian(x)(u) .- c[1] .* partial(x, 1, 1)(u) .- c[2] .* partial(x, 1, 2)(u)
maximum(abs, advdiff_op(u) .- expected)
```

## Equivalent Forms

Each recipe above can also be written using explicit operator algebra (no macro):

```@example pde
# Helmholtz without macro
helm_algebra = custom(x, Laplacian() + k² * Identity(); rank=0)

# Anisotropic diffusion without macro
aniso_algebra = custom(x, κ_x * Partial(2, 1) + κ_y * Partial(2, 2); rank=0)

maximum(abs, helm_algebra(u) .- helm_op(u)),
maximum(abs, aniso_algebra(u) .- aniso_op(u))
```

## Sharing Stencils

When multiple `custom` operators act on the same point set, precompute the neighbor list
once and pass it to avoid redundant nearest-neighbor searches:

```@example pde
adjl = find_neighbors(x, 30)

helm_op  = custom(x, @operator(∇² + k² * f); rank=0, adjl=adjl)
aniso_op = custom(x, @operator(κ_x * ∂²(1) + κ_y * ∂²(2)); rank=0, adjl=adjl)
```
