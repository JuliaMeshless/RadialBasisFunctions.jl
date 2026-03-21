# PDE Operators Cookbook

Recipes for assembling PDE-specific differential operators via [`@operator`](@ref) and [`custom`](@ref), producing a
**single weight matrix** that applies the full PDE operator in one matrix-vector multiply.

```@example pde
using RadialBasisFunctions
using StaticArrays

x = [SVector{2}(rand(2)) for _ in 1:100]
f(p) = sin(p[1]) * cos(p[2])
u = f.(x);
```

## Helmholtz Operator

The Helmholtz equation ``\nabla^2 f + k^2 f = 0`` appears in acoustics, electromagnetics,
and quantum mechanics. The operator combines a Laplacian with a scaled identity.

```@example pde
k² = 4.0

op = @operator ∇² + k² * f
helm_op = custom(x, op)

# Verify against separate built-in operators
expected = laplacian(x)(u) .+ k² .* u
maximum(abs, helm_op(u) .- expected)
```

## Diffusion — Textbook Notation

The diffusion operator ``\nabla \cdot (\kappa \nabla f)`` appears in heat conduction, mass transfer,
and many other physical models. The `@operator` macro recognizes the textbook form directly:

```@example pde
κ = [2.0, 0.5]

op = @operator ∇ ⋅ (κ * ∇)
diff_op = custom(x, op)

# Verify against separate built-in operators
expected = κ[1] .* partial(x, 2, 1)(u) .+ κ[2] .* partial(x, 2, 2)(u)
maximum(abs, diff_op(u) .- expected)
```

Scalar ``\kappa`` produces an isotropic operator (scaled Laplacian):

```@example pde
op = @operator ∇ ⋅ (3.0 * ∇)
diff_iso = custom(x, op)  # equivalent to 3∇²f
expected = 3.0 .* laplacian(x)(u)
maximum(abs, diff_iso(u) .- expected)
```

## Anisotropic Diffusion — Explicit Partials

The same anisotropic diffusion can also be written with explicit per-dimension coefficients:

```@example pde
κ_x = 2.0
κ_y = 0.5

op = @operator κ_x * ∂²(1) + κ_y * ∂²(2)
aniso_op = custom(x, op)

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

op = @operator ν * ∇² - c[1] * ∂(1) - c[2] * ∂(2)
advdiff_op = custom(x, op)

# Verify against separate built-in operators
expected = ν .* laplacian(x)(u) .- c[1] .* partial(x, 1, 1)(u) .- c[2] .* partial(x, 1, 2)(u)
maximum(abs, advdiff_op(u) .- expected)
```

## Sharing Stencils

When multiple `custom` operators act on the same point set, precompute the neighbor list
once and pass it to avoid redundant nearest-neighbor searches:

```@example pde
adjl = find_neighbors(x, 30)

op = @operator ∇² + k² * f
helm_op  = custom(x, op; adjl=adjl)

op = @operator κ_x * ∂²(1) + κ_y * ∂²(2)
aniso_op = custom(x, op; adjl=adjl)
```
