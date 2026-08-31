# Building PDE Operators

The [`@operator`](@ref) macro lets you write PDE operators in mathematical notation. It
translates symbolic expressions into composable operator objects, producing a **single
weight matrix** that applies the full PDE operator in one matrix-vector multiply.

This is the recommended way to build operators the built-ins don't cover. If your operator
can't be expressed here, see [Custom Operators](@ref "Custom Operators") for the closure
escape hatch.

```@example pde
using RadialBasisFunctions
using StaticArrays

x = rand(SVector{2,Float64}, 100)
f(p) = sin(p[1]) * cos(p[2])
u = f.(x)
nothing # hide
```

## Recognized symbols

| Symbol | Meaning |
|:-------|:--------|
| `∇²`, `Δ` | [`Laplacian`](@ref RadialBasisFunctions.Laplacian) |
| `∂(dim)` | First partial derivative in dimension `dim` |
| `∂²(dim)` | Second partial derivative in dimension `dim` |
| `∇ ⋅ (κ * ∇)` | Diffusion operator (scalar or vector `κ`) |
| `c ⋅ ∇` | Advection operator (vector `c`) |
| `f`, `I` | [`Identity`](@ref) operator |
| Everything else | Scalar coefficient |

Standard arithmetic (`+`, `-`, `*`) and unary negation work as expected. Scalars can be
literals, variables, or expressions like `k^2` or `c[1]`.

The macro produces rank-0 operators — see [Understanding Rank (`N`)](@ref) for what that
means, and [Custom Operators](@ref "Custom Operators") if you need rank 1.

## Helmholtz Operator

The Helmholtz equation ``\nabla^2 f + k^2 f = 0`` appears in acoustics, electromagnetics,
and quantum mechanics. The operator combines a Laplacian with a scaled identity.

```@example pde
k² = 4.0

op = @operator ∇² + k² * f
helm_op = op(x)

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
diff_op = op(x)

# Verify against separate built-in operators
expected = κ[1] .* partial(x, 2, 1)(u) .+ κ[2] .* partial(x, 2, 2)(u)
maximum(abs, diff_op(u) .- expected)
```

Scalar ``\kappa`` produces an isotropic operator (scaled Laplacian):

```@example pde
op = @operator ∇ ⋅ (3.0 * ∇)
diff_iso = op(x)  # equivalent to 3∇²f
expected = 3.0 .* laplacian(x)(u)
maximum(abs, diff_iso(u) .- expected)
```

## Anisotropic Diffusion — Explicit Partials

The same anisotropic diffusion can also be written with explicit per-dimension coefficients:

```@example pde
κ_x = 2.0
κ_y = 0.5

op = @operator κ_x * ∂²(1) + κ_y * ∂²(2)
aniso_op = op(x)

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

op = @operator ν * ∇² - c ⋅ ∇
advdiff_op = op(x)

# Verify against separate built-in operators
expected = ν .* laplacian(x)(u) .- c[1] .* partial(x, 1, 1)(u) .- c[2] .* partial(x, 1, 2)(u)
maximum(abs, advdiff_op(u) .- expected)
```

## Sharing Stencils

When multiple operators act on the same point set, precompute the neighbor list
once and pass it to avoid redundant nearest-neighbor searches:

```@example pde
adjl = find_neighbors(x, 30)

helm_op  = (@operator ∇² + k² * f)(x; adjl=adjl)
aniso_op = (@operator κ_x * ∂²(1) + κ_y * ∂²(2))(x; adjl=adjl)
```
