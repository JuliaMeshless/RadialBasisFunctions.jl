# Custom Operators

Define your own differential operators when the built-ins (`partial`, `laplacian`, `jacobian`, `directional`) don't cover your use case.

Prerequisite: [Operators & Type Hierarchy](@ref) explains `AbstractOperator{N}`, rank semantics, and basis derivative functors.

```@example custom
using RadialBasisFunctions
using RadialBasisFunctions: ∂, ∂², ∇²
using StaticArrays

x = [SVector{2}(rand(2)) for _ in 1:100]
f(p) = sin(p[1]) * cos(p[2])
u = f.(x);
```

## `@operator` Macro (Recommended)

The [`@operator`](@ref) macro lets you write PDE operators in mathematical notation. It translates symbolic expressions into composable operator objects:

```@example custom
k² = 4.0
op = @operator ∇² + k² * f
helm = custom(x, op)

# Verify against separate built-in operators
expected = laplacian(x)(u) .+ k² .* u
maximum(abs, helm(u) .- expected)
```

### Recognized symbols

| Symbol | Meaning |
|:-------|:--------|
| `∇²`, `Δ` | [`Laplacian`](@ref) |
| `∂(dim)` | First partial derivative in dimension `dim` |
| `∂²(dim)` | Second partial derivative in dimension `dim` |
| `∇ ⋅ (κ * ∇)` | Diffusion operator (scalar or vector `κ`) |
| `f`, `I` | [`Identity`](@ref) operator |
| Everything else | Scalar coefficient |

Standard arithmetic (`+`, `-`, `*`) and unary negation work as expected. Scalars can be literals, variables, or expressions like `k^2` or `c[1]`.

### More examples

```@example custom
# Anisotropic diffusion: κx ∂²f/∂x² + κy ∂²f/∂y²
κx = 2.0; κy = 0.5
op = @operator κx * ∂²(1) + κy * ∂²(2)
aniso = custom(x, op)

expected = κx .* partial(x, 2, 1)(u) .+ κy .* partial(x, 2, 2)(u)
maximum(abs, aniso(u) .- expected)
```

```@example custom
# Anisotropic diffusion: ∇⋅(κ∇f) using textbook notation
κ = [2.0, 0.5]
op = @operator ∇ ⋅ (κ * ∇)
diff = custom(x, op)

expected = κ[1] .* partial(x, 2, 1)(u) .+ κ[2] .* partial(x, 2, 2)(u)
maximum(abs, diff(u) .- expected)
```

```@example custom
# Advection-diffusion: ν∇²f - c⋅∇f
ν = 0.01; c = SVector(1.0, 0.5)
op = @operator ν * ∇² - c[1] * ∂(1) - c[2] * ∂(2)
advdiff = custom(x, op)

expected = ν .* laplacian(x)(u) .- c[1] .* partial(x, 1, 1)(u) .- c[2] .* partial(x, 1, 2)(u)
maximum(abs, advdiff(u) .- expected)
```

## Understanding Rank

The rank is auto-inferred — you don't need to specify it. This table explains what each rank means:

| Rank 0 (scalar output) | Rank 1 (vector output) |
|:---|:---|
| Output has same shape as input | Output gains a spatial dimension |
| Single weight matrix `W` | Tuple of `D` weight matrices |
| Laplacian, partial derivative, directional derivative | Gradient, Jacobian, Hessian-like operators |

For `AbstractOperator` inputs (from `@operator` or algebra), the rank is encoded in the type parameter. For `Function` closures, it's inferred by probing: a tuple return means rank 1, a single callable means rank 0. You can still pass `rank` explicitly as an override if needed.

## Hermite Boundary Conditions

Custom operators support Hermite interpolation via the `hermite` keyword, just like built-in operators:

```julia
op = custom(data, my_ℒ; hermite=(
    is_boundary=is_boundary,
    bc=bcs,
    normals=normals
))
```

See the [Boundary Conditions](@ref "Getting Started") section of Getting Started for details on Hermite interpolation.

## The Contract

The [`custom`](@ref) function builds a `RadialBasisOperator` from a user-defined operator function:

```julia
custom(data, ℒ)
```

The function `ℒ` must follow a three-layer structure:

1. **`ℒ` receives the basis instance** — e.g., `PHS(3; poly_deg=2)`
2. **Returns a callable `(x, xᵢ) -> value`** — this evaluates the operator applied to the basis function
3. **The value is ``\mathcal{L}[\phi(\|x - x_i\|)]``** — the operator acting on the basis function centered at ``x_i``

This callable fills the right-hand side of the stencil system that determines the weights. For a rank-0 operator it returns a scalar; for rank-1 it returns a tuple of callables (one per spatial dimension).

## Escape Hatch: Function Form

If `@operator` can't express your operator, you can pass a closure directly to `custom()`. This is a last resort — if you find yourself needing this, consider [opening an issue](https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues) so macro support can be added.

### Rank-1 example

For a rank-1 operator (one that adds a trailing dimension), return a **tuple** of callables. The `@operator` macro currently only produces rank-0 operators, so rank-1 requires the function form:

```@example custom
# Custom gradient: tuple of ∂/∂x₁ and ∂/∂x₂
custom_grad = custom(x, basis -> (∂(basis, 1), ∂(basis, 2)))

# Compare with built-in jacobian
builtin_jac = jacobian(x)
maximum(abs, custom_grad(u) .- builtin_jac(u))
```

Each element of the tuple produces one column of the output matrix.

### Dual dispatch for composed functors

When you **compose multiple functors with arithmetic** inside a lambda, you need two methods — one for the RBF basis and one for `MonomialBasis`. The `@operator` macro handles this automatically, which is why it's preferred.

**Why dual dispatch is needed:** The system calls `ℒ` with both the RBF basis (e.g., `PHS(3)`) and a [`MonomialBasis`](@ref) (for polynomial augmentation). RBF functors like `∇²(basis)` return `(x, xᵢ) -> scalar`, but monomial functors return `(b, x) -> nothing` (in-place buffer fill). Arithmetic on `nothing` fails.

```@example custom
using RadialBasisFunctions: MonomialBasis  # hide

# Two-method operator function (advanced — prefer @operator for this)
function helmholtz_op(basis)
    lap = ∇²(basis)
    (x, xc) -> lap(x, xc) + k² * basis(x, xc)
end
function helmholtz_op(basis::MonomialBasis)
    lap = ∇²(basis)
    function (b, x)
        b .= lap(x) .+ k² .* basis(x)
        return nothing
    end
end

helm3 = custom(x, helmholtz_op)
maximum(abs, helm3(u) .- helm(u))
```

!!! note
    Simple cases that return a single functor directly — like `basis -> ∂(basis, 1)` — don't need dual dispatch. The built-in functors already handle both basis types internally. Two methods are only needed when you compose multiple functors with arithmetic.

## Example: PDE Operators

For more worked examples — Helmholtz, anisotropic diffusion, advection-diffusion — see the [PDE Operators Cookbook](@ref).
