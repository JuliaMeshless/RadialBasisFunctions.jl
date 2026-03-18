# Custom Operators

Define your own differential operators when the built-ins (`partial`, `laplacian`, `jacobian`, `directional`) don't cover your use case.

Prerequisite: [Operators & Type Hierarchy](@ref) explains `AbstractOperator{N}`, rank semantics, and basis derivative functors.

```@example custom
using RadialBasisFunctions
using RadialBasisFunctions: ∂, ∂², ∇²
using StaticArrays

x = [SVector{2}(rand(2)) for _ in 1:100]
f(p) = sin(p[1]) * cos(p[2])
u = f.(x)
```

## `@operator` Macro (Recommended)

The [`@operator`](@ref) macro lets you write PDE operators in mathematical notation. It translates symbolic expressions into composable operator objects:

```@example custom
k² = 4.0
helm = custom(x, @operator(∇² + k² * f); rank=0)

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
aniso = custom(x, @operator(κx * ∂²(1) + κy * ∂²(2)); rank=0)

expected = κx .* partial(x, 2, 1)(u) .+ κy .* partial(x, 2, 2)(u)
maximum(abs, aniso(u) .- expected)
```

```@example custom
# Anisotropic diffusion: ∇⋅(κ∇f) using textbook notation
κ = [2.0, 0.5]
diff = custom(x, @operator(∇ ⋅ (κ * ∇)); rank=0)

expected = κ[1] .* partial(x, 2, 1)(u) .+ κ[2] .* partial(x, 2, 2)(u)
maximum(abs, diff(u) .- expected)
```

```@example custom
# Advection-diffusion: ν∇²f - c⋅∇f
ν = 0.01; c = SVector(1.0, 0.5)
advdiff = custom(x, @operator(ν * ∇² - c[1] * ∂(1) - c[2] * ∂(2)); rank=0)

expected = ν .* laplacian(x)(u) .- c[1] .* partial(x, 1, 1)(u) .- c[2] .* partial(x, 1, 2)(u)
maximum(abs, advdiff(u) .- expected)
```

## Operator Algebra (No Macro)

If you prefer explicit construction, use operator types with arithmetic directly:

```@example custom
# Same Helmholtz operator without the macro
helm2 = custom(x, Laplacian() + k² * Identity(); rank=0)

maximum(abs, helm2(u) .- helm(u))
```

Operator algebra supports `+`, `-`, scalar `*`, and unary `-`:

```@example custom
# Combining operators
combined = Laplacian() + Partial(1, 1)
op = RadialBasisOperator(combined, x)

lap_result = laplacian(x)(u)
∂x_result = partial(x, 1, 1)(u)
maximum(abs, op(u) .- (lap_result .+ ∂x_result))
```

```@example custom
# Subtraction and scaling
diff_op = 2.0 * Partial(2, 1) - Partial(2, 2)
op = RadialBasisOperator(diff_op, x)
result = op(u)
typeof(result)
```

Both operands of `+`/`-` must share the same rank `N`.

## The Contract

The [`custom`](@ref) function builds a `RadialBasisOperator` from a user-defined operator function:

```julia
custom(data, ℒ; rank=N)
```

The function `ℒ` must follow a three-layer structure:

1. **`ℒ` receives the basis instance** — e.g., `PHS(3; poly_deg=2)`
2. **Returns a callable `(x, xᵢ) -> value`** — this evaluates the operator applied to the basis function
3. **The value is ``\mathcal{L}[\phi(\|x - x_i\|)]``** — the operator acting on the basis function centered at ``x_i``

This callable fills the right-hand side of the stencil system that determines the weights. For a rank-0 operator it returns a scalar; for rank-1 it returns a tuple of callables (one per spatial dimension).

### Identity operator via function form

```@example custom
op = custom(x, basis -> (x, xc) -> basis(x, xc); rank=0)
result = op(u)
typeof(result)
```

### Reproducing a built-in

```@example custom
# Custom ∂f/∂x₁ using the ∂ functor
custom_∂x = custom(x, basis -> ∂(basis, 1); rank=0)

# Compare with built-in
builtin_∂x = partial(x, 1, 1)
maximum(abs, custom_∂x(u) .- builtin_∂x(u))
```

## Advanced: Composing Functors with Dual Dispatch

When you **compose multiple functors with arithmetic** inside a lambda, you need two methods — one for the RBF basis and one for `MonomialBasis`. For most use cases, prefer `@operator` or operator algebra instead, which handle this automatically.

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

helm3 = custom(x, helmholtz_op; rank=0)
maximum(abs, helm3(u) .- helm(u))
```

!!! note
    Simple cases that return a single functor directly — like `basis -> ∂(basis, 1)` — don't need dual dispatch. The built-in functors already handle both basis types internally. Two methods are only needed when you compose multiple functors with arithmetic.

## Example: PDE Operators

For more worked examples — Helmholtz, anisotropic diffusion, advection-diffusion — see the [PDE Operators Cookbook](@ref).

## Example: Rank-1 Custom Operator

For a rank-1 operator (one that adds a trailing dimension), return a **tuple** of callables:

```@example custom
# Custom gradient: tuple of ∂/∂x₁ and ∂/∂x₂
custom_grad = custom(x, basis -> (∂(basis, 1), ∂(basis, 2)); rank=1)

# Compare with built-in jacobian
builtin_jac = jacobian(x)
maximum(abs, custom_grad(u) .- builtin_jac(u))
```

Each element of the tuple produces one column of the output matrix.

## Choosing `rank=0` vs `rank=1`

| Use `rank=0` when... | Use `rank=1` when... |
|:---|:---|
| Output has same shape as input | Output gains a spatial dimension |
| Single weight matrix `W` | Tuple of `D` weight matrices |
| Laplacian, partial derivative, directional derivative | Gradient, Jacobian, Hessian-like operators |

**Rule of thumb:** if your operator differentiates with respect to all spatial dimensions simultaneously and keeps them separate, use `rank=1`. Otherwise, use `rank=0`.

## Hermite Boundary Conditions

Custom operators support Hermite interpolation via the `hermite` keyword, just like built-in operators:

```julia
op = custom(data, my_ℒ; rank=0, hermite=(
    is_boundary=is_boundary,
    bc=bcs,
    normals=normals
))
```

See the [Boundary Conditions](@ref "Getting Started") section of Getting Started for details on Hermite interpolation.
