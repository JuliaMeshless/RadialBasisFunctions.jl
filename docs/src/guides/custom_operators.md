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

## Example: Identity Operator

The simplest custom operator just evaluates the basis function itself — equivalent to [`regrid`](@ref):

```@example custom
op = custom(x, basis -> (x, xc) -> basis(x, xc); rank=0)
result = op(u)
typeof(result)
```

## Example: Reproducing a Built-in

Use the `∂` functor to build a first partial derivative manually:

```@example custom
# Custom ∂f/∂x₁ using the ∂ functor
custom_∂x = custom(x, basis -> ∂(basis, 1); rank=0)

# Compare with built-in
builtin_∂x = partial(x, 1, 1)

maximum(abs, custom_∂x(u) .- builtin_∂x(u))
```

The `∂` functor returned by `∂(basis, dim)` is already a callable `(x, xᵢ) -> scalar`, so it can be passed directly.

## Composing Functors

When your operator function returns a single functor directly (like `basis -> ∂(basis, 1)` above), both the RBF and monomial paths are handled automatically. But when you **compose multiple functors with arithmetic** inside the lambda, you need two methods — one for the RBF basis and one for `MonomialBasis`.

**Why:** The system calls `ℒ` with both the RBF basis (e.g., `PHS(3)`) and a [`MonomialBasis`](@ref) (for polynomial augmentation). RBF functors like `∇²(basis)` return `(x, xᵢ) -> scalar`, but monomial functors return `(b, x) -> nothing` (in-place buffer fill). When you compose functors with arithmetic — e.g., `∇²(basis)(x, xᵢ) + k² * basis(x, xᵢ)` — the monomial path fails because you can't do arithmetic on `nothing`.

**The fix:** Define a function with two methods using Julia's multiple dispatch. The monomial path uses the allocating form `functor(x)` (returns a vector) instead of the in-place form:

```@example custom
using RadialBasisFunctions: MonomialBasis  # hide

k² = 4.0

# Two-method operator function
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

helm = custom(x, helmholtz_op; rank=0)

# Verify against separate built-in operators
expected = laplacian(x)(u) .+ k² .* u
maximum(abs, helm(u) .- expected)
```

This follows the same pattern used internally by [operator algebra](@ref "Combining Operators") (see `operator_algebra.jl`).

!!! note
    Simple cases that return a single functor directly — like `basis -> ∂(basis, 1)` — don't need dual dispatch. The built-in functors already handle both basis types internally. Two methods are only needed when you compose multiple functors with arithmetic.

## Example: PDE Operators

For more worked examples using this dual-dispatch pattern — Helmholtz, anisotropic diffusion,
advection-diffusion — see the [PDE Operators Cookbook](@ref).

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

## Combining Operators

Operator algebra is the practical way to compose operators rather than function composition:

```@example custom
# ∇²f + ∂f/∂x₁ as a combined operator
combined = Laplacian() + Partial(1, 1)
op = RadialBasisOperator(combined, x)

# Equivalent to computing separately and adding
lap_result = laplacian(x)(u)
∂x_result = partial(x, 1, 1)(u)

maximum(abs, op(u) .- (lap_result .+ ∂x_result))
```

Subtraction works too:

```@example custom
diff_op = Partial(2, 1) - Partial(2, 2)
op = RadialBasisOperator(diff_op, x)
result = op(u)
typeof(result)
```

Both operands must share the same rank `N`.

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
