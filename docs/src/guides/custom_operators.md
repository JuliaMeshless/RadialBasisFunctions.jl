# Custom Operators

The escape hatch for operators that [`@operator`](@ref) can't express. Reach for
[Building PDE Operators](@ref) first — it covers the common cases with less ceremony.
If you find yourself needing the function form below, consider
[opening an issue](https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues) so
macro support can be added.

Prerequisite: [Operators & Type Hierarchy](@ref) explains `AbstractOperator{N}`, rank
semantics, and basis derivative functors.

```@example custom
using RadialBasisFunctions
using RadialBasisFunctions: ∂, ∇²
using StaticArrays

x = rand(SVector{2,Float64}, 100)
f(p) = sin(p[1]) * cos(p[2])
u = f.(x)
k² = 4.0
nothing # hide
```

## The Contract

Any `AbstractOperator` — including results from `@operator` — can be called directly with
data points to build a `RadialBasisOperator`. Under the hood this calls
`RadialBasisOperator(op, data; kw...)`.

For raw closure-based operators, the [`custom`](@ref) function wraps a function in a
`Custom{N}` operator:

```julia
custom(data, ℒ)
```

The function `ℒ` must follow a three-layer structure:

1. **`ℒ` receives the basis instance** — e.g., `PHS(3; poly_deg=2)`
2. **Returns a callable `(x, xᵢ) -> value`** — this evaluates the operator applied to the basis function
3. **The value is ``\mathcal{L}[\phi(\|x - x_i\|)]``** — the operator acting on the basis function centered at ``x_i``

This callable fills the right-hand side of the stencil system that determines the weights. For a rank-0 operator it returns a scalar; for rank-1 it returns a tuple of callables (one per spatial dimension).

### Rank is inferred, not declared

See [Understanding Rank (`N`)](@ref) for what rank means. You rarely need to state it: for
`AbstractOperator` inputs (from `@operator` or algebra) the rank is encoded in the type
parameter, and for `Function` closures it's inferred by probing — a tuple return means
rank 1, a single callable means rank 0. Pass `rank` explicitly only to override that
inference.

## Function Form

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

When you **compose multiple functors with arithmetic**, you need two methods — one for the RBF basis and one for `MonomialBasis`. The `@operator` macro handles this automatically, which is why it's preferred.

**Why dual dispatch is needed:** this is the user-facing consequence of the
[two differentiation protocols](@ref "Two differentiation protocols"). The system calls
`ℒ` with both the RBF basis (e.g., `PHS(3)`) and a [`MonomialBasis`](@ref) (for polynomial
augmentation). RBF functors like `∇²(basis)` return `(x, xᵢ) -> scalar`, but monomial
functors return `(b, x) -> nothing` (in-place buffer fill). Arithmetic on `nothing` fails.

**Return a callable struct, not a closure.** The package implements every operator action
as a functor — `∇²(basis)`, `SumKernel`, and `ScaledKernel` are all callable structs,
deliberately *"replacing closures with proper multiple dispatch."* Follow that convention:
it keeps the action's return type inferable and the object inspectable, and it stores the
coefficient in a typed field instead of capturing it in a boxed closure. Make the action a
`struct` whose fields are the pieces you compose, with a call method for the arithmetic:

```@example custom
using RadialBasisFunctions: MonomialBasis  # hide

# ℒφ = ∇²φ + k²·φ acting on the RBF basis — a functor, not a closure
struct HelmholtzRBF{L, B, T}
    ∇²φ::L      # the ∇²(basis) functor
    φ::B        # the basis itself
    k²::T
end
(h::HelmholtzRBF)(x, xᵢ) = h.∇²φ(x, xᵢ) + h.k² * h.φ(x, xᵢ)

# the action on the polynomial tail — monomial functors fill a buffer in place
struct HelmholtzPoly{L, B, T} <: Function
    ∇²p::L
    p::B
    k²::T
end
function (h::HelmholtzPoly)(b, x)
    b .= h.∇²p(x) .+ h.k² .* h.p(x)
    return nothing
end

# ℒ: overload one method per basis protocol, each returning its functor
helmholtz_op(basis)                = HelmholtzRBF(∇²(basis), basis, k²)
helmholtz_op(basis::MonomialBasis) = HelmholtzPoly(∇²(basis), basis, k²)

helm3 = Custom{0}(helmholtz_op)(x)

# Verify against the same operator assembled from built-ins
expected = laplacian(x)(u) .+ k² .* u
maximum(abs, helm3(u) .- expected)
```

!!! tip
    For an operator you reach for often, promote it to a first-class
    [`AbstractOperator`](@ref) subtype: hold `k²` in the operator struct and give it
    `(op::Helmholtz)(basis)` methods, exactly like the built-in `Laplacian` and `Partial`.
    Then `Helmholtz(4.0)(points)` builds it with no captured globals at all — and, for this
    particular operator, it is identical to `(@operator ∇² + k² * f)(points)`.

!!! note
    Simple cases that return a single functor directly — like `basis -> ∂(basis, 1)` — don't need dual dispatch. The built-in functors already handle both basis types internally. Two methods are only needed when you compose multiple functors with arithmetic.

## Boundary Conditions on Custom Operators

Custom operators support Hermite interpolation via the `hermite` keyword, just like
built-in operators:

```julia
op = my_ℒ(data; hermite=(is_boundary=is_boundary, bc=bcs, normals=normals))
```

See [Boundary Conditions](@ref) for details.
