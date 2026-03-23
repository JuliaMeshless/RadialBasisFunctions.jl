# Operators & Type Hierarchy

Operators are the core abstraction in RadialBasisFunctions.jl for RBF-FD differentiation on scattered data. This page explains the operator type system, rank semantics, and how everything fits together.

For basic usage, see [Getting Started](@ref). For the underlying math, see [Radial Basis Functions Theory](@ref).

```@example operators
using RadialBasisFunctions
using StaticArrays
```

## Math Refresher

An RBF-FD operator ``\mathcal{L}`` approximates a differential operator at a point ``x_c`` using a weighted sum over its local stencil:

```math
\mathcal{L}u(x_c) \approx \sum_i w_i \, u(x_i)
```

The weights ``w_i`` are precomputed by solving a local collocation system (see [Radial Basis Functions Theory](@ref) for the full derivation). Once computed, applying the operator is just a sparse matrix-vector multiply:

```math
\mathcal{L}\mathbf{u} = W \mathbf{u}
```

This is what `RadialBasisOperator` stores and evaluates.

## The Type Hierarchy

All operators inherit from [`AbstractOperator`](@ref), where the parameter `N` is the tensor rank added to the output:

```
AbstractOperator{N}
├── N=0 (rank-preserving)
│   ├── Partial          ∂ⁿf/∂xᵢⁿ
│   ├── Laplacian        ∇²f
│   ├── Directional      ∇f⋅v
│   ├── Identity         f (function itself)
│   ├── ScaledOperator   α * op
│   ├── Regrid           interpolation to new points
│   └── Custom{0}        user-defined / algebra result
└── N=1 (rank-adding)
    ├── Jacobian          [∂fᵢ/∂xⱼ]
    └── Custom{1}         user-defined
```

## Understanding Rank (`N`)

The parameter `N` captures whether differentiation adds a tensor index to the output.

**`N=0` (rank-preserving):** The output has the same shape as the input. The operator stores a single weight matrix `W`, and evaluation is `W * u`.

```@example operators
x = [SVector{2}(rand(2)) for _ in 1:100]
u = sin.(getindex.(x, 1))
nothing # hide

lap = laplacian(x)
result = lap(u)
size(result)  # (100,) — same shape as input
```

**`N=1` (rank-adding):** The output gains a trailing dimension of size `D` (spatial dimension). The operator stores a tuple of `D` weight matrices `(W₁, W₂, …, W_D)`, one per spatial dimension.

```@example operators
jac = jacobian(x)
result = jac(u)
size(result)  # (100, 2) — trailing dimension added
```

When a rank-1 operator is applied to a vector field (matrix input), the output gains yet another dimension:

```@example operators
v = hcat(sin.(getindex.(x, 1)), cos.(getindex.(x, 2)))  # (100, 2) vector field
result = jac(v)
size(result)  # (100, 2, 2) — full Jacobian tensor
```

### Input/Output Shape Summary

| Operator rank | Input shape | Output shape | Example |
|:---:|:---:|:---:|:---|
| `N=0` | `(N,)` | `(N,)` | `laplacian`, `partial` |
| `N=0` | `(N, D)` | `(N, D)` | `laplacian` on vector field |
| `N=1` | `(N,)` | `(N, D)` | `jacobian` on scalar field |
| `N=1` | `(N, D)` | `(N, D, D)` | `jacobian` on vector field |

## `RadialBasisOperator`: The Wrapper

[`RadialBasisOperator`](@ref) wraps an operator with everything needed to compute and apply it:

```@example operators
op = laplacian(x)
```

Key fields:

| Field | Description |
|:---|:---|
| `ℒ` | The operator type (e.g., `Laplacian()`) |
| `weights` | Precomputed weight matrix (or tuple for rank-1) |
| `data` | Source points used to build stencils |
| `eval_points` | Points where the operator is evaluated |
| `adjl` | Adjacency list (neighbor indices per stencil) |
| `basis` | RBF basis function used |

### Lazy Evaluation and Caching

Weights are computed eagerly during construction and cached. If you mutate the underlying data (e.g., move points), invalidate the cache to trigger recomputation on next evaluation:

```@example operators
# Manually invalidate if data changes
RadialBasisFunctions.invalidate_cache!(op)

# Next call recomputes weights automatically
result = op(u)
typeof(result)
```

You can also force an immediate recomputation with `update_weights!`.

## Basis Derivative Functors

When you call an operator type on a basis, it returns a **functor** — a callable struct that evaluates the differentiated basis function at two points `(x, xᵢ)`. These functors are the building blocks for both built-in and custom operators.

```@example operators
basis = PHS(3; poly_deg=2)

# Laplacian() applied to a basis returns a ∇² functor
lap_functor = Laplacian()(basis)
typeof(lap_functor)
```

```@example operators
# Partial(1, 1) applied to a basis returns a ∂ functor
partial_functor = Partial(1, 1)(basis)
typeof(partial_functor)
```

These functors are callable as `(x, xᵢ) -> scalar`:

```@example operators
x1 = SVector(0.5, 0.3)
x2 = SVector(0.1, 0.2)

# Evaluate ∇²ϕ(‖x₁ - x₂‖)
lap_functor(x1, x2)
```

The Jacobian operator returns a **tuple** of functors (one per spatial dimension):

```@example operators
jac_functors = Jacobian{2}()(basis)
typeof(jac_functors)
```

Available functor types (accessed via `RadialBasisFunctions.∂` etc.):

| Functor | Constructor | Signature |
|:---|:---|:---|
| `∂` | `∂(basis, dim)` | `(x, xᵢ) -> scalar` |
| `∂²` | `∂²(basis, dim)` | `(x, xᵢ) -> scalar` |
| `∇²` | `∇²(basis)` | `(x, xᵢ) -> scalar` |
| `D` | `D(basis, v)` | `(x, xᵢ) -> scalar` |
| `H` | `H(basis)` | `(x, xᵢ) -> matrix` |

These functors are the interface between operators and the [`custom`](@ref) function. See [Custom Operators](@ref "Custom Operators") for how to use them.

## Operator Algebra

Built `RadialBasisOperator`s can be combined with `+` and `-`. This operates on precomputed weights and returns a new operator:

```@example operators
∂x = partial(x, 1, 1)
∂y = partial(x, 1, 2)

combined = ∂x + ∂y  # ∂f/∂x + ∂f/∂y
result = combined(u)
typeof(result)
```

Both operands must share the same data, stencils, and rank `N`.

See [Custom Operators](@ref "Custom Operators") for more on building your own operators.
