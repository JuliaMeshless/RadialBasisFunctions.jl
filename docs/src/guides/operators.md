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

The weights ``w_i`` are precomputed by solving a local collocation system (see [Radial Basis Functions Theory](@ref) for the full derivation). Once computed, applying the operator is a matrix-vector product with the weight matrix:

```math
\mathcal{L}\mathbf{u} = W \mathbf{u}
```

Because every row of ``W`` has exactly `k` nonzeros (one per stencil neighbor), the weights are stored stencil-wise in ELL format ([`StencilWeights`](@ref)) and the product is evaluated as a row-parallel gather kernel, ``y_i = \sum_l \mathrm{vals}[l,i] \, x_{\mathrm{idx}[l,i]}`` — multithreaded (with SIMD) on CPU and a KernelAbstractions kernel on GPU backends. This is roughly 7.6× faster than the sparse matrix-vector product it replaced (measured at ``N = 10^5``, ``k = 50``, 13 threads).

This is what `RadialBasisOperator` stores and evaluates.

## The Type Hierarchy

All operators inherit from [`AbstractOperator`](@ref), where the parameter `N` is the tensor rank added to the output:

```
AbstractOperator{N}
├── N=0 (rank-preserving)
│   ├── Partial          ∂ⁿf/∂xᵢⁿ
│   ├── MixedPartial     ∂²f/∂xᵢ∂xⱼ
│   ├── Laplacian        ∇²f
│   ├── Directional      ∇f⋅v
│   ├── Divergence       ∇⋅u (vector field → scalar)
│   ├── Curl             ∇×u (vector field → scalar/vector)
│   ├── Identity         f (function itself)
│   ├── ScaledOperator   α * op (algebra result, any rank N)
│   ├── SumOperator      op₁ + op₂ (algebra result, any rank N)
│   ├── StrainRate       ½(∇u + (∇u)ᵀ)
│   ├── RotationRate     ½(∇u − (∇u)ᵀ)
│   ├── Regrid           interpolation to new points
│   ├── VirtualPartial   ∂f/∂xᵢ via offset-point finite differences
│   └── Custom{0}        user-defined
├── N=1 (rank-adding)
│   ├── Jacobian          [∂fᵢ/∂xⱼ]
│   └── Custom{1}         user-defined
└── N=2 (rank+2)
    └── Hessian           [∂²f/∂xᵢ∂xⱼ]
```

## Understanding Rank (`N`)

The parameter `N` captures whether differentiation adds a tensor index to the output.

**`N=0` (rank-preserving):** The output has the same shape as the input. The operator stores a single weight matrix `W`, and evaluation is `W * u`.

```@example operators
x = rand(SVector{2,Float64}, 100)
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
| `N=0` | `(N, D)` | `(N,)` | `divergence` (vector field → scalar) |
| `N=0` | `(N, 2)` | `(N,)` | `curl` in 2D (vector field → scalar) |
| `N=0` | `(N, 3)` | `(N, 3)` | `curl` in 3D (vector field → vector) |
| `N=1` | `(N,)` | `(N, D)` | `jacobian` on scalar field |
| `N=1` | `(N, D)` | `(N, D, D)` | `jacobian` on vector field |
| `N=2` | `(N,)` | `(N, D, D)` | `hessian` on scalar field |

## `RadialBasisOperator`: The Wrapper

[`RadialBasisOperator`](@ref) wraps an operator with everything needed to compute and apply it:

```@example operators
op = laplacian(x)
```

Key fields:

| Field | Description |
|:---|:---|
| `ℒ` | The operator type (e.g., `Laplacian()`) |
| `weights` | Precomputed [`StencilWeights`](@ref) (or tuple for multi-component operators) |
| `data` | Source points used to build stencils |
| `eval_points` | Points where the operator is evaluated |
| `adjl` | Adjacency list (neighbor indices per stencil) |
| `basis` | RBF basis function used |

### Weight Caching and Invalidation

Weights are computed eagerly during construction and cached. If you mutate the underlying data (e.g., move points), invalidate the cache to trigger recomputation on next evaluation:

```@example operators
# Manually invalidate if data changes
RadialBasisFunctions.invalidate_cache!(op)

# Next call recomputes weights automatically
result = op(u)
typeof(result)
```

You can also force an immediate recomputation with `update_weights!`.

### Getting a sparse matrix

`weights(op)` returns the operator's [`StencilWeights`](@ref) — a tuple of them for gradient-family operators. When you need a standard sparse matrix, e.g. for global system assembly or an implicit solve, convert with `sparse(op)` (or `SparseMatrixCSC(op)`); this is the supported path, and the result can go straight into `sparse(op) \ rhs` or any sparse linear-algebra library:

```@example operators
using SparseArrays
A = sparse(op)  # SparseMatrixCSC, ready for global assembly or `A \ rhs`
typeof(A)
```

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
| `∂mixed` | `∂mixed(basis, dim1, dim2)` | `(x, xᵢ) -> scalar` |
| `∇` | `∇(basis)` | `(x, xᵢ) -> vector` |
| `∇²` | `∇²(basis)` | `(x, xᵢ) -> scalar` |
| `H` | `H(basis)` | `(x, xᵢ) -> matrix` |

These functors are the interface between operators and [Custom Operators](@ref "Custom Operators"). See that page for how to use them.

### Two differentiation protocols

The names `∂`, `∂²`, `∇`, `∇²`, `H`, and `∂mixed` deliberately serve **two protocols**, selected by the basis type:

| Basis | What `∂(basis, dim)` returns | How it evaluates |
|:---|:---|:---|
| `AbstractRadialBasis` (PHS, IMQ, Gaussian) | A functor **struct** (`∂`, `∇²`, …, defined in `src/basis/basis.jl`) | `(x, xᵢ) -> scalar` |
| `MonomialBasis` | A **factory function** result: `ℒMonomialBasis` (defined in `src/operators/monomial/monomial.jl`) | in-place `(b, x)` fill of the differentiated monomial vector |

This dual identity is load-bearing: an operator action like `(op::Partial)(basis) = ∂(basis, op.order, op.dim)` is written once, and weight building applies it to the RBF basis for the collocation rows and to the `MonomialBasis` for the polynomial-augmentation rows — each returning the form the assembly kernels expect. When adding a basis or operator, implement both halves; do not try to unify them.

This is also why hand-written operators that *compose* functors with arithmetic need two
methods — see [Dual dispatch for composed functors](@ref) for the user-facing consequence.

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

## Virtual Operators

Virtual operators ([`∂virtual`](@ref)) take a different route to a derivative: instead of
solving for weights that approximate ``\partial/\partial x_i`` directly, they interpolate
the field at points offset by ``\pm\Delta`` along the axis and apply a standard finite
difference formula to those interpolated values. This is useful for schemes that need the
derivative to be consistent with a particular FD stencil.

```@example operators
# Virtual partial derivative in x-direction with spacing Δ=0.01
virtual_dx = ∂virtual(x, 1, 0.01)
result = virtual_dx(u)
size(result)
```

The offset ``\Delta`` is a genuine tuning parameter: too large and the finite difference
truncation error dominates; too small and interpolation error is amplified by the ``1/\Delta``
factor.

## Next Steps

- [Building PDE Operators](@ref) — compose operators with [`@operator`](@ref)
- [Custom Operators](@ref "Custom Operators") — the closure escape hatch for anything the macro can't express
