# Automatic Differentiation Implementation

This document provides a detailed explanation of how reverse-mode automatic differentiation (AD) is implemented in RadialBasisFunctions.jl through package extensions.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [ChainRulesCore Extension](#chainrulescore-extension)
3. [Mooncake Extension](#mooncake-extension)
4. [Mathematical Background](#mathematical-background)
5. [Implementation Details](#implementation-details)
6. [Usage Examples](#usage-examples)

---

## Architecture Overview

RadialBasisFunctions.jl supports automatic differentiation through two package extensions:

1. **`RadialBasisFunctionsChainRulesCoreExt`**: Provides custom reverse-mode AD rules using ChainRulesCore.jl
2. **`RadialBasisFunctionsMooncakeExt`**: Bridges ChainRulesCore rules to Mooncake.jl's AD system

### Extension Loading

Extensions are loaded automatically when the required packages are available:

```julia
using RadialBasisFunctions
using ChainRulesCore  # Loads RadialBasisFunctionsChainRulesCoreExt
using Mooncake        # Loads RadialBasisFunctionsMooncakeExt (also requires ChainRulesCore)
```

### Design Philosophy

The AD implementation leverages **analytical derivatives** already implemented in the package (via `∇`, `∂`, and `∇²` methods) rather than relying on AD to trace through the computational graph. This approach:

- **Improves performance**: Avoids tracing through complex operations
- **Increases accuracy**: Uses exact mathematical formulas
- **Simplifies debugging**: Derivatives are explicit and testable
- **Enables GPU support**: Works seamlessly with KernelAbstractions.jl

---

## ChainRulesCore Extension

The ChainRulesCore extension (`ext/RadialBasisFunctionsChainRulesCoreExt/`) provides custom `rrule` definitions for key operations in the package.

### File Structure

```
ext/RadialBasisFunctionsChainRulesCoreExt/
├── RadialBasisFunctionsChainRulesCoreExt.jl  # Main extension module
├── basis_rules.jl                             # RBF evaluation derivatives
├── operator_rules.jl                          # Operator application derivatives
├── interpolation_rules.jl                     # Interpolator derivatives
├── build_weights_rrule.jl                     # Weight construction rrules
├── build_weights_backward.jl                  # Backward pass implementations
├── build_weights_cache.jl                     # Forward pass caching structures
└── operator_second_derivatives.jl             # Second-order derivatives
```

### Supported Operations

#### 1. Basis Function Evaluation (`basis_rules.jl`)

**Forward operation**: `φ = basis(x, xi)` computes the RBF value at distance `||x - xi||`

**Backward operation** (rrule):
```julia
∂φ/∂x = Δy * ∇φ(x, xi)
∂φ/∂xi = -Δy * ∇φ(x, xi)
```

**Supported basis types**:
- `PHS1`, `PHS3`, `PHS5`, `PHS7` (Polyharmonic Splines)
- `IMQ` (Inverse Multiquadric)
- `Gaussian`

**Key insight**: Since φ depends on `x - xi`, the gradient w.r.t. `xi` is the negative of the gradient w.r.t. `x`.

**Implementation example** (PHS3):
```julia
function ChainRulesCore.rrule(basis::PHS3, x::AbstractVector, xi::AbstractVector)
    y = basis(x, xi)

    function phs3_pullback(Δy)
        Δy_real = unthunk(Δy)
        grad_fn = ∇(basis)
        ∇φ = grad_fn(x, xi)
        Δx = Δy_real .* ∇φ
        Δxi = -Δx
        return NoTangent(), Δx, Δxi
    end

    return y, phs3_pullback
end
```

#### 2. Operator Application (`operator_rules.jl`)

**Forward operation**: `y = op(x)` applies RBF operator to field values, computing `y = W * x`

**Backward operation**:
```julia
∂L/∂x = W' * ∂L/∂y
```

**Key insight**: The weights matrix `W` is treated as a constant (depends on point geometry, not field values). The pullback is a simple matrix-vector transpose multiplication.

**Scalar operators** (Laplacian, Partial):
```julia
function ChainRulesCore.rrule(::typeof(_eval_op), op::RadialBasisOperator, x::AbstractVector)
    y = _eval_op(op, x)

    function _eval_op_pullback(Δy)
        Δy_unthunked = unthunk(Δy)
        Δx = op.weights' * Δy_unthunked
        return NoTangent(), NoTangent(), Δx
    end

    return y, _eval_op_pullback
end
```

**Vector-valued operators** (Gradient, Jacobian):
```julia
# y[:,d] = W[d] * x for each dimension d
# Pullback: Δx = Σ_d W[d]' * Δy[:,d]
function ChainRulesCore.rrule(
    ::typeof(_eval_op),
    op::RadialBasisOperator{<:VectorValuedOperator{D}},
    x::AbstractVector
) where {D}
    y = _eval_op(op, x)

    function _eval_op_vector_pullback(Δy)
        Δy_unthunked = unthunk(Δy)
        Δx = similar(x)
        fill!(Δx, zero(eltype(Δx)))
        for d in 1:D
            Δx .+= op.weights[d]' * view(Δy_unthunked, :, d)
        end
        return NoTangent(), NoTangent(), Δx
    end

    return y, _eval_op_vector_pullback
end
```

#### 3. Interpolator Evaluation (`interpolation_rules.jl`)

**Forward operation**:
```julia
f(x) = Σᵢ wᵢ φ(x, xᵢ) + Σⱼ wⱼ pⱼ(x)
```

**Backward operation**:
```julia
∂f/∂x = Σᵢ wᵢ ∇φ(x, xᵢ) + Σⱼ wⱼ ∇pⱼ(x)
```

**Key insight**: Weights and data points are constants; only the evaluation point `x` is differentiated.

**Implementation**:
```julia
function ChainRulesCore.rrule(interp::Interpolator, x::AbstractVector)
    y = interp(x)

    function interpolator_pullback(Δy)
        Δy_real = unthunk(Δy)
        Δx = zero(x)

        # RBF contribution
        grad_fn = ∇(interp.rbf_basis)
        for i in eachindex(interp.rbf_weights)
            ∇φ = grad_fn(x, interp.x[i])
            Δx = Δx .+ (interp.rbf_weights[i] * Δy_real) .* ∇φ
        end

        # Polynomial contribution
        if !isempty(interp.monomial_weights)
            ∇mon = ∇(interp.monomial_basis)
            ∇p = zeros(eltype(x), n_terms, dim)
            ∇mon(∇p, x)

            for j in eachindex(interp.monomial_weights)
                Δx = Δx .+ (interp.monomial_weights[j] * Δy_real) .* view(∇p, j, :)
            end
        end

        return NoTangent(), Δx
    end

    return y, interpolator_pullback
end
```

#### 4. Weight Construction (`build_weights_*.jl`)

**Purpose**: Enables **shape optimization** - differentiating through operator construction w.r.t. point positions.

**Forward operation**: `W = _build_weights(ℒ, data, eval_points, adjl, basis)`

Constructs sparse weight matrix for RBF operator by solving local systems:
```julia
For each evaluation point xₑ with neighbors {x₁, ..., xₖ}:
  Build collocation matrix A[i,j] = φ(xᵢ, xⱼ)
  Build RHS vector b[i] = ℒφ(xₑ, xᵢ)
  Solve: A λ = b
  Extract weights: w = λ[1:k]
```

**Backward operation**: Computes gradients `∂L/∂data` and `∂L/∂eval_points` given `∂L/∂W`.

This is the **most complex rrule** in the package, involving:
- Forward pass caching of solutions and matrices
- Implicit differentiation through linear solves
- Chain rule through collocation matrix and RHS construction
- Accumulation of gradients across stencils

See [Weight Construction Details](#weight-construction-details) for full explanation.

---

## Mooncake Extension

The Mooncake extension (`ext/RadialBasisFunctionsMooncakeExt/`) enables the use of Mooncake.jl's AD system by importing ChainRulesCore rrules.

### How It Works

Mooncake.jl has a different tangent type system than ChainRulesCore. The extension:

1. **Imports rrules** using `@from_rrule` macro
2. **Handles type conversions** between Mooncake and ChainRulesCore tangent types
3. **Provides custom increment methods** for special types (e.g., `SVector`)

### Custom Tangent Handling

**Problem**: Mooncake represents `SVector{N,T}` tangents as `Tangent{@NamedTuple{data::NTuple{N,T}}}`, while ChainRulesCore returns `Vector{SVector{N,T}}`.

**Solution**: Custom `increment_and_get_rdata!` method:
```julia
function Mooncake.increment_and_get_rdata!(
    f::Vector{<:Mooncake.Tangent}, ::Mooncake.NoRData, t::Vector{SVector{N,T}}
) where {N,T}
    for i in eachindex(f, t)
        old_data = f[i].fields.data
        sv = t[i]
        new_data = ntuple(j -> old_data[j] + sv[j], Val(N))
        f[i] = typeof(f[i])((data=new_data,))
    end
    return Mooncake.NoRData()
end
```

### Imported Rules

The extension explicitly imports rrules for:
- Operator evaluation (`_eval_op`)
- Basis function calls (PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian)
- Interpolator evaluation
- Weight construction (`_build_weights`) for Partial and Laplacian operators

**Example import**:
```julia
# Scalar operator: _eval_op(op, x) -> vector
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_eval_op), RadialBasisOperator, Vector{Float64}}
)

# Partial operator with PHS3
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS3}
)
```

---

## Mathematical Background

### Chain Rule Basics

For a function `f(g(x))`, the chain rule gives:
```
∂L/∂x = (∂L/∂f) * (∂f/∂g) * (∂g/∂x)
```

In reverse-mode AD:
- Forward pass: compute `f` and cache intermediate values
- Backward pass: accumulate gradients using cached values

### RBF Gradient Formulas

**Polyharmonic Spline (PHS) basis**: `φ(r) = rⁿ`

```
∇φ = n * rⁿ⁻² * (x - xi)
```

**Examples**:
- PHS1: `∇φ = (x - xi) / r`
- PHS3: `∇φ = 3r * (x - xi)`
- PHS5: `∇φ = 5r³ * (x - xi)`

**Applied operators** (first derivatives):
- Partial derivative: `∂φ/∂xₐ(x, xi) = n * rⁿ⁻² * (x[d] - xi[d])`
- Laplacian: `∇²φ = n(n+1) * rⁿ⁻²` for PHS

**Second derivatives** (for weight construction backward pass):
- `∂²φ/∂xⱼ∂xₐ` (Hessian of basis function)
- `∂/∂xⱼ [∂φ/∂xₐ]` and `∂/∂xⱼ [∇²φ]` (gradients of applied operators)

See `operator_second_derivatives.jl` for detailed formulas.

### Implicit Differentiation

For weight construction, we solve a linear system:
```
A(data) λ = b(data, eval_points)
w = λ[1:k]
```

**Implicit function theorem** gives:
```
Given: Δw (cotangent of weights)
Compute: Δdata, Δeval_points

Steps:
1. Pad cotangent: Δλ = [Δw; 0] (zeros for monomial part)
2. Solve adjoint: η = A⁻ᵀ Δλ
3. Compute matrix cotangent: ΔA = -η λᵀ
4. Compute RHS cotangent: Δb = η
5. Chain through A construction: ΔA → Δdata
6. Chain through b construction: Δb → Δdata, Δeval_points
```

This approach avoids differentiating through the linear solve itself.

---

## Implementation Details

### Weight Construction Details

The weight construction backward pass is the most sophisticated component. It enables **shape optimization** - optimizing point positions to minimize a loss function.

#### Forward Pass with Caching (`build_weights_cache.jl`)

**Data structures**:
```julia
struct StencilForwardCache{T, M<:AbstractMatrix{T}}
    lambda::M          # Full solution (k+nmon) × num_ops
    A_mat::Matrix{T}   # Collocation matrix (for A⁻ᵀ solve in backward)
    k::Int             # Number of neighbors
    nmon::Int          # Number of monomial basis functions
end

struct WeightsBuildForwardCache{T}
    stencil_caches::Vector{StencilForwardCache{T, Matrix{T}}}
    k::Int
    nmon::Int
    num_ops::Int
end
```

**Forward pass** (`_forward_with_cache`):
```julia
function _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, ℒType)
    # For each evaluation point:
    for eval_idx in 1:N_eval
        neighbors = adjl[eval_idx]
        local_data = [data[i] for i in neighbors]

        # Build collocation matrix A
        A = build_collocation_matrix(local_data, basis, mon)

        # Build RHS vector b
        b = build_rhs(ℒrbf, ℒmon, local_data, eval_point, basis, mon)

        # Solve system
        λ = A \ b

        # Extract weights
        w = λ[1:k, :]

        # Store in sparse matrix
        W[eval_idx, neighbors] = w

        # Cache for backward pass
        cache[eval_idx] = StencilForwardCache(λ, A, k, nmon)
    end

    return W, cache
end
```

**Note**: The collocation matrix is symmetric but **not positive definite** (has zero blocks from polynomial constraints), so Cholesky factorization doesn't work. We store the full matrix for backsolves.

#### Backward Pass (`build_weights_backward.jl`)

**Per-stencil backward pass**:
```julia
function backward_stencil_partial!(Δdata, Δeval_point, Δw, cache, neighbors, eval_point, data, basis, mon, k, dim)
    # 1. Backprop through linear solve
    ΔA, Δb = backward_linear_solve(Δw, cache)

    # 2. Backprop through collocation matrix
    backward_collocation!(Δdata, ΔA, neighbors, data, basis, mon, k)

    # 3. Backprop through RHS
    backward_rhs_partial!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, dim, k)
end
```

**1. Linear Solve Backward**:
```julia
function backward_linear_solve!(ΔA, Δb, Δw, cache)
    # Pad cotangent with zeros for monomial part
    Δλ = [Δw; zeros(nmon, num_ops)]

    # Solve adjoint system: A'η = Δλ
    # (A is symmetric, so A' = A)
    η = cache.A_mat \ Δλ

    # ΔA = -η λᵀ (outer product)
    fill!(ΔA, 0)
    for op_idx in 1:num_ops
        ΔA .-= η[:, op_idx] * cache.lambda[:, op_idx]'
    end

    # Δb = η
    Δb .= η
end
```

**2. Collocation Matrix Backward**:

The collocation matrix has structure:
```
A[i,j] = φ(xi, xj)     for i,j ≤ k       (RBF block)
A[i,k+j] = pⱼ(xi)      for i ≤ k, j > 0  (polynomial block)
```

Gradients:
```julia
function backward_collocation!(Δdata, ΔA, neighbors, data, basis, mon, k)
    grad_φ = ∇(basis)

    # RBF block: symmetric matrix
    for j in 1:k
        xj = data[neighbors[j]]
        for i in 1:(j-1)  # Upper triangle only
            xi = data[neighbors[i]]
            ∇φ_ij = grad_φ(xi, xj)

            # Account for symmetry: ΔA[i,j] and ΔA[j,i]
            scale = ΔA[i,j] + ΔA[j,i]

            # φ depends on xi - xj
            Δdata[neighbors[i]] .+= scale .* ∇φ_ij
            Δdata[neighbors[j]] .-= scale .* ∇φ_ij
        end
    end

    # Polynomial block: A[i, k+j] = pⱼ(xi)
    if nmon > 0
        ∇p = zeros(nmon, dim)
        for i in 1:k
            xi = data[neighbors[i]]
            ∇mon(∇p, xi)  # Compute all monomial gradients

            for j in 1:nmon
                # Account for both A[i, k+j] and A[k+j, i]
                scale = ΔA[i, k+j] + ΔA[k+j, i]
                Δdata[neighbors[i]] .+= scale .* ∇p[j, :]
            end
        end
    end
end
```

**3. RHS Backward (Partial)**:

RHS structure for `∂/∂xₐ` operator:
```
b[i] = ∂φ/∂xₐ(eval_point, xi)  for i = 1:k
b[k+j] = ∂pⱼ/∂xₐ(eval_point)   for j = 1:nmon
```

Need second derivatives:
- `∂/∂eval_point [∂φ/∂xₐ]` → gradient w.r.t. evaluation point
- `∂/∂xi [∂φ/∂xₐ]` → gradient w.r.t. data point

```julia
function backward_rhs_partial!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, dim, k)
    # Get second derivative functions
    grad_Lφ_x = grad_applied_partial_wrt_x(basis, dim)
    grad_Lφ_xi = grad_applied_partial_wrt_xi(basis, dim)

    # RBF section
    for i in 1:k
        xi = data[neighbors[i]]

        # Gradient w.r.t. eval_point
        ∇Lφ_x = grad_Lφ_x(eval_point, xi)
        Δeval_point .+= Δb[i] .* ∇Lφ_x

        # Gradient w.r.t. xi
        ∇Lφ_xi = grad_Lφ_xi(eval_point, xi)
        Δdata[neighbors[i]] .+= Δb[i] .* ∇Lφ_xi
    end

    # Polynomial section: TODO (requires third derivatives of monomials)
end
```

**3. RHS Backward (Laplacian)**:

RHS structure for Laplacian:
```
b[i] = ∇²φ(eval_point, xi)  for i = 1:k
b[k+j] = ∇²pⱼ(eval_point)   for j = 1:nmon
```

Similar to Partial but uses Laplacian second derivatives:
```julia
function backward_rhs_laplacian!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, k)
    grad_Lφ_x = grad_applied_laplacian_wrt_x(basis)
    grad_Lφ_xi = grad_applied_laplacian_wrt_xi(basis)

    for i in 1:k
        xi = data[neighbors[i]]
        ∇Lφ_x = grad_Lφ_x(eval_point, xi)
        ∇Lφ_xi = grad_Lφ_xi(eval_point, xi)

        Δeval_point .+= Δb[i] .* ∇Lφ_x
        Δdata[neighbors[i]] .+= Δb[i] .* ∇Lφ_xi
    end
end
```

#### Second Derivatives (`operator_second_derivatives.jl`)

Implements Hessian-like terms for all supported basis functions:

**PHS3 Partial Derivative Second Derivatives**:
```
∂φ/∂xₐ = 3 * δₐ * r  where δₐ = x[d] - xi[d]

∂²φ/∂xⱼ∂xₐ = {
    3 * (r + δₐ²/r)           if j == d
    3 * δₐ * δⱼ / r           if j ≠ d
}
```

Implementation:
```julia
function grad_partial_phs3_wrt_x(dim::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r_safe = r + AVOID_INF
        δ = x .- xi
        δ_d = δ[dim]

        result = similar(x, eltype(x))
        for j in eachindex(x)
            if j == dim
                result[j] = 3 * (r + δ_d^2 / r_safe)
            else
                result[j] = 3 * δ_d * δ[j] / r_safe
            end
        end
        return result
    end
    return grad_Lφ_x
end
```

By symmetry: `∂/∂xi = -∂/∂x`

**PHS3 Laplacian Second Derivatives**:
```
∇²φ = 12r

∂/∂xⱼ [12r] = 12 * δⱼ / r
```

**Similar formulas** implemented for PHS1, PHS5, PHS7.

#### Main rrule (`build_weights_rrule.jl`)

Ties everything together:

```julia
function ChainRulesCore.rrule(
    ::typeof(_build_weights),
    ℒ::Partial,
    data::AbstractVector,
    eval_points::AbstractVector,
    adjl::AbstractVector,
    basis::AbstractRadialBasis,
)
    # Build monomial basis
    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    # Forward pass with caching
    W, cache = _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, Partial)

    function _build_weights_partial_pullback(ΔW_raw)
        PT = eltype(data)  # Point type (e.g., SVector{2,Float64})
        N_data = length(data)
        N_eval = length(eval_points)

        ΔW = materialize_sparse_tangent(ΔW_raw, W)

        # Initialize gradient accumulators
        Δdata_raw = [zeros(dim) for _ in 1:N_data]
        Δeval_points_raw = [zeros(dim) for _ in 1:N_eval]

        # Process each stencil
        for eval_idx in 1:N_eval
            neighbors = adjl[eval_idx]
            stencil_cache = cache.stencil_caches[eval_idx]

            # Extract cotangent for this stencil
            Δw = extract_stencil_cotangent(ΔW, eval_idx, neighbors, k, num_ops)

            if sum(abs, Δw) > 0  # Skip if zero cotangent
                local_data = [data[i] for i in neighbors]
                Δlocal_data = [zeros(dim) for _ in 1:k]
                Δeval_pt = zeros(dim)

                # Run backward pass
                backward_stencil_partial!(
                    Δlocal_data, Δeval_pt, Δw, stencil_cache,
                    collect(1:k), eval_points[eval_idx], local_data,
                    basis, mon, k, ℒ.dim
                )

                # Accumulate to global gradients
                for (local_idx, global_idx) in enumerate(neighbors)
                    Δdata_raw[global_idx] .+= Δlocal_data[local_idx]
                end
                Δeval_points_raw[eval_idx] .+= Δeval_pt
            end
        end

        # Convert to match input types (required for Mooncake compatibility)
        return (
            NoTangent(),                                      # function
            NoTangent(),                                      # ℒ
            [PT(Δdata_raw[i]) for i in 1:N_data],           # data
            [PT(Δeval_points_raw[i]) for i in 1:N_eval],    # eval_points
            NoTangent(),                                      # adjl
            NoTangent(),                                      # basis
        )
    end

    return W, _build_weights_partial_pullback
end
```

**Key details**:
1. **`materialize_sparse_tangent`**: Handles Mooncake's `Tangent{SparseMatrixCSC}` type
2. **Per-stencil processing**: Accumulates gradients from each local system
3. **Type conversion**: Converts arrays to `SVector` types to match input
4. **Sparse cotangent extraction**: Efficiently extracts relevant cotangent values

---

## Usage Examples

### Example 1: Operator Differentiation with Zygote

```julia
using RadialBasisFunctions
using StaticArrays
using Zygote

# Setup
data = [SVector(x, y) for x in 0:0.1:1 for y in 0:0.1:1]
basis = PHS3(1)
neighbors = find_neighbors(data, data, 15)
∇² = Laplacian(data, data, neighbors, basis)

# Field values
u = rand(length(data))

# Compute Laplacian and gradient w.r.t. field values
loss(u) = sum(abs2, ∇²(u))
grad_u = Zygote.gradient(loss, u)[1]
```

### Example 2: Interpolation with Mooncake

```julia
using RadialBasisFunctions
using StaticArrays
using Mooncake

# Setup interpolator
data = [SVector(randn(), randn()) for _ in 1:100]
values = [sin(norm(x)) for x in data]
interp = Interpolator(data, values, PHS3(1))

# Differentiate interpolation
x_eval = SVector(0.5, 0.5)
loss(x) = interp(x)^2

# Mooncake AD
rule = build_rrule(loss, x_eval)
grad_x = Mooncake.value_and_gradient!!(rule, loss, x_eval)[2]
```

### Example 3: Shape Optimization

**Goal**: Optimize point positions to minimize operator error

```julia
using RadialBasisFunctions
using StaticArrays
using Zygote

# True solution and its Laplacian
u_exact(x) = sin(π * x[1]) * sin(π * x[2])
Δu_exact(x) = -2π^2 * u_exact(x)

# Initial point cloud
data = [SVector(x, y) for x in 0:0.2:1 for y in 0:0.2:1]
N = length(data)

# Loss function: error in Laplacian operator
function shape_loss(data_flat)
    # Reshape flat vector to points
    data = [SVector(data_flat[2i-1], data_flat[2i]) for i in 1:N]

    # Build operator
    neighbors = find_neighbors(data, data, 12)
    ∇² = Laplacian(data, data, neighbors, PHS3(1))

    # Apply to true solution
    u_vals = [u_exact(x) for x in data]
    Δu_computed = ∇²(u_vals)
    Δu_true = [Δu_exact(x) for x in data]

    # Return MSE
    return sum(abs2, Δu_computed - Δu_true) / N
end

# Optimize
data_flat = reduce(vcat, [[x[1], x[2]] for x in data])
grad_data = Zygote.gradient(shape_loss, data_flat)[1]

# Update positions (simple gradient descent)
α = 0.01
data_flat_new = data_flat - α * grad_data
```

**How it works**:
1. `shape_loss` calls `Laplacian(...)`, which internally calls `_build_weights`
2. The `_build_weights` rrule computes gradients w.r.t. `data` positions
3. Zygote propagates gradients through the entire computation
4. Result: gradient of loss w.r.t. point positions

### Example 4: Parameter Sensitivity

```julia
using RadialBasisFunctions
using StaticArrays
using Zygote

# Solve Poisson equation: ∇²u = f
function solve_poisson(data, f_vals, boundary_indices, boundary_vals, basis)
    neighbors = find_neighbors(data, data, 15)
    ∇² = Laplacian(data, data, neighbors, basis)

    # Build system: ∇²u = f with boundary conditions
    N = length(data)
    A = Matrix(∇².weights)
    b = copy(f_vals)

    # Apply boundary conditions
    for (i, val) in zip(boundary_indices, boundary_vals)
        A[i, :] .= 0
        A[i, i] = 1
        b[i] = val
    end

    # Solve
    u = A \ b
    return u
end

# Differentiate solution w.r.t. source term
data = [SVector(x, y) for x in 0:0.1:1 for y in 0:0.1:1]
f_vals = rand(length(data))
boundary_indices = [1, 11, 121]  # Example boundary nodes
boundary_vals = [0.0, 0.0, 0.0]

loss(f) = sum(abs2, solve_poisson(data, f, boundary_indices, boundary_vals, PHS3(1)))
grad_f = Zygote.gradient(loss, f_vals)[1]
```

---

## Performance Considerations

### When to Use Custom Rules

**Benefits**:
- Avoid tracing through complex RBF weight construction
- Exact derivatives (no approximation error)
- GPU compatibility maintained

**Costs**:
- Memory for caching (forward pass stores matrices)
- Additional code complexity

### Memory Usage

For weight construction with `N_eval` stencils of size `k` with `nmon` monomials:
```
Cache memory ≈ N_eval * [(k + nmon)² * sizeof(Float64) + (k + nmon) * num_ops * sizeof(Float64)]
```

For large problems (N_eval > 10,000), consider:
- Checkpointing (recompute instead of cache)
- Sparse storage for collocation matrices
- Limiting polynomial degree

### GPU Support

All rrules are **GPU-compatible**:
- Use `similar(x)` to maintain array type
- Avoid CPU-only operations (e.g., `push!`, `Dict`)
- KernelAbstractions.jl handles parallelization

---

## Future Extensions

### Planned Features

1. **Gradient operator rrule**: Currently only Partial and Laplacian support `_build_weights` rrule
2. **Polynomial third derivatives**: Complete RHS backward pass for polynomial augmentation
3. **IMQ/Gaussian second derivatives**: Extend to non-PHS bases
4. **Checkpointing**: Reduce memory for large-scale shape optimization
5. **Forward-mode rules**: Add `frule` for forward-mode AD systems

### Contributing

To add support for a new operator:

1. **Implement second derivatives** in `operator_second_derivatives.jl`:
   ```julia
   function grad_applied_myoperator_wrt_x(basis::PHS3)
       # Return function computing ∂/∂x [ℒφ(x, xi)]
   end
   ```

2. **Add backward RHS function** in `build_weights_backward.jl`:
   ```julia
   function backward_rhs_myoperator!(Δdata, Δeval_point, Δb, ...)
       # Chain through RHS construction
   end
   ```

3. **Add rrule** in `build_weights_rrule.jl`:
   ```julia
   function ChainRulesCore.rrule(::typeof(_build_weights), ℒ::MyOperator, ...)
       # Use backward_stencil_myoperator!
   end
   ```

4. **Import in Mooncake extension**:
   ```julia
   Mooncake.@from_rrule(
       Mooncake.DefaultCtx,
       Tuple{typeof(_build_weights), MyOperator, ...}
   )
   ```

---

## References

- [ChainRulesCore.jl Documentation](https://juliadiff.org/ChainRulesCore.jl/stable/)
- [Mooncake.jl Documentation](https://github.com/compintell/Mooncake.jl)
- [Implicit Differentiation Tutorial](https://implicit-layers-tutorial.org/)
- Fornberg, B., & Flyer, N. (2015). *A Primer on Radial Basis Functions with Applications to the Geosciences*

---

## Summary

The AD implementation in RadialBasisFunctions.jl demonstrates:

1. **Leveraging analytical derivatives** for performance and accuracy
2. **Implicit differentiation** for differentiating through linear solves
3. **Modular design** separating forward/backward passes and operator logic
4. **Cross-AD compatibility** via ChainRulesCore and Mooncake bridges
5. **GPU readiness** using array-agnostic operations

This enables advanced applications like shape optimization, parameter sensitivity analysis, and neural PDE solvers with RBF layers.
