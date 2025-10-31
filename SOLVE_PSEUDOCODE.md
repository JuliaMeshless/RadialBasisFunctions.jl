# Pseudocode: RBF Weight Building System

This document explains the code flow in the solve system, which builds sparse weight matrices for RBF operators.

## Architecture Overview

The solve system is organized into **three distinct layers**:

```
src/solve/
├── types.jl          # Layer 0: Shared data structures & traits
├── stencil_math.jl   # Layer 1: Pure mathematical operations
├── kernel_exec.jl    # Layer 2: Parallel execution & allocation
└── api.jl            # Layer 3: Entry points & routing
```

### Layer Responsibilities

**Layer 0: `types.jl`** - Shared Data Structures
- Boundary condition types (`BoundaryCondition`, `HermiteStencilData`)
- Stencil classification types (`InteriorStencil`, `DirichletStencil`, `HermiteStencil`)
- Operator arity traits (`SingleOperator`, `MultiOperator{N}`)

**Layer 1: `stencil_math.jl`** - Pure Mathematics
- Collocation matrix building (`_build_collocation_matrix!`)
- RHS vector building (`_build_rhs!`)
- Stencil assembly (`_build_stencil!`)
- Hermite boundary modifications (dispatch-based)
- **No I/O, no parallelism, fully testable**

**Layer 2: `kernel_exec.jl`** - Parallel Execution
- Memory allocation (`allocate_sparse_arrays`)
- Kernel launching (`launch_kernel!`)
- Batch processing and synchronization
- Sparse matrix construction

**Layer 3: `api.jl`** - Entry Points
- Public `_build_weights` functions
- Routing (standard vs Hermite paths)
- Operator application to basis functions

---

## System Flow

The system builds sparse weight matrices using exact allocation:
- Interior points get k entries (full stencil)
- Dirichlet boundary points get 1 entry (identity row)
- Other boundary points get k entries (Hermite stencil)

```
User Code
    ↓
api.jl: Route based on boundary conditions
    ↓
kernel_exec.jl: Allocate memory & launch parallel kernel
    ↓
stencil_math.jl: Build collocation matrix A, RHS b, solve A\b
    ↓
kernel_exec.jl: Construct sparse matrix
    ↓
Return to user
```

---

## Part 1: Entry Points (api.jl)

### Main Entry from Operators

```julia
function _build_weights(ℒ, op)
    # Extract configuration from operator
    data = op.data
    eval_points = op.eval_points
    adjl = op.adjl
    basis = op.basis

    return _build_weights(ℒ, data, eval_points, adjl, basis)
end
```

### Apply Operator to Basis

```julia
function _build_weights(ℒ, data, eval_points, adjl, basis)
    dim = length(first(data))

    # Build monomial basis and apply operator
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    return _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
end
```

### Interior-Only Path (No Boundary Conditions)

```julia
function _build_weights(
    data, eval_points, adjl, basis, ℒrbf, ℒmon, mon;
    batch_size=10, device=CPU()
)
    # Create empty boundary data for interior-only case
    TD = eltype(first(data))
    is_boundary = fill(false, length(data))
    boundary_conditions = BoundaryCondition{TD}[]
    normals = similar(data, 0)
    boundary_data = BoundaryData(is_boundary, boundary_conditions, normals)

    return build_weights_kernel(
        data, eval_points, adjl,
        basis, ℒrbf, ℒmon, mon,
        boundary_data;
        batch_size=batch_size, device=device
    )
end
```

### Hermite Path (With Boundary Conditions)

```julia
function _build_weights(
    data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
    is_boundary, boundary_conditions, normals;
    batch_size=10, device=CPU()
)
    boundary_data = BoundaryData(is_boundary, boundary_conditions, normals)
    return build_weights_kernel(
        data, eval_points, adjl,
        basis, ℒrbf, ℒmon, mon,
        boundary_data;
        batch_size=batch_size, device=device
    )
end
```

---

## Part 2: Kernel Orchestration (kernel_exec.jl)

### Main Orchestrator

```julia
function build_weights_kernel(
    data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
    boundary_data; batch_size=10, device=CPU()
)
    # Extract dimensions
    TD = eltype(first(data))
    k = length(first(adjl))
    nmon = binomial(length(first(data)) + basis.poly_deg, basis.poly_deg)
    num_ops = _num_ops(ℒrbf)
    N_eval = length(eval_points)

    # Allocate sparse arrays with exact counting
    I, J, V, row_offsets = allocate_sparse_arrays(
        TD, k, N_eval, num_ops, adjl, boundary_data
    )

    # Launch parallel kernel
    n_batches = ceil(Int, N_eval / batch_size)
    launch_kernel!(
        I, J, V, data, eval_points, adjl,
        basis, ℒrbf, ℒmon, mon, boundary_data, row_offsets,
        batch_size, N_eval, n_batches, k, nmon, num_ops, device
    )

    # Construct sparse matrix
    nrows, ncols = N_eval, length(data)
    if num_ops == 1
        return sparse(I, J, V[:, 1], nrows, ncols)
    else
        return ntuple(i -> sparse(I, J, V[:, i], nrows, ncols), num_ops)
    end
end
```

### Memory Allocation

```julia
function allocate_sparse_arrays(TD, k, N_eval, num_ops, adjl, boundary_data)
    # Count exact non-zeros:
    # - Interior points: k entries (full stencil)
    # - Dirichlet points: 1 entry (identity row)
    # - Other boundary points: k entries (Hermite stencil)
    total_nnz, row_offsets = count_nonzeros(
        adjl, boundary_data.is_boundary, boundary_data.boundary_conditions
    )

    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Matrix{TD}(undef, total_nnz, num_ops)

    return I, J, V, row_offsets
end
```

### Kernel Execution

```julia
@kernel function weight_kernel(I, J, V, data, eval_points, adjl, boundary_data, ...)
    batch_idx = @index(Global)
    start_idx, end_idx = calculate_batch_range(batch_idx, batch_size, N_eval)

    # Pre-allocate work arrays (reused within batch)
    A = Symmetric(zeros(TD, n, n), :U)
    b = _prepare_buffer(ℒrbf, TD, n)

    for eval_idx in start_idx:end_idx
        # Classify stencil type based on boundary conditions
        stype = classify_stencil(
            boundary_data.is_boundary, boundary_data.boundary_conditions,
            eval_idx, neighbors, global_to_boundary
        )

        if stype isa DirichletStencil
            # Identity row: only diagonal is 1.0
            fill_dirichlet_entry!(I, J, V, eval_idx, start_pos, num_ops)
        elseif stype isa InteriorStencil
            # Standard interior stencil (no boundary points)
            local_data = view(data, neighbors)
            weights = _build_stencil!(A, b, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k)
            fill_entries!(I, J, V, weights, eval_idx, neighbors, start_pos, k, num_ops)
        else  # HermiteStencil
            # Mixed interior/boundary stencil
            update_hermite_stencil_data!(hermite_data, data, neighbors, ...)
            weights = _build_stencil!(A, b, ℒrbf, ℒmon, hermite_data, eval_point, basis, mon, k)
            fill_entries!(I, J, V, weights, eval_idx, neighbors, start_pos, k, num_ops)
        end
    end
end
```

Stencil classification via dispatch:
- **DirichletStencil**: Identity row (only diagonal)
- **InteriorStencil**: Standard stencil (all neighbors are interior)
- **HermiteStencil**: Mixed interior/boundary stencil

---

## Part 3: Stencil Mathematics (stencil_math.jl)

### Stencil Assembly

```julia
function _build_stencil!(A, b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    return (A \ b)[1:k, :]
end
```

**Key insight**: Multiple dispatch on `data` type automatically selects:
- `data::AbstractVector` → Standard interior stencil
- `data::HermiteStencilData` → Hermite boundary stencil

### Collocation Matrix Building

**Standard (Interior)**:
```julia
function _build_collocation_matrix!(A, data::AbstractVector, basis, mon, k)
    AA = parent(A)
    N = size(A, 2)

    # RBF block (upper triangular)
    for j in 1:k, i in 1:j
        AA[i, j] = basis(data[i], data[j])  # Φ(xᵢ, xⱼ)
    end

    # Polynomial augmentation block
    if basis.poly_deg > -1
        for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data[i])  # P(xᵢ)
        end
    end
end
```

**Hermite (With Boundary Conditions)**:
```julia
function _build_collocation_matrix!(A, data::HermiteStencilData, basis, mon, k)
    AA = parent(A)
    N = size(A, 2)

    # RBF block with Hermite modifications
    for j in 1:k, i in 1:j
        AA[i, j] = compute_hermite_rbf_entry(i, j, data, basis)
    end

    # Polynomial block with boundary modifications
    if basis.poly_deg > -1
        for i in 1:k
            a = view(AA, i, (k + 1):N)
            compute_hermite_poly_entry!(a, i, data, mon)
        end
    end
end
```

### Hermite RBF Entry Dispatch

Uses **9 dispatch methods** based on point types (Interior/Dirichlet/NeumannRobin):

```julia
function compute_hermite_rbf_entry(i, j, data, basis)
    xi, xj = data.data[i], data.data[j]
    type_i = point_type(data.is_boundary[i], data.boundary_conditions[i])
    type_j = point_type(data.is_boundary[j], data.boundary_conditions[j])

    return hermite_rbf_dispatch(type_i, type_j, i, j, xi, xj, data, basis)
end
```

**Example dispatches**:

```julia
# Interior-Interior: Standard evaluation
hermite_rbf_dispatch(::InteriorPoint, ::InteriorPoint, ...) = basis(xi, xj)

# Interior-NeumannRobin: Apply boundary operator to second argument
hermite_rbf_dispatch(::InteriorPoint, ::NeumannRobinPoint, ...) =
    α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)

# NeumannRobin-NeumannRobin: Apply to both arguments
hermite_rbf_dispatch(::NeumannRobinPoint, ::NeumannRobinPoint, ...) =
    α(bc_i) * α(bc_j) * φ +
    α(bc_i) * β(bc_j) * dot(nj, -∇φ) +
    β(bc_i) * α(bc_j) * dot(ni, ∇φ) +
    β(bc_i) * β(bc_j) * ∂i∂j_φ
```

### RHS Vector Building

Similar dispatch pattern:
- `_build_rhs!(b::Vector, ℒrbf, ...)` → Single operator
- `_build_rhs!(b::Matrix, ℒrbf::Tuple, ...)` → Multiple operators
- `_build_rhs!(b, ..., data::AbstractVector, ...)` → Interior
- `_build_rhs!(b, ..., data::HermiteStencilData, ...)` → Hermite

---

## Part 4: Data Structures (types.jl)

### Boundary Condition

```julia
struct BoundaryCondition{T<:Real}
    α::T  # Coefficient for value
    β::T  # Coefficient for normal derivative
end

# Boundary operator: Bu = α*u + β*∂ₙu
# Special cases:
#   Dirichlet: α=1, β=0
#   Neumann:   α=0, β=1
#   Robin:     α≠0, β≠0
```

### Hermite Stencil Data

```julia
struct HermiteStencilData{T<:Real}
    data::AbstractVector{Vector{T}}              # k stencil points
    is_boundary::Vector{Bool}                    # k flags
    boundary_conditions::Vector{BoundaryCondition{T}}  # k conditions
    normals::Vector{Vector{T}}                   # k normal vectors
end
```

### Stencil Classification

```julia
abstract type StencilType end
struct InteriorStencil <: StencilType end   # All neighbors interior
struct DirichletStencil <: StencilType end  # Eval point is Dirichlet
struct HermiteStencil <: StencilType end    # Mixed interior/boundary

function classify_stencil(is_boundary, boundary_conditions, eval_idx, neighbors, ...)
    if sum(is_boundary[neighbors]) == 0
        return InteriorStencil()
    elseif is_boundary[eval_idx] && is_dirichlet(boundary_conditions[...])
        return DirichletStencil()
    else
        return HermiteStencil()
    end
end
```

---

## Call Graph

```
User Code (e.g., Laplacian construction)
    │
    └─→ api.jl: _build_weights(ℒ, op)
            │
            ├─→ Apply operator to basis: ℒrbf = ℒ(basis), ℒmon = ℒ(mon)
            │
            └─→ Route based on boundary conditions
                    │
                    └─→ kernel_exec.jl: build_weights_kernel(...)
                            │
                            ├─→ allocate_sparse_arrays(...)
                            │       └─→ count_nonzeros → Exact allocation
                            │
                            ├─→ launch_kernel!(...)
                            │       │
                            │       └─→ @kernel weight_kernel
                            │               │
                            │               └─→ FOR each eval_point in batch:
                            │                       │
                            │                       ├─→ types.jl: classify_stencil(...)
                            │                       │       └─→ {Dirichlet, Interior, Hermite}
                            │                       │
                            │                       ├─→ IF DirichletStencil:
                            │                       │       └─→ Fill identity row
                            │                       │
                            │                       ├─→ IF InteriorStencil:
                            │                       │       └─→ stencil_math.jl: _build_stencil!
                            │                       │               (standard interior)
                            │                       │
                            │                       └─→ IF HermiteStencil:
                            │                               │
                            │                               ├─→ types.jl: update_hermite_stencil_data!
                            │                               │
                            │                               └─→ stencil_math.jl: _build_stencil!
                            │                                       ├─→ _build_collocation_matrix!
                            │                                       │       └─→ compute_hermite_rbf_entry
                            │                                       │               └─→ hermite_rbf_dispatch
                            │                                       │                       (9 point type combos)
                            │                                       │
                            │                                       ├─→ _build_rhs!
                            │                                       │       ├─→ apply_boundary_to_rbf
                            │                                       │       └─→ apply_boundary_to_mono!
                            │                                       │
                            │                                       └─→ solve(A, b)
                            │
                            └─→ sparse(I, J, V) → Return
```

---

## Key Concepts

### 1. Layer Separation

**Why it matters**:
- **stencil_math.jl** contains pure functions → easily testable
- **kernel_exec.jl** handles parallelism → can benchmark separately
- **api.jl** routes requests → single place to trace flow

### 2. Stencil

A **stencil** is a local approximation of a differential operator at a point:

```
For a point x₀ with k neighbors {x₁, x₂, ..., xₖ}:

    ℒu(x₀) ≈ Σᵢ₌₁ᵏ wᵢ * u(xᵢ)

where wᵢ are the weights we compute by solving A \ b.
```

### 3. Collocation Matrix Structure

```
Standard (Interior):
┌─────────────────┬─────────┐
│  Φ(xᵢ, xⱼ)      │ P(xᵢ)   │  k×k RBF + k×nmon polynomial
├─────────────────┼─────────┤
│  P(xⱼ)ᵀ         │   0     │  nmon×k poly + nmon×nmon zero
└─────────────────┴─────────┘

Hermite (with Boundary):
Same structure, but entries modified by boundary operators:
- Interior-Interior: Φ(xᵢ, xⱼ)
- Interior-Boundary: BⱼΦ(xᵢ, xⱼ)
- Boundary-Boundary: BᵢBⱼΦ(xᵢ, xⱼ)

where Bᵢ = α*I + β*∂ₙᵢ is the boundary operator at point i
```

### 4. Multiple Dispatch Benefits

The code uses Julia's multiple dispatch to automatically select:
- Vector vs Matrix RHS (single vs tuple operators)
- AbstractVector vs HermiteStencilData (interior vs boundary)
- Point type combinations (9 Hermite dispatch variants)

This eliminates explicit conditionals and improves type stability.

---

## Performance Characteristics

### Memory Allocation
- Exact allocation via `count_nonzeros` (counts entries before allocating)
- Interior points: k entries per stencil
- Dirichlet points: 1 entry per stencil (identity row)
- Other boundary points: k entries per stencil
- No over-allocation for interior-only problems

### Parallelization
- Batch processing prevents memory exhaustion
- Work arrays reused within batch
- KernelAbstractions.jl enables CPU/GPU execution
- Type-stable buffers via operator arity traits

### Stencil Classification
- Runtime dispatch via `classify_stencil`
- Zero overhead for interior-only problems (all stencils classify as InteriorStencil)
- Dirichlet points bypass expensive matrix assembly

---

## Summary

The unified solve system achieves:

1. **Clear organization**: 3 layers with distinct responsibilities
2. **Single code path**: One allocation strategy, one kernel for all cases
3. **Better testability**: Pure math layer has no side effects
4. **Easy navigation**: "Where's X?" → Obvious file location
5. **Zero overhead**: Multiple dispatch is compile-time optimization
6. **Exact allocation**: No over-allocation for any problem type
7. **Backward compatible**: All exports maintained

**File Mapping**:
- Want to modify math? → `stencil_math.jl`
- Need better parallelism? → `kernel_exec.jl`
- Adding new operator entry point? → `api.jl`
- New boundary condition type? → `types.jl`
