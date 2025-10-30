# Pseudocode: RBF Weight Building System

This document explains the code flow in `solve.jl`, `solve_hermite.jl`, and `solve_utils.jl`.

## Overview

The system builds sparse weight matrices for RBF operators using one of two approaches:
1. **Standard**: For interior points only
2. **Hermite**: For problems with boundary conditions (Dirichlet/Neumann/Robin)

---

## Part 1: Entry Points & High-Level Flow

### Main Entry Points

```pseudocode
// Entry from operator construction (operators.jl)
function _build_weights(operator, operator_config)
    data = operator_config.data
    eval_points = operator_config.eval_points
    adjl = operator_config.adjl  // adjacency list (neighbors for each eval point)
    basis = operator_config.basis

    // Apply operator to basis functions
    ℒrbf = operator(basis)
    ℒmon = operator(monomial_basis)

    // Route to appropriate implementation
    if has_boundary_conditions(operator_config):
        return _build_weights_hermite(...)  // Hermite path
    else:
        return _build_weights_standard(...)  // Standard path
end
```

### Standard Path (solve.jl)

```pseudocode
function _build_weights_standard(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
    strategy = StandardAllocation()
    boundary_data = nothing

    return _build_weights_unified(
        strategy, data, eval_points, adjl,
        basis, ℒrbf, ℒmon, mon, boundary_data
    )
end
```

### Hermite Path (solve_hermite.jl)

```pseudocode
function _build_weights_hermite(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
                                is_boundary, boundary_conditions, normals)
    strategy = OptimizedAllocation()  // More memory-efficient
    boundary_data = (is_boundary, boundary_conditions, normals)

    return _build_weights_unified(
        strategy, data, eval_points, adjl,
        basis, ℒrbf, ℒmon, mon, boundary_data
    )
end
```

---

## Part 2: Unified Kernel Infrastructure (solve_utils.jl)

### Main Orchestrator

```pseudocode
function _build_weights_unified(strategy, data, eval_points, adjl,
                               basis, ℒrbf, ℒmon, mon, boundary_data)
    // Extract problem dimensions
    N_eval = length(eval_points)          // Number of evaluation points
    N_data = length(data)                 // Total data points
    k = length(first(adjl))               // Stencil size
    nmon = number_of_monomials(basis)     // Polynomial augmentation size
    num_ops = count_operators(ℒrbf)       // 1 for single, N for tuple

    // Allocate sparse matrix arrays (I, J, V)
    I, J, V, row_offsets = allocate_sparse_arrays(
        strategy, data_type, k, N_eval, num_ops, adjl, boundary_data
    )

    // Launch parallel kernel to fill arrays
    launch_kernel(
        strategy, I, J, V, data, eval_points, adjl,
        basis, ℒrbf, ℒmon, mon, boundary_data, row_offsets
    )

    // Construct and return sparse matrix
    if num_ops == 1:
        return sparse(I, J, V[:, 1], N_eval, N_data)
    else:
        return tuple_of_sparse_matrices(I, J, V, num_ops, N_eval, N_data)
end
```

### Memory Allocation Strategies

```pseudocode
// Strategy 1: Standard (simple, may over-allocate)
function allocate_sparse_arrays(StandardAllocation, TD, k, N_eval, num_ops, adjl)
    total_entries = k * N_eval
    I = Vector{Int}(undef, total_entries)
    J = Vector{Int}(undef, total_entries)
    V = Matrix{TD}(undef, total_entries, num_ops)

    // Fill J with neighbor indices
    idx = 1
    for each_eval_point in 1:N_eval:
        for neighbor in adjl[each_eval_point]:
            J[idx] = neighbor
            idx += 1

    return I, J, V, nothing
end

// Strategy 2: Optimized (exact allocation for Hermite with many Dirichlet BCs)
function allocate_sparse_arrays(OptimizedAllocation, TD, k, N_eval, num_ops, adjl,
                                is_boundary, boundary_conditions)
    // Count exact non-zeros (Dirichlet points only contribute diagonal)
    total_nnz = 0
    for eval_idx in 1:N_eval:
        if is_boundary[eval_idx] && is_dirichlet(boundary_conditions[eval_idx]):
            total_nnz += 1  // Only diagonal element
        else:
            total_nnz += k  // Full stencil

    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Matrix{TD}(undef, total_nnz, num_ops)
    row_offsets = compute_cumulative_offsets(...)

    return I, J, V, row_offsets
end
```

---

## Part 3: Kernel Execution (Parallel Processing)

### Standard Kernel

```pseudocode
@kernel function standard_kernel(I, J, V, data, eval_points, adjl,
                                basis, ℒrbf, ℒmon, mon, k, ...)
    // Each kernel invocation processes a batch of evaluation points
    batch_idx = get_global_index()
    start_idx, end_idx = calculate_batch_range(batch_idx, batch_size, N_eval)

    // Pre-allocate work arrays (reused within batch)
    n = k + nmon
    A = Symmetric(zeros(n, n))  // Collocation matrix
    b = prepare_buffer(ℒrbf, n)  // RHS vector(s)

    // Process each evaluation point in batch
    for eval_idx in start_idx:end_idx:
        // Get stencil data
        neighbors = adjl[eval_idx]
        local_data = data[neighbors]
        eval_point = eval_points[eval_idx]

        // Build stencil weights
        weights = build_stencil(A, b, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k)

        // Store in sparse matrix arrays
        fill_sparse_entries(I, J, V, weights, eval_idx, neighbors, k)
end
```

### Optimized Hermite Kernel

```pseudocode
@kernel function optimized_kernel(I, J, V, data, eval_points, adjl,
                                 basis, ℒrbf, ℒmon, mon,
                                 is_boundary, boundary_conditions, normals, ...)
    batch_idx = get_global_index()
    hermite_data = get_hermite_workspace(batch_idx)  // Pre-allocated workspace
    start_idx, end_idx = calculate_batch_range(batch_idx, batch_size, N_eval)

    // Pre-allocate work arrays
    n = k + nmon
    A = Symmetric(zeros(n, n))
    b = prepare_buffer(ℒrbf, n)

    for eval_idx in start_idx:end_idx:
        start_pos = row_offsets[eval_idx]
        neighbors = adjl[eval_idx]
        eval_point = eval_points[eval_idx]

        // Classify stencil type
        stencil_type = classify_stencil(eval_idx, neighbors, is_boundary, boundary_conditions)

        if stencil_type == DirichletStencil:
            // Identity row: only diagonal is 1.0
            fill_dirichlet_entry(I, J, V, eval_idx, start_pos)
            continue

        else if stencil_type == InternalStencil:
            // Standard stencil (no boundary points)
            local_data = view(data, neighbors)
            weights = build_stencil(A, b, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k)

        else if stencil_type == HermiteStencil:
            // Mixed interior/boundary stencil
            update_hermite_data(hermite_data, data, neighbors, is_boundary,
                              boundary_conditions, normals)
            weights = build_hermite_stencil(A, b, ℒrbf, ℒmon, hermite_data,
                                          eval_point, basis, mon, k)

        // Store weights
        fill_sparse_entries(I, J, V, weights, eval_idx, neighbors, start_pos, k)
end
```

---

## Part 4: Stencil Building (Core Math)

### Standard Stencil (solve.jl)

```pseudocode
function build_stencil(A, b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)
    // Build collocation matrix A
    build_collocation_matrix(A, data, basis, mon, k)

    // Build RHS vector b
    build_rhs(b, ℒrbf, ℒmon, data, eval_point, basis, k)

    // Solve linear system
    weights = solve(A, b)  // Returns first k rows (RBF part)
    return weights[1:k, :]
end
```

#### Collocation Matrix

```pseudocode
function build_collocation_matrix(A, data, basis, mon, k)
    // Upper triangular part (symmetric matrix)
    for j in 1:k:
        for i in 1:j:
            A[i, j] = basis(data[i], data[j])  // RBF Φ(xᵢ, xⱼ)

    // Polynomial augmentation block
    if has_polynomial_augmentation:
        for i in 1:k:
            A[i, k+1:end] = mon(data[i])  // P(xᵢ)

    // Matrix structure:
    // ┌─────────┬──────┐
    // │ Φ(xᵢ,xⱼ)│ P(xᵢ)│  k×k RBF block + k×nmon polynomial block
    // ├─────────┼──────┤
    // │ P(xⱼ)ᵀ │  0   │  nmon×k polynomial block + nmon×nmon zero block
    // └─────────┴──────┘
end
```

#### RHS Vector (Standard)

```pseudocode
function build_rhs(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    // RBF part: apply operator at eval_point
    for i in 1:k:
        b[i] = ℒrbf(eval_point, data[i])  // ℒΦ(x_eval, xᵢ)

    // Polynomial part: apply operator to monomials
    if has_polynomial_augmentation:
        bmono = view(b, k+1:end)
        ℒmon(bmono, eval_point)  // ℒP(x_eval)
end

// Note: For tuple operators (ℒ₁, ℒ₂, ..., ℒₙ), process each operator separately
// b becomes a matrix with columns [b₁, b₂, ..., bₙ]
```

### Hermite Stencil (solve_hermite.jl)

```pseudocode
function build_hermite_stencil(A, b, ℒrbf, ℒmon, hermite_data, eval_point, basis, mon, k)
    // Build modified collocation matrix
    build_hermite_collocation_matrix(A, hermite_data, basis, mon, k)

    // Build modified RHS
    build_hermite_rhs(b, ℒrbf, ℒmon, hermite_data, eval_point, basis, mon, k)

    // Solve
    weights = solve(A, b)
    return weights[1:k, :]
end
```

#### Hermite Collocation Matrix

```pseudocode
function build_hermite_collocation_matrix(A, hermite_data, basis, mon, k)
    // Build RBF block with boundary modifications
    for j in 1:k:
        for i in 1:j:
            A[i, j] = hermite_rbf_entry(i, j, hermite_data, basis)

    // Build polynomial block with boundary modifications
    if has_polynomial_augmentation:
        for i in 1:k:
            A[i, k+1:end] = hermite_poly_entry(i, hermite_data, mon)
end
```

#### RBF Entry Calculation (Dispatch-Based)

```pseudocode
function hermite_rbf_entry(i, j, hermite_data, basis)
    xi = hermite_data.data[i]
    xj = hermite_data.data[j]

    // Determine point types
    type_i = point_type(hermite_data.is_boundary[i], hermite_data.bc[i])
    type_j = point_type(hermite_data.is_boundary[j], hermite_data.bc[j])

    // Dispatch to specialized method (9 combinations)
    return hermite_rbf_entry_dispatch(type_i, type_j, i, j, xi, xj, hermite_data, basis)
end

// Example dispatches:

// Interior-Interior: Standard evaluation
function dispatch(InteriorPoint, InteriorPoint, i, j, xi, xj, data, basis)
    return basis(xi, xj)  // Φ(xi, xj)
end

// Interior-NeumannRobin: Apply boundary operator to second point
function dispatch(InteriorPoint, NeumannRobinPoint, i, j, xi, xj, data, basis)
    φ = basis(xi, xj)
    ∇φ = gradient(basis)(xi, xj)
    bc_j = data.boundary_conditions[j]
    nj = data.normals[j]

    return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)  // B_j[Φ(xi, ·)](xj)
end

// NeumannRobin-NeumannRobin: Apply boundary operators to both points
function dispatch(NeumannRobinPoint, NeumannRobinPoint, i, j, xi, xj, data, basis)
    φ = basis(xi, xj)
    ∇φ = gradient(basis)(xi, xj)
    bc_i = data.boundary_conditions[i]
    bc_j = data.boundary_conditions[j]
    ni = data.normals[i]
    nj = data.normals[j]

    // Mixed second derivative term
    ∂i∂j_φ = directional_second_derivative(basis, ni, nj)(xi, xj)

    return (
        α(bc_i) * α(bc_j) * φ +
        α(bc_i) * β(bc_j) * dot(nj, -∇φ) +
        β(bc_i) * α(bc_j) * dot(ni, ∇φ) +
        β(bc_i) * β(bc_j) * ∂i∂j_φ
    )
end

// Similar dispatches for all 9 combinations:
// Interior × {Interior, Dirichlet, NeumannRobin}
// Dirichlet × {Interior, Dirichlet, NeumannRobin}
// NeumannRobin × {Interior, Dirichlet, NeumannRobin}
```

#### Hermite RHS Building

```pseudocode
function build_hermite_rhs(b, ℒrbf, ℒmon, hermite_data, eval_point, basis, mon, k)
    // RBF part: apply boundary conditions to each stencil point
    for i in 1:k:
        b[i] = apply_boundary_to_rbf(
            ℒrbf, eval_point, hermite_data.data[i],
            hermite_data.is_boundary[i],
            hermite_data.boundary_conditions[i],
            hermite_data.normals[i]
        )

    // Polynomial part: apply boundary conditions at evaluation point
    if has_polynomial_augmentation:
        bmono = view(b, k+1:end)
        eval_idx = find_eval_point_index(hermite_data.data, eval_point)

        apply_boundary_to_mono(
            bmono, ℒmon, mon, eval_point,
            hermite_data.is_boundary[eval_idx],
            hermite_data.boundary_conditions[eval_idx],
            hermite_data.normals[eval_idx]
        )
end

// Helper: Apply boundary conditions to RBF operator
function apply_boundary_to_rbf(ℒrbf, eval_point, data_point, is_bound, bc, normal)
    if !is_bound || is_dirichlet(bc):
        return ℒrbf(eval_point, data_point)  // ℒΦ(x_eval, x_data)
    else:
        // Neumann/Robin: α*ℒΦ + β*ℒ(∂ₙΦ)
        return α(bc) * ℒrbf(eval_point, data_point) +
               β(bc) * ℒrbf(eval_point, data_point, normal)
end

// Helper: Apply boundary conditions to monomial operator
function apply_boundary_to_mono(bmono, ℒmon, mon, eval_point, is_bound, bc, normal)
    if !is_bound || is_dirichlet(bc):
        ℒmon(bmono, eval_point)  // ℒP(x_eval)
    else:
        // Neumann/Robin: α*ℒP + β*ℒ(∂ₙP)
        // Pre-compute polynomial values and derivatives
        poly_vals = mon(eval_point)
        deriv_vals = ∂_normal(mon, normal)(eval_point)

        for idx in 1:length(bmono):
            bmono[idx] = α(bc) * poly_vals[idx] + β(bc) * deriv_vals[idx]
end
```

---

## Part 5: Key Data Structures

### HermiteStencilData

```pseudocode
struct HermiteStencilData:
    data: Vector{Point}                    // k points in stencil
    is_boundary: Vector{Bool}              // k flags
    boundary_conditions: Vector{BC}        // k boundary conditions
    normals: Vector{Normal}                // k outward normal vectors
```

### Boundary Condition Types

```pseudocode
// Base type
abstract type BoundaryCondition

// Concrete types
struct Dirichlet <: BoundaryCondition:
    value: Float64

struct Neumann <: BoundaryCondition:
    value: Float64

struct Robin <: BoundaryCondition:
    α: Float64  // coefficient for value
    β: Float64  // coefficient for normal derivative

// Boundary operator: Bu = α*u + β*∂ₙu
```

---

## Part 6: Call Graph Summary

```
User Code
    │
    ├─→ RadialBasisOperator construction
    │       │
    │       └─→ _build_weights(ℒ, op)
    │               │
    │               ├─→ ℒrbf = ℒ(basis)
    │               ├─→ ℒmon = ℒ(mon)
    │               │
    │               └─→ _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, [boundary_data])
    │                       │
    │                       ├─→ [Standard Path]
    │                       │       └─→ _build_weights_unified(StandardAllocation, ...)
    │                       │               │
    │                       │               ├─→ allocate_sparse_arrays(StandardAllocation, ...)
    │                       │               ├─→ launch_kernel(StandardAllocation, ...)
    │                       │               │       │
    │                       │               │       └─→ @kernel standard_kernel
    │                       │               │               │
    │                       │               │               └─→ FOR each eval_point:
    │                       │               │                       │
    │                       │               │                       ├─→ _build_stencil!
    │                       │               │                       │       │
    │                       │               │                       │       ├─→ _build_collocation_matrix!
    │                       │               │                       │       ├─→ _build_rhs!
    │                       │               │                       │       │       │
    │                       │               │                       │       │       ├─→ _build_rhs_core!
    │                       │               │                       │       │       │       ├─→ RBF section loop
    │                       │               │                       │       │       │       └─→ Monomial section
    │                       │               │                       │       │
    │                       │               │                       │       └─→ solve(A, b)
    │                       │               │                       │
    │                       │               │                       └─→ fill_sparse_entries!
    │                       │               │
    │                       │               └─→ sparse(I, J, V)
    │                       │
    │                       └─→ [Hermite Path]
    │                               └─→ _build_weights_unified(OptimizedAllocation, ...)
    │                                       │
    │                                       ├─→ allocate_sparse_arrays(OptimizedAllocation, ...)
    │                                       ├─→ launch_kernel(OptimizedAllocation, ...)
    │                                       │       │
    │                                       │       └─→ @kernel optimized_kernel
    │                                       │               │
    │                                       │               └─→ FOR each eval_point:
    │                                       │                       │
    │                                       │                       ├─→ stencil_type(...)
    │                                       │                       │       └─→ {DirichletStencil, InternalStencil, HermiteStencil}
    │                                       │                       │
    │                                       │                       ├─→ IF DirichletStencil:
    │                                       │                       │       └─→ fill_dirichlet_entry! (I[row]=row, J[row]=row, V[row]=1.0)
    │                                       │                       │
    │                                       │                       ├─→ IF InternalStencil:
    │                                       │                       │       └─→ _build_stencil! (standard)
    │                                       │                       │
    │                                       │                       └─→ IF HermiteStencil:
    │                                       │                               └─→ _build_stencil!(HermiteStencilData)
    │                                       │                                       │
    │                                       │                                       ├─→ _build_collocation_matrix!(HermiteStencilData)
    │                                       │                                       │       │
    │                                       │                                       │       ├─→ FOR i,j: _hermite_rbf_entry(i, j, data, basis)
    │                                       │                                       │       │       │
    │                                       │                                       │       │       ├─→ point_type(i) + point_type(j)
    │                                       │                                       │       │       └─→ dispatch to appropriate method (9 cases)
    │                                       │                                       │       │
    │                                       │                                       │       └─→ FOR i: _hermite_poly_entry!(i, data, mon)
    │                                       │                                       │
    │                                       │                                       ├─→ _build_rhs!(HermiteStencilData)
    │                                       │                                       │       │
    │                                       │                                       │       ├─→ FOR i: _apply_boundary_to_rbf(...)
    │                                       │                                       │       └─→ _apply_boundary_to_mono!(...)
    │                                       │                                       │
    │                                       │                                       └─→ solve(A, b)
    │                                       │
    │                                       └─→ sparse(I, J, V)
    │
    └─→ Sparse weight matrices returned
```

---

## Part 7: Key Concepts

### 1. Stencil

A **stencil** is a local approximation of a differential operator at a point using nearby data points:

```
For a point x₀ with k neighbors {x₁, x₂, ..., xₖ}:

    ℒu(x₀) ≈ Σᵢ₌₁ᵏ wᵢ * u(xᵢ)

where wᵢ are the weights we compute.
```

### 2. Collocation Matrix Structure

```
Standard (Interior):
┌─────────────────┬─────────┐
│  Φ(xᵢ, xⱼ)      │ P(xᵢ)   │  k × k RBF block + k × nmon polynomial
├─────────────────┼─────────┤
│  P(xⱼ)ᵀ         │   0     │  nmon × k polynomial + nmon × nmon zero
└─────────────────┴─────────┘

Hermite (with Boundary):
Same structure, but entries modified by boundary operators:
- Interior-Interior: standard Φ(xᵢ, xⱼ)
- Interior-Boundary: BⱼΦ(xᵢ, xⱼ)
- Boundary-Boundary: BᵢBⱼΦ(xᵢ, xⱼ)

where Bᵢ = α*I + β*∂ₙᵢ is the boundary operator at point i
```

### 3. Operator Types

```pseudocode
// Single operator (returns 1 sparse matrix)
ℒ = Laplacian(basis, data, ...)
weights = ℒ.weights  // N_eval × N_data sparse matrix

// Tuple operators (returns tuple of sparse matrices)
ℒ = (∂_x, ∂_y)  // Gradient
weights = (W_x, W_y)  // Each is N_eval × N_data
```

### 4. Type Stability

The refactored code uses **trait dispatch** to ensure type-stable buffer allocation:

```pseudocode
// OLD (type-unstable):
function prepare_b(ℒ, T, n)
    if ℒ isa Tuple:
        return zeros(T, n, length(ℒ))  // Matrix
    else:
        return zeros(T, n)              // Vector
    end
end

// NEW (type-stable):
trait OperatorArity{N}  // N known at compile-time
function prepare_b(SingleOperator, T, n)   → Vector{T}(undef, n)
function prepare_b(MultiOperator{N}, T, n) → Matrix{T}(undef, n, N)
```

---

## Part 8: Example Walkthrough

### Example: Laplacian with Mixed Boundary Conditions

```pseudocode
// Setup
data = [interior_points..., boundary_points...]
eval_points = data  // Collocation (eval at same points)
basis = PHS(3)  // Polyharmonic spline r³
operator = Laplacian
boundary_conditions = [Dirichlet(0.0), Neumann(1.0), Robin(1.0, 0.5), ...]
normals = [n₁, n₂, n₃, ...]

// Step 1: Entry point
ℒ = Laplacian(basis, data, eval_points, neighbors, boundary_conditions, normals)

// Step 2: Build weights
ℒrbf = Laplacian(basis)        // ∇²Φ(x, y)
ℒmon = Laplacian(mon)          // ∇²P(x) = ∇²(1, x, y, x², xy, y², ...)
weights = _build_weights_hermite(...)

// Step 3: Unified infrastructure
strategy = OptimizedAllocation  // Because we have boundary conditions
boundary_data = (is_boundary, boundary_conditions, normals)
_build_weights_unified(strategy, ..., boundary_data)

// Step 4: Allocate exact memory
//   - Interior points: k entries each
//   - Dirichlet BCs: 1 entry each (diagonal only)
//   - Neumann/Robin BCs: k entries each
total_nnz = (num_interior + num_neumann_robin) * k + num_dirichlet
allocate_sparse_arrays(OptimizedAllocation, ..., total_nnz)

// Step 5: Launch kernel (parallel processing)
@kernel optimized_kernel:
    FOR eval_idx in batch:
        neighbors = adjl[eval_idx]

        // Classify stencil
        IF eval_point is Dirichlet:
            // Identity row: u(x₀) = boundary_value
            I[row] = eval_idx
            J[row] = eval_idx
            V[row] = 1.0

        ELSE IF all neighbors are interior:
            // Standard stencil (no boundary modifications)
            build_stencil(standard)

        ELSE:
            // Hermite stencil (mixed interior/boundary)

            // Build collocation matrix A
            FOR i, j in neighbors:
                IF both interior:
                    A[i,j] = basis(xᵢ, xⱼ)  // r³

                ELSE IF i interior, j Neumann:
                    // Apply boundary operator to j
                    φ = r³
                    ∇φ = [3r²cos(θ), 3r²sin(θ)]
                    A[i,j] = α * φ + β * dot(nⱼ, -∇φ)

                ELSE IF both Neumann/Robin:
                    // Mixed derivative term
                    φ = r³
                    ∇φ = ...
                    ∂ᵢ∂ⱼφ = directional_second_derivative(nᵢ, nⱼ)
                    A[i,j] = αᵢαⱼφ + αᵢβⱼ·(∇φ·nⱼ) + βᵢαⱼ·(∇φ·nᵢ) + βᵢβⱼ∂ᵢ∂ⱼφ

            // Build RHS b
            FOR i in neighbors:
                IF interior:
                    b[i] = ∇²r³(x₀, xᵢ)
                ELSE IF Neumann:
                    b[i] = α*∇²r³(x₀, xᵢ) + β*∇²(∂ₙr³)(x₀, xᵢ)

            // Monomial part (if eval_point is Neumann/Robin)
            IF eval_point has Neumann/Robin BC:
                // Boundary operator modifies monomial evaluation
                poly_vals = [1, x₀, y₀, x₀², ...]
                deriv_vals = ∂ₙ[1, x₀, y₀, x₀², ...] = [0, n_x, n_y, 2x₀n_x, ...]
                bmono = α*poly_vals + β*deriv_vals
            ELSE:
                bmono = ∇²[1, x₀, y₀, x₀², ...] = [0, 0, 0, 2, ...]

            // Solve: A * weights = b
            weights = solve(A, b)[1:k]

        // Store in sparse arrays
        fill_sparse_entries(I, J, V, eval_idx, neighbors, weights)

// Step 6: Construct sparse matrix
return sparse(I, J, V, N_eval, N_data)
```

---

## Part 9: Summary

### Code Flow Pipeline

```
Operator Construction
    ↓
Dispatch to Standard or Hermite
    ↓
Unified Infrastructure (_build_weights_unified)
    ├─→ Allocate sparse arrays (strategy-based)
    ├─→ Launch parallel kernel
    │      ├─→ Process batches of evaluation points
    │      ├─→ Build local stencils (A\b)
    │      │      ├─→ Build collocation matrix
    │      │      ├─→ Build RHS vector
    │      │      └─→ Solve linear system
    │      └─→ Fill sparse matrix entries
    └─→ Construct final sparse matrix

Return: Sparse weight matrix W where W * u ≈ ℒu
```

### Key Differences: Standard vs Hermite

| Aspect | Standard | Hermite |
|--------|----------|---------|
| **Stencil Points** | All interior | Mix of interior and boundary |
| **Collocation Matrix** | Standard RBF evaluation | Modified by boundary operators |
| **RHS Vector** | Standard operator evaluation | Modified by boundary conditions |
| **Memory Strategy** | Simple (k*N allocation) | Optimized (exact counting) |
| **Dirichlet BCs** | N/A | Identity rows (trivial stencil) |
| **Neumann/Robin BCs** | N/A | Normal derivatives in equations |

### Performance Characteristics

```
Standard Path:
  - Simpler logic
  - Slightly over-allocates memory
  - No boundary condition overhead
  - Best for interior-only problems

Hermite Path:
  - More complex logic (9 dispatch cases)
  - Exact memory allocation
  - Handles all BC types
  - Best for PDEs with boundary conditions
  - Dirichlet BCs are "free" (identity rows)
```

---

## Questions or Clarifications?

This document should give you a complete mental model of how the weight-building system works. The key insights are:

1. **Unified infrastructure**: Both paths use the same kernel template
2. **Dispatch-based flexibility**: Type system handles complexity
3. **Local stencils**: Each evaluation point has its own small linear system
4. **Boundary conditions**: Modify the collocation matrix and RHS, not the solve

Let me know if you'd like me to expand on any particular section!
