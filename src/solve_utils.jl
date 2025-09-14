# Required imports for unified kernel infrastructure
using LinearAlgebra: Symmetric
using SparseArrays: sparse
using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU

# Sparse matrix allocation strategies
abstract type SparseAllocationStrategy end
struct StandardAllocation <: SparseAllocationStrategy end
struct OptimizedAllocation <: SparseAllocationStrategy end

_num_ops(_) = 1
_num_ops(ℒ::Tuple) = length(ℒ)
_prepare_b(_, T, n) = zeros(T, n)
_prepare_b(ℒ::Tuple, T, n) = zeros(T, n, length(ℒ))

"""
Count the actual number of non-zero elements for optimized sparse matrix allocation.
Handles boundary conditions where Dirichlet points only contribute diagonal elements.

Returns:
- total_nnz: Total number of non-zero elements across all stencils
- nnz_per_row: Vector containing number of non-zeros for each evaluation point
- row_offsets: Cumulative offsets for filling I,J,V arrays
"""
function _count_nonzeros(
    adjl, is_boundary::Vector{Bool}, boundary_conditions::Vector{<:BoundaryCondition}
)
    N_eval = length(adjl)
    nnz_per_row = Vector{Int}(undef, N_eval)
    row_offsets = Vector{Int}(undef, N_eval + 1)
    global_to_boundary = _construct_global_to_boundary(is_boundary)

    total_nnz = 0
    row_offsets[1] = 1  # 1-based indexing for first row

    for eval_idx in 1:N_eval
        if is_boundary[eval_idx]
            boundary_idx = global_to_boundary[eval_idx]
            bc = boundary_conditions[boundary_idx]
            if is_dirichlet(bc)
                # Dirichlet: only diagonal element is non-zero
                nnz_per_row[eval_idx] = 1
            else
                # Neumann/Robin center: full stencil
                nnz_per_row[eval_idx] = length(adjl[eval_idx])
            end
        else
            # Interior: full stencil
            nnz_per_row[eval_idx] = length(adjl[eval_idx])
        end

        total_nnz += nnz_per_row[eval_idx]
        row_offsets[eval_idx + 1] = total_nnz + 1
    end

    return total_nnz, nnz_per_row, row_offsets
end

"""
Construct global_to_boundary index mapping using is_boundary global vector.
is_boundary is N_tot x 1 and global_to_boundary is N_tot x 1.
This structure ensures we can use normals and boundary_conditions which are only defined on boundary points.
"""
function _construct_global_to_boundary(is_boundary::Vector{Bool})
    N_tot = length(is_boundary)
    global_to_boundary = Vector{Int}(undef, N_tot)

    boundary_counter = 0
    for i in 1:N_tot
        if is_boundary[i]
            boundary_counter += 1
            global_to_boundary[i] = boundary_counter
        else
            global_to_boundary[i] = 0  # Non-boundary points map to 0
        end
    end

    return global_to_boundary
end

"""
Allocate sparse matrix arrays using standard strategy (simple k*Na allocation).
May over-allocate but is simple and fast for standard (non-Hermite) cases.
"""
function _allocate_sparse_arrays(
    ::StandardAllocation, TD, k::Int, Na::Int, num_ops::Int, adjl
)
    I = zeros(Int, k * Na)
    J = reduce(vcat, adjl)
    V = zeros(TD, k * Na, num_ops)
    row_offsets = nothing  # Not needed for standard allocation
    return I, J, V, row_offsets
end

"""
Allocate sparse matrix arrays using optimized strategy (exact allocation based on boundary conditions).
More memory efficient for Hermite cases with many Dirichlet boundary conditions.
"""
function _allocate_sparse_arrays(
    ::OptimizedAllocation,
    TD,
    k::Int,
    Na::Int,
    num_ops::Int,
    adjl,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
)
    # Count exact number of non-zeros needed
    total_nnz, _, row_offsets = _count_nonzeros(adjl, is_boundary, boundary_conditions)

    # Allocate exact memory
    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Matrix{TD}(undef, total_nnz, num_ops)

    return I, J, V, row_offsets
end

"""
Shared kernel utilities for building RBF weights with batch processing.
This provides the common infrastructure used by both standard and Hermite implementations.
"""

"""
Calculate batch indices for kernel execution.
"""
@inline function _calculate_batch_range(batch_idx::Int, batch_size::Int, N_eval::Int)
    start_idx = (batch_idx - 1) * batch_size + 1
    end_idx = min(batch_idx * batch_size, N_eval)
    return start_idx, end_idx
end

"""
Fill sparse matrix indices for standard allocation strategy.
"""
@inline function _fill_indices_standard!(I, eval_idx::Int, k::Int)
    @inbounds for idx in 1:k
        I[(eval_idx - 1) * k + idx] = eval_idx
    end
end

"""
Fill sparse matrix values for standard allocation strategy.
"""
@inline function _fill_values_standard!(V, weights, eval_idx::Int, k::Int, num_ops::Int)
    @inbounds for op in 1:num_ops
        for idx in 1:k
            V[(eval_idx - 1) * k + idx, op] = weights[idx, op]
        end
    end
end

"""
Fill sparse matrix entries for optimized allocation strategy.
"""
@inline function _fill_entries_optimized!(
    I, J, V, weights, eval_idx::Int, neighbors, start_pos::Int, k::Int, num_ops::Int
)
    @inbounds for local_idx in 1:k
        pos = start_pos + local_idx - 1
        I[pos] = eval_idx
        J[pos] = neighbors[local_idx]
        if num_ops == 1
            V[pos, 1] = weights[local_idx]
        else
            for op in 1:num_ops
                V[pos, op] = weights[local_idx, op]
            end
        end
    end
end

"""
Handle Dirichlet stencil case for optimized allocation.
"""
@inline function _handle_dirichlet_optimized!(
    I, J, V, eval_idx::Int, start_pos::Int, num_ops::Int
)
    I[start_pos] = eval_idx
    J[start_pos] = eval_idx
    @inbounds for op in 1:num_ops
        V[start_pos, op] = 1.0
    end
end

"""
Unified kernel template for building RBF weights.
This template handles both standard and Hermite cases through multiple dispatch.
"""
function _build_weights_unified(
    strategy::SparseAllocationStrategy,
    data,
    eval_points,
    adjl,
    basis,
    ℒrbf,
    ℒmon,
    mon,
    boundary_data=nothing;
    batch_size::Int=10,
    device=CPU(),
)
    TD = eltype(first(data))
    k = length(first(adjl))
    nmon = binomial(length(first(data)) + basis.poly_deg, basis.poly_deg)
    num_ops = _num_ops(ℒrbf)
    N_eval = length(eval_points)

    # Allocate arrays using strategy
    if boundary_data === nothing
        I, J, V, row_offsets = _allocate_sparse_arrays(
            strategy, TD, k, N_eval, num_ops, adjl
        )
    else
        is_boundary, boundary_conditions, normals = boundary_data
        I, J, V, row_offsets = _allocate_sparse_arrays(
            strategy, TD, k, N_eval, num_ops, adjl, is_boundary, boundary_conditions
        )
    end

    # Calculate batches
    n_batches = ceil(Int, N_eval / batch_size)

    # Launch unified kernel
    _launch_unified_kernel!(
        strategy,
        I,
        J,
        V,
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        boundary_data,
        row_offsets,
        batch_size,
        N_eval,
        n_batches,
        k,
        nmon,
        num_ops,
        device,
    )

    # Create and return sparse matrix/matrices
    nrows = N_eval
    ncols = length(data)

    if num_ops == 1
        return sparse(I, J, V[:, 1], nrows, ncols)
    else
        return ntuple(i -> sparse(I, J, V[:, i], nrows, ncols), num_ops)
    end
end

"""
Launch unified kernel with appropriate dispatch based on allocation strategy.
"""
function _launch_unified_kernel!(
    ::StandardAllocation,
    I,
    J,
    V,
    data,
    eval_points,
    adjl,
    basis,
    ℒrbf,
    ℒmon,
    mon,
    boundary_data,
    row_offsets,
    batch_size,
    N_eval,
    n_batches,
    k,
    nmon,
    num_ops,
    device,
)
    TD = eltype(first(data))

    @kernel function standard_kernel(
        I,
        J,
        V,
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        k,
        batch_size,
        N_eval,
        nmon,
        num_ops,
        TD,
    )
        batch_idx = @index(Global)
        start_idx, end_idx = _calculate_batch_range(batch_idx, batch_size, N_eval)

        # Pre-allocate work arrays for this thread
        n = k + nmon
        A = Symmetric(zeros(TD, n, n), :U)
        b = _prepare_b(ℒrbf, TD, n)

        # Process each point in the batch
        for eval_idx in start_idx:end_idx
            _fill_indices_standard!(I, eval_idx, k)

            # Get data points in the influence domain
            local_data = [data[j] for j in adjl[eval_idx]]

            # Build stencil
            weights = _build_stencil!(
                A, b, ℒrbf, ℒmon, local_data, eval_points[eval_idx], basis, mon, k
            )

            # Store weights
            _fill_values_standard!(V, weights, eval_idx, k, num_ops)
        end
    end

    kernel = standard_kernel(device)
    kernel(
        I,
        J,
        V,
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        k,
        batch_size,
        N_eval,
        nmon,
        num_ops,
        TD;
        ndrange=n_batches,
        workgroupsize=1,
    )
    return KernelAbstractions.synchronize(device)
end

function _launch_unified_kernel!(
    ::OptimizedAllocation,
    I,
    J,
    V,
    data,
    eval_points,
    adjl,
    basis,
    ℒrbf,
    ℒmon,
    mon,
    boundary_data,
    row_offsets,
    batch_size,
    N_eval,
    n_batches,
    k,
    nmon,
    num_ops,
    device,
)
    is_boundary, boundary_conditions, normals = boundary_data
    TD = eltype(first(data))
    dim = length(first(data))

    # Pre-allocate Hermite stencil data for each batch
    batch_hermite_datas = [HermiteStencilData{TD}(k, dim) for _ in 1:n_batches]
    global_to_boundary = _construct_global_to_boundary(is_boundary)

    @kernel function optimized_kernel(
        I,
        J,
        V,
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        is_boundary,
        boundary_conditions,
        normals,
        batch_hermite_datas,
        global_to_boundary,
        row_offsets,
        batch_size,
        N_eval,
        nmon,
        k,
        num_ops,
        TD,
    )
        batch_idx = @index(Global)
        hermite_data = batch_hermite_datas[batch_idx]
        start_idx, end_idx = _calculate_batch_range(batch_idx, batch_size, N_eval)

        # Pre-allocate work arrays for this thread
        n = k + nmon
        A = Symmetric(zeros(TD, n, n), :U)
        b = _prepare_b(ℒrbf, TD, n)

        for eval_idx in start_idx:end_idx
            start_pos = row_offsets[eval_idx]
            neighbors = adjl[eval_idx]
            eval_point = eval_points[eval_idx]

            # Handle different stencil types
            stencil_type_result = stencil_type(
                is_boundary, boundary_conditions, eval_idx, neighbors, global_to_boundary
            )

            if isa(stencil_type_result, DirichletStencil)
                _handle_dirichlet_optimized!(I, J, V, eval_idx, start_pos, num_ops)
                continue
            end
            if isa(stencil_type_result, InternalStencil)
                weights = _build_stencil!(
                    A, b, ℒrbf, ℒmon, view(data, neighbors), eval_point, basis, mon, k
                )
            end
            if isa(stencil_type_result, HermiteStencil)
                update_stencil_data!(
                    hermite_data,
                    data,
                    neighbors,
                    is_boundary,
                    boundary_conditions,
                    normals,
                    global_to_boundary,
                )
                weights = _build_stencil!(
                    A, b, ℒrbf, ℒmon, hermite_data, eval_point, basis, mon, k
                )
            end
            _fill_entries_optimized!(
                I, J, V, weights, eval_idx, neighbors, start_pos, k, num_ops
            )
        end
    end

    kernel = optimized_kernel(device)
    kernel(
        I,
        J,
        V,
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        is_boundary,
        boundary_conditions,
        normals,
        batch_hermite_datas,
        global_to_boundary,
        row_offsets,
        batch_size,
        N_eval,
        nmon,
        k,
        num_ops,
        TD;
        ndrange=n_batches,
        workgroupsize=1,
    )
    return KernelAbstractions.synchronize(device)
end
