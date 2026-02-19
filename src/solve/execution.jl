using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU
using SparseArrays: sparse, sparsevec
using LinearAlgebra: Symmetric

# ============================================================================
# Memory Allocation
# ============================================================================

"""
    allocate_sparse_arrays(TD, k, N_eval, num_ops, adjl, boundary_data)

Allocate sparse matrix arrays for COO format sparse matrix construction.
Exactly counts non-zeros: interior points get k entries, Dirichlet points get 1 entry.
"""
function allocate_sparse_arrays(
        TD, k::Int, N_eval::Int, num_ops::Int, adjl, boundary_data::BoundaryData
    )
    # Count exact non-zeros needed
    total_nnz, row_offsets = count_nonzeros(
        adjl, boundary_data.is_boundary, boundary_data.boundary_conditions
    )

    I = Vector{Int}(undef, total_nnz)
    J = Vector{Int}(undef, total_nnz)
    V = Matrix{TD}(undef, total_nnz, num_ops)

    return I, J, V, row_offsets
end

"""
    count_nonzeros(adjl, is_boundary, boundary_conditions)

Count exact number of non-zero entries for optimized allocation.
Returns (total_nnz, row_offsets) where row_offsets[i] is the starting position for row i.
"""
function count_nonzeros(
        adjl, is_boundary::Vector{Bool}, boundary_conditions::Vector{<:BoundaryCondition}
    )
    N_eval = length(adjl)
    row_offsets = Vector{Int}(undef, N_eval + 1)
    global_to_boundary = construct_global_to_boundary(is_boundary)

    total_nnz = 0
    row_offsets[1] = 1  # 1-based indexing

    for eval_idx in 1:N_eval
        if is_boundary[eval_idx]
            boundary_idx = global_to_boundary[eval_idx]
            bc = boundary_conditions[boundary_idx]
            if is_dirichlet(bc)
                # Dirichlet: only diagonal element
                total_nnz += 1
            else
                # Neumann/Robin: full stencil
                total_nnz += length(adjl[eval_idx])
            end
        else
            # Interior: full stencil
            total_nnz += length(adjl[eval_idx])
        end

        row_offsets[eval_idx + 1] = total_nnz + 1
    end

    return total_nnz, row_offsets
end

"""
    construct_global_to_boundary(is_boundary)

Construct mapping from global indices to boundary-only indices.
For boundary points: global_to_boundary[i] = boundary array index
For interior points: global_to_boundary[i] = 0 (sentinel)
"""
function construct_global_to_boundary(is_boundary::Vector{Bool})
    N_tot = length(is_boundary)
    global_to_boundary = Vector{Int}(undef, N_tot)

    boundary_counter = 0
    for i in 1:N_tot
        if is_boundary[i]
            boundary_counter += 1
            global_to_boundary[i] = boundary_counter
        else
            global_to_boundary[i] = 0
        end
    end

    return global_to_boundary
end

# ============================================================================
# Sparse Matrix Construction
# ============================================================================

"""
    _construct_sparse(I, J, V, N_eval, N_data, num_ops)

Construct sparse matrix/vector from COO arrays.
# Future GPU support: convert to device-sparse format here (see #88)
"""
function _construct_sparse(I, J, V, N_eval, N_data, num_ops)
    if num_ops == 1
        return sparse(I, J, V[:, 1], N_eval, N_data)
    else
        if N_eval == 1
            return ntuple(i -> sparsevec(J, V[:, i], N_data), num_ops)
        else
            return ntuple(i -> sparse(I, J, V[:, i], N_eval, N_data), num_ops)
        end
    end
end

# ============================================================================
# Kernel Orchestration
# ============================================================================

"""
    build_weights_kernel(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
                        boundary_data; batch_size, device)

Main orchestrator for weight computation. Currently CPU-only.
GPU stencil solve is not yet supported — see GitHub issue #88.
"""
function build_weights_kernel(
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        boundary_data::BoundaryData;
        batch_size::Int = 10,
        device = CPU(),
    )
    if !(device isa CPU)
        throw(ArgumentError(
            "GPU weight computation is not yet supported. " *
            "A GPU-kernel-compatible dense solver for stencil matrices is required. " *
            "See https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/88"
        ))
    end

    TD = eltype(first(data))
    k = length(first(adjl))
    nmon = binomial(length(first(data)) + basis.poly_deg, basis.poly_deg)
    num_ops = _num_ops(ℒrbf)
    N_eval = length(eval_points)

    # Allocate sparse arrays
    I, J, V, row_offsets = allocate_sparse_arrays(
        TD, k, N_eval, num_ops, adjl, boundary_data
    )

    # Calculate batches
    n_batches = ceil(Int, N_eval / batch_size)

    # Launch kernel
    launch_kernel!(
        I, J, V, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        boundary_data, row_offsets, batch_size, N_eval, n_batches,
        k, nmon, num_ops, device,
    )

    return _construct_sparse(I, J, V, N_eval, length(data), num_ops)
end

# ============================================================================
# CPU Kernel
# ============================================================================

"""
    launch_kernel!(...)

Launch parallel CPU kernel for weight computation.
Handles Dirichlet/Interior/Hermite stencil classification via dispatch.
"""
function launch_kernel!(
        I, J, V, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        boundary_data::BoundaryData, row_offsets,
        batch_size, N_eval, n_batches, k, nmon, num_ops, device,
    )
    TD = eltype(first(data))
    dim = length(first(data))

    # Pre-allocate Hermite workspace for each batch (includes polynomial workspace)
    batch_hermite_datas = [HermiteStencilData{TD}(k, dim, nmon) for _ in 1:n_batches]
    global_to_boundary = construct_global_to_boundary(boundary_data.is_boundary)

    @kernel function weight_kernel(
            I, J, V, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
            is_boundary, boundary_conditions, normals,
            batch_hermite_datas, global_to_boundary,
            row_offsets, batch_size, N_eval, nmon, k, num_ops, TD,
        )
        batch_idx = @index(Global)
        hermite_data = batch_hermite_datas[batch_idx]
        start_idx, end_idx = calculate_batch_range(batch_idx, batch_size, N_eval)

        # Pre-allocate work arrays for this thread
        n = k + nmon
        A_full = zeros(TD, n, n)
        A = Symmetric(A_full, :U)
        b = _prepare_buffer(ℒrbf, TD, n)
        λ = _prepare_buffer(ℒrbf, TD, n)

        for eval_idx in start_idx:end_idx
            start_pos = row_offsets[eval_idx]
            neighbors = adjl[eval_idx]
            eval_point = eval_points[eval_idx]

            # Classify stencil type
            stype = classify_stencil(
                is_boundary, boundary_conditions, eval_idx, neighbors, global_to_boundary
            )

            if stype isa DirichletStencil
                # Identity row: only diagonal is 1.0
                fill_dirichlet_entry!(I, J, V, eval_idx, start_pos, num_ops)
                continue
            end

            # Reset workspace for reuse
            fill!(A_full, zero(TD))
            fill!(b, zero(TD))

            if stype isa InteriorStencil
                # Standard interior stencil (no boundary points)
                local_data = view(data, neighbors)
                weights = _build_stencil!(
                    λ, A, b, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k
                )
            else  # HermiteStencil
                # Mixed interior/boundary stencil
                update_hermite_stencil_data!(
                    hermite_data, data, neighbors, is_boundary,
                    boundary_conditions, normals, global_to_boundary,
                )
                weights = _build_stencil!(
                    λ, A, b, ℒrbf, ℒmon, hermite_data, eval_point, basis, mon, k
                )
            end

            # Store weights in sparse arrays
            fill_entries!(I, J, V, weights, eval_idx, neighbors, start_pos, k, num_ops)
        end
    end

    kernel! = weight_kernel(device)
    kernel!(
        I, J, V, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        boundary_data.is_boundary, boundary_data.boundary_conditions,
        boundary_data.normals, batch_hermite_datas, global_to_boundary,
        row_offsets, batch_size, N_eval, nmon, k, num_ops, TD;
        ndrange = n_batches, workgroupsize = 1,
    )
    return KernelAbstractions.synchronize(device)
end

# ============================================================================
# Helper Utilities
# ============================================================================

"""Calculate batch index range for kernel execution"""
@inline function calculate_batch_range(batch_idx::Int, batch_size::Int, N_eval::Int)
    start_idx = (batch_idx - 1) * batch_size + 1
    end_idx = min(batch_idx * batch_size, N_eval)
    return start_idx, end_idx
end

"""Fill sparse matrix entries using indexed storage (row_offsets)"""
@inline function fill_entries!(
        I, J, V, weights, eval_idx::Int, neighbors, start_pos::Int, k::Int, num_ops::Int
    )
    return @inbounds for local_idx in 1:k
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

"""Fill Dirichlet identity row for optimized allocation"""
@inline function fill_dirichlet_entry!(I, J, V, eval_idx::Int, start_pos::Int, num_ops::Int)
    I[start_pos] = eval_idx
    J[start_pos] = eval_idx
    return @inbounds for op in 1:num_ops
        V[start_pos, op] = 1.0
    end
end
