using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU
using LinearAlgebra: Symmetric

# ============================================================================
# Memory Allocation
# ============================================================================

"""
    allocate_ell(backend, TD, k, N_eval, num_ops)

Allocate ELL (stencil-wise) weight storage on `backend`: one dense `k × N_eval` value
matrix per operator component and a shared `k × N_eval` `Int32` neighbor-index matrix.
"""
function allocate_ell(backend, TD, k::Int, N_eval::Int, num_ops::Int)
    vals_list = [KernelAbstractions.allocate(backend, TD, k, N_eval) for _ in 1:num_ops]
    idx = KernelAbstractions.allocate(backend, Int32, k, N_eval)
    return vals_list, idx
end

"""
    construct_global_to_boundary(is_boundary)

Construct mapping from global indices to boundary-only indices.
For boundary points: global_to_boundary[i] = boundary array index
For interior points: global_to_boundary[i] = 0 (sentinel)
"""
function construct_global_to_boundary(is_boundary::AbstractVector{Bool})
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
# Weight Assembly
# ============================================================================

"""
    _assemble_weights(vals_list, idx, N_data, num_ops)

Wrap the filled ELL buffers into [`StencilWeights`](@ref) — one per operator component,
all sharing the same neighbor-index matrix.
"""
function _assemble_weights(vals_list, idx, N_data, num_ops)
    first_w = StencilWeights(vals_list[1], idx, N_data)
    if num_ops == 1
        return first_w
    else
        # All components share one structure object: one idx matrix AND one transpose map
        return ntuple(
            o -> o == 1 ? first_w : _shared_component(first_w, vals_list[o]), num_ops
        )
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
        throw(
            ArgumentError(
                "GPU weight computation is not yet supported. " *
                    "A GPU-kernel-compatible dense solver for stencil matrices is required. " *
                    "See https://github.com/JuliaMeshless/RadialBasisFunctions.jl/issues/88"
            )
        )
    end

    TD = eltype(first(data))
    k = length(first(adjl))
    if any(neighbors -> length(neighbors) != k, adjl)
        throw(
            ArgumentError(
                "stencil-wise (ELL) weight storage requires a uniform stencil size; " *
                    "adjl contains stencils of differing lengths"
            )
        )
    end
    nmon = binomial(length(first(data)) + basis.poly_deg, basis.poly_deg)
    num_ops = _num_ops(ℒrbf)
    N_eval = length(eval_points)

    global_to_boundary = construct_global_to_boundary(boundary_data.is_boundary)

    # Allocate ELL weight storage
    vals_list, idx = allocate_ell(device, TD, k, N_eval, num_ops)

    # Calculate batches
    n_batches = ceil(Int, N_eval / batch_size)

    # Launch kernel
    launch_kernel!(
        vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        boundary_data, global_to_boundary, batch_size, N_eval, n_batches,
        k, nmon, num_ops, device,
    )

    return _assemble_weights(vals_list, idx, length(data), num_ops)
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
        vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        boundary_data::BoundaryData, global_to_boundary,
        batch_size, N_eval, n_batches, k, nmon, num_ops, device,
    )
    TD = eltype(first(data))
    dim = length(first(data))

    # Pre-allocate Hermite workspace for each batch (includes polynomial workspace)
    batch_hermite_datas = [HermiteStencilData{TD}(k, dim, nmon) for _ in 1:n_batches]

    @kernel function weight_kernel(
            vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
            is_boundary, boundary_conditions, normals,
            batch_hermite_datas, global_to_boundary,
            batch_size, N_eval, nmon, k, num_ops, TD,
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
            neighbors = adjl[eval_idx]
            eval_point = eval_points[eval_idx]

            # Classify stencil type
            stype = classify_stencil(
                is_boundary, boundary_conditions, eval_idx, neighbors, global_to_boundary
            )

            if stype isa DirichletStencil
                # Identity row: weight 1 at slot 1, zero pads sharing the same index
                fill_dirichlet_column!(vals_list, idx, eval_idx, k, num_ops)
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
                    boundary_conditions, normals, global_to_boundary, eval_point,
                )
                weights = _build_stencil!(
                    λ, A, b, ℒrbf, ℒmon, hermite_data, eval_point, basis, mon, k
                )
            end

            # Store weights in the ELL columns
            fill_entries!(vals_list, idx, weights, eval_idx, neighbors, k, num_ops)
        end
    end

    kernel! = weight_kernel(device)
    kernel!(
        vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        boundary_data.is_boundary, boundary_data.boundary_conditions,
        boundary_data.normals, batch_hermite_datas, global_to_boundary,
        batch_size, N_eval, nmon, k, num_ops, TD;
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

"""Write one stencil's weights into its ELL column"""
@inline function fill_entries!(
        vals_list, idx, weights, eval_idx::Int, neighbors, k::Int, num_ops::Int
    )
    return @inbounds for local_idx in 1:k
        idx[local_idx, eval_idx] = Int32(neighbors[local_idx])
        if num_ops == 1
            vals_list[1][local_idx, eval_idx] = weights[local_idx]
        else
            for op in 1:num_ops
                vals_list[op][local_idx, eval_idx] = weights[local_idx, op]
            end
        end
    end
end

"""
Fill a Dirichlet identity column: weight 1 at slot 1, zero pads elsewhere. All slots
carry the eval point's own index (in-bounds; Dirichlet stencils require
`eval_points === data`), so duplicate-combining `sparse` conversion collapses the column
to a single identity entry.
"""
@inline function fill_dirichlet_column!(vals_list, idx, eval_idx::Int, k::Int, num_ops::Int)
    @inbounds for local_idx in 1:k
        idx[local_idx, eval_idx] = Int32(eval_idx)
        for op in 1:num_ops
            vals_list[op][local_idx, eval_idx] = zero(eltype(vals_list[op]))
        end
    end
    return @inbounds for op in 1:num_ops
        vals_list[op][1, eval_idx] = one(eltype(vals_list[op]))
    end
end
