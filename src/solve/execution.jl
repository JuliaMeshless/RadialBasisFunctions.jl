using KernelAbstractions
using KernelAbstractions: @kernel, @index, CPU, get_backend

# ============================================================================
# Entry Point
# ============================================================================

"""
    _build_weights(ℒ, data, eval_points, adjl, basis; device = CPU())

Build the stencil weights for operator `ℒ`: apply it to the RBF and monomial bases,
then solve one local system per evaluation point.

Operators needing a different construction route add a method here — see
`Directional` (contracts Jacobian weights with a direction vector) and
`VirtualPartial` (differences two `Regrid` builds).

!!! warning "Signature is load-bearing"
    `ext/RadialBasisFunctionsEnzymeExt` attaches `EnzymeRules.augmented_primal` and
    `reverse` to this exact positional signature. Changing its shape silently drops
    the AD rules and breaks shape-parameter/node-position differentiation.
"""
function _build_weights(ℒ, data, eval_points, adjl, basis; device = CPU())
    mon = MonomialBasis(length(first(data)), basis.poly_deg)
    return build_weights_kernel(
        data, eval_points, adjl, basis, ℒ(basis), ℒ(mon), mon; device = device
    )
end

# ============================================================================
# Operator Arity Helpers (direct Tuple dispatch)
# ============================================================================

_num_ops(::Tuple{Vararg{Any, N}}) where {N} = N
_num_ops(_) = 1

_prepare_buffer(::Tuple{Vararg{Any, N}}, T, n) where {N} = zeros(T, n, N)
_prepare_buffer(_, T, n) = zeros(T, n)

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
    build_weights_kernel(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon;
                         batch_size, device)

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
        mon;
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

    # Allocate ELL weight storage
    vals_list, idx = allocate_ell(device, TD, k, N_eval, num_ops)

    # Calculate batches
    n_batches = ceil(Int, N_eval / batch_size)

    # Launch kernel
    launch_kernel!(
        vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        batch_size, N_eval, n_batches, k, nmon, num_ops, device,
    )

    return _assemble_weights(vals_list, idx, length(data), num_ops)
end

# ============================================================================
# CPU Kernel
# ============================================================================

"""
    launch_kernel!(...)

Launch the parallel CPU kernel for weight computation. Each batch owns its own
workspace and solves one stencil at a time.
"""
function launch_kernel!(
        vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
        batch_size, N_eval, n_batches, k, nmon, num_ops, device,
    )
    TD = eltype(first(data))

    @kernel function weight_kernel(
            vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
            batch_size, N_eval, nmon, k, num_ops, TD,
        )
        batch_idx = @index(Global)
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

            # Reset workspace for reuse
            fill!(A_full, zero(TD))
            fill!(b, zero(TD))

            local_data = view(data, neighbors)
            weights = _build_stencil!(
                λ, A, b, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k
            )

            # Store weights in the ELL columns
            fill_entries!(vals_list, idx, weights, eval_idx, neighbors, k, num_ops)
        end
    end

    kernel! = weight_kernel(device)
    kernel!(
        vals_list, idx, data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
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
