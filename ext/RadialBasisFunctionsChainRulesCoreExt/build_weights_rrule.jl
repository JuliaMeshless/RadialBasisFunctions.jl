#=
ChainRulesCore rrule for _build_weights function.

This enables differentiating through RBF operator construction w.r.t. point positions,
enabling shape optimization applications.

The rrule defines:
  - Forward pass: build weights with caching for backward pass
  - Pullback: compute gradients w.r.t. data and eval_points from weight cotangents
=#

using LinearAlgebra: Symmetric
using SparseArrays: sparse, SparseMatrixCSC, findnz

"""
    materialize_sparse_tangent(ΔW_raw, W::SparseMatrixCSC)

Convert a potentially wrapped cotangent (Thunk or Tangent{SparseMatrixCSC}) into a concrete sparse matrix.
Mooncake passes Tangent types where only nzval has gradients - m, n, colptr, rowval are NoTangent.
We use the original matrix W's structure and only extract nzval from the tangent.
"""
function materialize_sparse_tangent(ΔW_raw, W::SparseMatrixCSC)
    ΔW = unthunk(ΔW_raw)

    # Handle Tangent{SparseMatrixCSC} from Mooncake/ChainRulesCore
    if ΔW isa ChainRulesCore.Tangent
        b = ChainRulesCore.backing(ΔW)
        # Use original matrix structure, only nzval has gradients
        return SparseMatrixCSC(W.m, W.n, copy(W.colptr), copy(W.rowval), b.nzval)
    end

    return ΔW
end

"""
    extract_stencil_cotangent(ΔW, eval_idx, neighbors, k, num_ops)

Extract the cotangent values for a single stencil from the sparse matrix cotangent.
"""
function extract_stencil_cotangent(
        ΔW::AbstractMatrix{T}, eval_idx::Int, neighbors::Vector{Int}, k::Int, num_ops::Int
    ) where {T}
    Δw = zeros(T, k, num_ops)
    for (local_idx, global_idx) in enumerate(neighbors)
        Δw[local_idx, 1] = ΔW[eval_idx, global_idx]
    end
    return Δw
end

# ============================================================================
# rrule for Partial operator
# ============================================================================

function ChainRulesCore.rrule(
        ::typeof(_build_weights),
        ℒ::Partial,
        data::AbstractVector,
        eval_points::AbstractVector,
        adjl::AbstractVector,
        basis::AbstractRadialBasis,
    )
    # Build monomial basis and apply operator (same as forward pass)
    dim_space = length(first(data))
    mon = MonomialBasis(dim_space, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    # Forward pass with caching
    W, cache = _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, Partial)

    # Get gradient functions for the partial derivative direction
    grad_Lφ_x = grad_applied_partial_wrt_x(basis, ℒ.dim)
    grad_Lφ_xi = grad_applied_partial_wrt_xi(basis, ℒ.dim)

    function _build_weights_partial_pullback(ΔW_raw)
        TD = eltype(first(data))
        PT = eltype(data)  # Point type (e.g., SVector{2,Float64})
        N_data = length(data)
        N_eval = length(eval_points)
        k = cache.k

        ΔW = materialize_sparse_tangent(ΔW_raw, W)

        # Initialize gradient accumulators (use mutable vectors for accumulation)
        Δdata_raw = [zeros(TD, length(first(data))) for _ in 1:N_data]
        Δeval_points_raw = [zeros(TD, length(first(eval_points))) for _ in 1:N_eval]
        Δε_acc = Ref(zero(TD))  # Shape parameter gradient accumulator

        # Process each stencil
        for eval_idx in 1:N_eval
            neighbors = adjl[eval_idx]
            eval_point = eval_points[eval_idx]
            stencil_cache = cache.stencil_caches[eval_idx]

            # Extract cotangent for this stencil
            Δw = extract_stencil_cotangent(ΔW, eval_idx, neighbors, k, cache.num_ops)

            # Check if any non-zero cotangent
            if sum(abs, Δw) > 0
                # Get local data
                local_data = [data[i] for i in neighbors]

                # Create local gradient accumulators
                Δlocal_data = [zeros(TD, length(first(data))) for _ in 1:k]
                Δeval_pt = zeros(TD, length(eval_point))

                # Run backward pass for this stencil (with ε gradient)
                backward_stencil_partial_with_ε!(
                    Δlocal_data,
                    Δeval_pt,
                    Δε_acc,
                    Δw,
                    stencil_cache,
                    collect(1:k),  # Local indices
                    eval_point,
                    local_data,
                    basis,
                    mon,
                    k,
                    ℒ.dim,
                    grad_Lφ_x,
                    grad_Lφ_xi,
                )

                # Accumulate to global gradients
                for (local_idx, global_idx) in enumerate(neighbors)
                    Δdata_raw[global_idx] .+= Δlocal_data[local_idx]
                end
                Δeval_points_raw[eval_idx] .+= Δeval_pt
            end
        end

        # Build basis tangent (only for bases with shape parameter)
        Δbasis = _make_basis_tangent(basis, Δε_acc[])

        # Convert to match input types (required for Mooncake compatibility)
        return (
            NoTangent(),      # function
            NoTangent(),      # ℒ
            [PT(Δdata_raw[i]) for i in 1:N_data],            # data
            [PT(Δeval_points_raw[i]) for i in 1:N_eval],     # eval_points
            NoTangent(),      # adjl (discrete, non-differentiable)
            Δbasis,           # basis
        )
    end

    return W, _build_weights_partial_pullback
end

# ============================================================================
# rrule for Laplacian operator
# ============================================================================

function ChainRulesCore.rrule(
        ::typeof(_build_weights),
        ℒ::Laplacian,
        data::AbstractVector,
        eval_points::AbstractVector,
        adjl::AbstractVector,
        basis::AbstractRadialBasis,
    )
    # Build monomial basis and apply operator
    dim_space = length(first(data))
    mon = MonomialBasis(dim_space, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    # Forward pass with caching
    W, cache = _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, Laplacian)

    # Get gradient functions for the Laplacian
    grad_Lφ_x = grad_applied_laplacian_wrt_x(basis)
    grad_Lφ_xi = grad_applied_laplacian_wrt_xi(basis)

    function _build_weights_laplacian_pullback(ΔW_raw)
        TD = eltype(first(data))
        PT = eltype(data)  # Point type (e.g., SVector{2,Float64})
        N_data = length(data)
        N_eval = length(eval_points)
        k = cache.k

        ΔW = materialize_sparse_tangent(ΔW_raw, W)

        # Initialize gradient accumulators (use mutable vectors for accumulation)
        Δdata_raw = [zeros(TD, length(first(data))) for _ in 1:N_data]
        Δeval_points_raw = [zeros(TD, length(first(eval_points))) for _ in 1:N_eval]
        Δε_acc = Ref(zero(TD))  # Shape parameter gradient accumulator

        # Process each stencil
        for eval_idx in 1:N_eval
            neighbors = adjl[eval_idx]
            eval_point = eval_points[eval_idx]
            stencil_cache = cache.stencil_caches[eval_idx]

            # Extract cotangent for this stencil
            Δw = extract_stencil_cotangent(ΔW, eval_idx, neighbors, k, cache.num_ops)

            # Check if any non-zero cotangent
            if sum(abs, Δw) > 0
                # Get local data
                local_data = [data[i] for i in neighbors]

                # Create local gradient accumulators
                Δlocal_data = [zeros(TD, length(first(data))) for _ in 1:k]
                Δeval_pt = zeros(TD, length(eval_point))

                # Run backward pass for this stencil (with ε gradient)
                backward_stencil_laplacian_with_ε!(
                    Δlocal_data,
                    Δeval_pt,
                    Δε_acc,
                    Δw,
                    stencil_cache,
                    collect(1:k),
                    eval_point,
                    local_data,
                    basis,
                    mon,
                    k,
                    grad_Lφ_x,
                    grad_Lφ_xi,
                )

                # Accumulate to global gradients
                for (local_idx, global_idx) in enumerate(neighbors)
                    Δdata_raw[global_idx] .+= Δlocal_data[local_idx]
                end
                Δeval_points_raw[eval_idx] .+= Δeval_pt
            end
        end

        # Build basis tangent (only for bases with shape parameter)
        Δbasis = _make_basis_tangent(basis, Δε_acc[])

        # Convert to match input types (required for Mooncake compatibility)
        return (
            NoTangent(),      # function
            NoTangent(),      # ℒ
            [PT(Δdata_raw[i]) for i in 1:N_data],            # data
            [PT(Δeval_points_raw[i]) for i in 1:N_eval],     # eval_points
            NoTangent(),      # adjl
            Δbasis,           # basis
        )
    end

    return W, _build_weights_laplacian_pullback
end

# Helper to construct appropriate tangent for different basis types
_make_basis_tangent(::AbstractRadialBasis, Δε) = NoTangent()  # Default for PHS
_make_basis_tangent(::Gaussian, Δε) = Tangent{Gaussian}(; ε = Δε, poly_deg = NoTangent())
_make_basis_tangent(::IMQ, Δε) = Tangent{IMQ}(; ε = Δε, poly_deg = NoTangent())
