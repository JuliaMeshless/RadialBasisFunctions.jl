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

import RadialBasisFunctions: _build_weights, _build_collocation_matrix!, _build_rhs!
import RadialBasisFunctions: BoundaryData, MonomialBasis, AbstractRadialBasis
import RadialBasisFunctions: Partial, Laplacian

"""
    _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, ℒ)

Forward pass that builds weights while caching intermediate results for backward pass.

Returns: (W, cache) where W is the sparse weight matrix and cache contains
per-stencil factorizations and solutions needed for the pullback.
"""
function _forward_with_cache(
        data::AbstractVector,
        eval_points::AbstractVector,
        adjl::AbstractVector,
        basis::AbstractRadialBasis,
        ℒrbf,
        ℒmon,
        mon::MonomialBasis{Dim, Deg},
        ::Type{ℒType},
    ) where {Dim, Deg, ℒType}
    TD = eltype(first(data))
    k = length(first(adjl))
    nmon = Deg >= 0 ? binomial(Dim + Deg, Deg) : 0
    n = k + nmon
    N_eval = length(eval_points)
    N_data = length(data)

    # Determine number of operators (1 for scalar operators)
    num_ops = 1

    # Allocate COO arrays for sparse matrix
    nnz = k * N_eval
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{TD}(undef, nnz)

    # Allocate stencil caches
    stencil_caches = Vector{StencilForwardCache{TD, Matrix{TD}}}(undef, N_eval)

    # Process each evaluation point
    pos = 1
    for eval_idx in 1:N_eval
        neighbors = adjl[eval_idx]
        eval_point = eval_points[eval_idx]

        # Get local data for this stencil
        local_data = [data[i] for i in neighbors]

        # Build collocation matrix
        A_full = zeros(TD, n, n)
        A = Symmetric(A_full, :U)
        _build_collocation_matrix!(A, local_data, basis, mon, k)

        # Build RHS vector
        b = zeros(TD, n, num_ops)
        b_vec = view(b, :, 1)
        _build_rhs!(b_vec, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k)

        # Solve (symmetric matrix, not positive definite due to zero block)
        λ = Symmetric(A_full, :U) \ b

        # Extract weights (first k entries)
        w = λ[1:k, :]

        # Store in COO format
        for (local_idx, global_idx) in enumerate(neighbors)
            I[pos] = eval_idx
            J[pos] = global_idx
            V[pos] = w[local_idx, 1]
            pos += 1
        end

        # Cache for backward pass - store full symmetric matrix
        A_full_symmetric = copy(A_full)
        # Fill lower triangle from upper
        for j in 1:n
            for i in (j + 1):n
                A_full_symmetric[i, j] = A_full[j, i]
            end
        end
        stencil_caches[eval_idx] = StencilForwardCache(copy(λ), A_full_symmetric, k, nmon)
    end

    # Construct sparse matrix
    W = sparse(I, J, V, N_eval, N_data)

    # Build global cache
    cache = WeightsBuildForwardCache(stencil_caches, k, nmon, num_ops)

    return W, cache
end

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
    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    # Forward pass with caching
    W, cache = _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, Partial)

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

                # Run backward pass for this stencil
                backward_stencil_partial!(
                    Δlocal_data,
                    Δeval_pt,
                    Δw,
                    stencil_cache,
                    collect(1:k),  # Local indices
                    eval_point,
                    local_data,
                    basis,
                    mon,
                    k,
                    ℒ.dim,
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
            NoTangent(),      # function
            NoTangent(),      # ℒ
            [PT(Δdata_raw[i]) for i in 1:N_data],            # data
            [PT(Δeval_points_raw[i]) for i in 1:N_eval],     # eval_points
            NoTangent(),      # adjl (discrete, non-differentiable)
            NoTangent(),      # basis
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
    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    # Forward pass with caching
    W, cache = _forward_with_cache(
        data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, Laplacian
    )

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

                # Run backward pass for this stencil
                backward_stencil_laplacian!(
                    Δlocal_data,
                    Δeval_pt,
                    Δw,
                    stencil_cache,
                    collect(1:k),
                    eval_point,
                    local_data,
                    basis,
                    mon,
                    k,
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
            NoTangent(),      # function
            NoTangent(),      # ℒ
            [PT(Δdata_raw[i]) for i in 1:N_data],            # data
            [PT(Δeval_points_raw[i]) for i in 1:N_eval],     # eval_points
            NoTangent(),      # adjl
            NoTangent(),      # basis
        )
    end

    return W, _build_weights_laplacian_pullback
end
