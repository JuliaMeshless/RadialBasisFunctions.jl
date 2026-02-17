#=
Forward pass with caching for backward pass of _build_weights differentiation rules.

This builds weights while storing intermediate results needed for the pullback.
=#

using LinearAlgebra: Symmetric, copytri!
using SparseArrays: sparse

"""
    _forward_with_cache(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon, ℒType)

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

    # Pre-allocate workspace outside the loop
    A_full = zeros(TD, n, n)
    b = zeros(TD, n, num_ops)

    # Process each evaluation point
    pos = 1
    for eval_idx in 1:N_eval
        neighbors = adjl[eval_idx]
        eval_point = eval_points[eval_idx]

        # Use view instead of allocating new vector
        local_data = view(data, neighbors)

        # Reset and reuse pre-allocated collocation matrix
        fill!(A_full, zero(TD))
        A = Symmetric(A_full, :U)
        _build_collocation_matrix!(A, local_data, basis, mon, k)

        # Reset and reuse pre-allocated RHS
        fill!(b, zero(TD))
        b_vec = view(b, :, 1)
        _build_rhs!(b_vec, ℒrbf, ℒmon, local_data, eval_point, basis, mon, k)

        # Solve (symmetric matrix, not positive definite due to zero block)
        λ = Symmetric(A_full, :U) \ b

        # Store in COO format
        @inbounds for (local_idx, global_idx) in enumerate(neighbors)
            I[pos] = eval_idx
            J[pos] = global_idx
            V[pos] = λ[local_idx, 1]
            pos += 1
        end

        # Cache for backward pass - store full symmetric matrix
        A_cached = copy(A_full)
        copytri!(A_cached, 'U')
        stencil_caches[eval_idx] = StencilForwardCache(copy(λ), A_cached, k, nmon)
    end

    # Construct sparse matrix
    W = sparse(I, J, V, N_eval, N_data)

    # Build global cache
    cache = WeightsBuildForwardCache(stencil_caches, k, nmon, num_ops)

    return W, cache
end
