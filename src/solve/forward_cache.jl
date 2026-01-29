#=
Forward pass with caching for backward pass of _build_weights differentiation rules.

This builds weights while storing intermediate results needed for the pullback.
=#

using LinearAlgebra: Symmetric
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
