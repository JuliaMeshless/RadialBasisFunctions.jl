#=
Cache types for storing forward pass results needed by the backward pass
of _build_weights rrule.

The key data needed:
- lambda: Full solution vector for each stencil (k+nmon size)
- A_mat: The collocation matrix (stored for A⁻ᵀ solves in backward pass)

Note: We store the matrix directly rather than a factorization because
the RBF collocation matrix with polynomial augmentation is symmetric but
NOT positive definite (has zero blocks), so Cholesky doesn't work.
=#

using LinearAlgebra: lu, LU

"""
    StencilForwardCache{T}

Per-stencil storage from forward pass needed for backward pass.

- `lambda`: Full solution vector (k+nmon) × num_ops from solving Aλ = b
- `A_mat`: The symmetric collocation matrix (stored for backprop)
- `k`: Number of RBF neighbors in stencil
- `nmon`: Number of monomial basis functions
"""
struct StencilForwardCache{T,M<:AbstractMatrix{T}}
    lambda::M          # (k+nmon) × num_ops solution
    A_mat::Matrix{T}   # Collocation matrix (dense, for A⁻ᵀ solve)
    k::Int
    nmon::Int
end

"""
    WeightsBuildForwardCache{T, TD}

Global cache storing all stencil results and references to inputs.

- `stencil_caches`: Vector of StencilForwardCache, one per evaluation point
- `k`: Stencil size (number of neighbors)
- `nmon`: Number of monomial basis functions
- `num_ops`: Number of operators (1 for scalar, D for gradient)
"""
struct WeightsBuildForwardCache{T}
    stencil_caches::Vector{StencilForwardCache{T,Matrix{T}}}
    k::Int
    nmon::Int
    num_ops::Int
end
