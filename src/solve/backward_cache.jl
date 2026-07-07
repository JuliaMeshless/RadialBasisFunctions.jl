#=
Cache types for storing forward pass results needed by the backward pass
of _build_weights differentiation rules.

The key data needed:
- lambda: Full solution vector for each stencil (k+nmon size)
- A_fact: Bunch-Kaufman factorization of the collocation matrix (for A⁻ᵀ solves)

Note: The RBF collocation matrix with polynomial augmentation is symmetric but
NOT positive definite (has zero blocks), so Cholesky is impossible. Bunch-Kaufman
handles symmetric indefinite systems, and caching the factorization lets the
backward pass solve in O(n²) instead of refactorizing at O(n³) per stencil.
=#

using LinearAlgebra: Factorization

"""
    StencilForwardCache{T}

Per-stencil storage from forward pass needed for backward pass.

- `lambda`: Full solution vector (k+nmon) × num_ops from solving Aλ = b
- `A_fact`: Bunch-Kaufman factorization of the symmetric collocation matrix
- `k`: Number of RBF neighbors in stencil
- `nmon`: Number of monomial basis functions
"""
struct StencilForwardCache{T, M <: AbstractMatrix{T}, F <: Factorization{T}}
    lambda::M          # (k+nmon) × num_ops solution
    A_fact::F          # Factorized collocation matrix (for A⁻ᵀ solve)
    k::Int
    nmon::Int
end

"""
    WeightsBuildForwardCache{T}

Global cache storing all stencil results and references to inputs.

- `stencil_caches`: Vector of StencilForwardCache, one per evaluation point
- `k`: Stencil size (number of neighbors)
- `nmon`: Number of monomial basis functions
- `num_ops`: Number of operators (1 for scalar, D for gradient)
"""
struct WeightsBuildForwardCache{T, C <: StencilForwardCache{T}}
    stencil_caches::Vector{C}
    k::Int
    nmon::Int
    num_ops::Int
end
