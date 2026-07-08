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

"""
    BackwardWorkspace{T,P,M,V}

Per-pass scratch for the `_build_weights` backward pass, allocated once and reused across
every stencil to eliminate per-stencil heap allocations. All buffer shapes are fixed and
uniform for a pass (`n = k + nmon`, `num_ops`, `dim_space`), so they are sized once from the
forward cache.

Container fields are type-parameterized (`M`, `V`) rather than hard-coded `Matrix`/`Vector`,
matching [`StencilForwardCache`](@ref); the constructor builds concrete CPU arrays and lets
the parameters be inferred.

- `ΔA`: `n × n` adjoint of the collocation matrix
- `Δb`: `n × num_ops` adjoint of the RHS
- `Δλ`: `n × num_ops` padded stencil cotangent (overwritten by the adjoint solve)
- `∇p`: `nmon × dim_space` monomial-gradient scratch
- `Δw`: `k × num_ops` extracted stencil cotangent
- `Δlocal_data`: `k` buffers of length `dim_space` (local data-point cotangents)
- `Δeval_pt`: length `dim_space` eval-point cotangent
- `local_data`: `k` gathered stencil points
- `local_idx`: `[1, …, k]`, the constant local neighbor indices
"""
struct BackwardWorkspace{T, P, M <: AbstractMatrix{T}, V <: AbstractVector{T}}
    ΔA::M
    Δb::M
    Δλ::M
    ∇p::M
    Δw::M
    Δlocal_data::Vector{V}
    Δeval_pt::V
    local_data::Vector{P}
    local_idx::Vector{Int}
end

function BackwardWorkspace(cache::WeightsBuildForwardCache{T}, data::AbstractVector) where {T}
    k = cache.k
    nmon = cache.nmon
    num_ops = cache.num_ops
    n = k + nmon
    dim_space = length(first(data))
    P = eltype(data)
    return BackwardWorkspace{T, P, Matrix{T}, Vector{T}}(
        zeros(T, n, n),
        zeros(T, n, num_ops),
        zeros(T, n, num_ops),
        zeros(T, nmon, dim_space),
        zeros(T, k, num_ops),
        [zeros(T, dim_space) for _ in 1:k],
        zeros(T, dim_space),
        Vector{P}(undef, k),
        collect(1:k),
    )
end
