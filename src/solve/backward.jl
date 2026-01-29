#=
Backward pass functions for _build_weights differentiation rules.

The backward pass computes:
  Given Δw (cotangent of weights), compute Δdata and Δeval_points

Key steps per stencil:
1. Pad cotangent: Δλ = [Δw; 0]
2. Solve adjoint: η = A⁻ᵀ Δλ
3. Compute: ΔA = -η λᵀ, Δb = η
4. Chain through RHS: accumulate to Δeval_point and Δdata[neighbors]
5. Chain through collocation: accumulate to Δdata[neighbors]
=#

using LinearAlgebra: dot, mul!, axpy!

"""
    backward_linear_solve!(ΔA, Δb, Δw, cache)

Compute cotangents of collocation matrix A and RHS vector b
from cotangent of weights Δw.

Given: Aλ = b, w = λ[1:k]
We have: Δλ = [Δw; 0]  (padded with zeros for monomial part)

Using implicit function theorem:
  η = A⁻ᵀ Δλ
  ΔA = -η λᵀ
  Δb = η
"""
function backward_linear_solve!(
        ΔA::AbstractMatrix{T},
        Δb::AbstractVecOrMat{T},
        Δw::AbstractVecOrMat{T},
        cache::StencilForwardCache{T},
    ) where {T}
    k = cache.k
    nmon = cache.nmon
    n = k + nmon
    num_ops = size(cache.lambda, 2)

    # Pad Δw with zeros for monomial part
    Δλ = zeros(T, n, num_ops)
    Δλ[1:k, :] .= Δw

    # Solve adjoint system: A'η = Δλ
    # The matrix is symmetric, so A' = A
    η = cache.A_mat \ Δλ

    # ΔA = -η * λᵀ (outer product, accumulated across operators)
    # Use BLAS mul! for O(n²) instead of scalar triple loop
    fill!(ΔA, zero(T))
    for op_idx in 1:num_ops
        η_vec = view(η, :, op_idx)
        λ_vec = view(cache.lambda, :, op_idx)
        # Rank-1 update: ΔA -= η_vec * λ_vec' (outer product)
        mul!(ΔA, η_vec, λ_vec', -one(T), one(T))
    end

    # Δb = η
    Δb .= η

    return nothing
end

"""
    backward_collocation!(Δdata, ΔA, neighbors, data, basis, mon, k)

Chain rule through collocation matrix construction.

The collocation matrix has structure:
  A[i,j] = φ(xi, xj)     for i,j ≤ k (RBF block)
  A[i,k+j] = pⱼ(xi)      for i ≤ k (polynomial block)

For RBF block (using ∇φ from existing basis_rules):
  Δxi += ΔA[i,j] * ∇φ(xi, xj)
  Δxj -= ΔA[i,j] * ∇φ(xi, xj)  (by symmetry of φ(x-y))

For polynomial block:
  Δxi += ΔA[i,k+j] * ∇pⱼ(xi)

Note: A is symmetric, so we need to handle both triangles.
"""
function backward_collocation!(
        Δdata::Vector{Vector{T}},
        ΔA::AbstractMatrix{T},
        neighbors::Vector{Int},
        data::AbstractVector,
        basis::AbstractRadialBasis,
        mon::MonomialBasis{Dim, Deg},
        k::Int,
    ) where {T, Dim, Deg}
    grad_φ = ∇(basis)
    n = k + binomial(Dim + Deg, Deg)

    # RBF block: accumulate gradients from symmetric matrix
    # Only upper triangle stored, but gradients flow both ways
    @inbounds for j in 1:k
        xj = data[neighbors[j]]
        Δdata_j = Δdata[neighbors[j]]
        for i in 1:(j - 1)  # Skip diagonal (i == j) since φ(x,x) = 0 always, no gradient contribution
            xi = data[neighbors[i]]
            Δdata_i = Δdata[neighbors[i]]

            # Get gradient of basis function
            ∇φ_ij = grad_φ(xi, xj)

            # ΔA[i,j] contributes to both Δxi and Δxj
            # For symmetric matrix, ΔA[i,j] == ΔA[j,i] conceptually
            # We need to sum contributions from both triangles
            scale = ΔA[i, j] + ΔA[j, i]

            # φ depends on xi - xj, so: ∂φ/∂xi = ∇φ, ∂φ/∂xj = -∇φ
            # In-place accumulation avoids broadcast allocation
            for d in eachindex(∇φ_ij)
                Δdata_i[d] += scale * ∇φ_ij[d]
                Δdata_j[d] -= scale * ∇φ_ij[d]
            end
        end
    end

    # Polynomial block: A[i, k+j] = pⱼ(xi)
    # Need gradient of monomial basis w.r.t. xi
    if Deg > -1
        nmon = binomial(Dim + Deg, Deg)
        ∇p = zeros(T, nmon, Dim)
        # Hoist functor construction outside loop
        ∇mon = ∇(mon)

        @inbounds for i in 1:k
            xi = data[neighbors[i]]
            ∇mon(∇p, xi)
            Δdata_i = Δdata[neighbors[i]]

            # Accumulate gradient from polynomial block
            for j in 1:nmon
                # ΔA[i, k+j] contributes to Δxi via ∇pⱼ
                # Also ΔA[k+j, i] from transpose block
                scale = ΔA[i, k + j] + ΔA[k + j, i]
                # Use axpy! pattern for in-place accumulation without broadcast allocation
                for d in 1:Dim
                    Δdata_i[d] += scale * ∇p[j, d]
                end
            end
        end
    end

    return nothing
end

"""
    backward_rhs_partial!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, dim, k, grad_Lφ_x, grad_Lφ_xi)

Chain rule through RHS vector construction for Partial operator.

RHS structure:
  b[i] = ℒφ(eval_point, xi)  for i = 1:k
  b[k+j] = ℒpⱼ(eval_point)   for j = 1:nmon

For RBF section, we need:
  ∂/∂eval_point [ℒφ(eval_point, xi)]
  ∂/∂xi [ℒφ(eval_point, xi)]

For polynomial section, we need:
  ∂/∂eval_point [ℒpⱼ(eval_point)]
"""
function backward_rhs_partial!(
        Δdata::Vector{Vector{T}},
        Δeval_point::Vector{T},
        Δb::AbstractVecOrMat{T},
        neighbors::Vector{Int},
        eval_point,
        data::AbstractVector,
        basis::AbstractRadialBasis,
        dim::Int,
        k::Int,
        grad_Lφ_x,
        grad_Lφ_xi,
    ) where {T}
    num_ops = size(Δb, 2)

    # RBF section: b[i] = ∂φ/∂x[dim](eval_point, xi)
    @inbounds for i in 1:k
        xi = data[neighbors[i]]
        Δdata_i = Δdata[neighbors[i]]

        # Gradient w.r.t. eval_point and xi
        ∇Lφ_x = grad_Lφ_x(eval_point, xi)
        ∇Lφ_xi = grad_Lφ_xi(eval_point, xi)

        # Accumulate across operators with in-place scalar ops
        for op_idx in 1:num_ops
            Δb_val = Δb[i, op_idx]
            for d in eachindex(∇Lφ_x)
                Δeval_point[d] += Δb_val * ∇Lφ_x[d]
                Δdata_i[d] += Δb_val * ∇Lφ_xi[d]
            end
        end
    end

    return nothing
end

"""
    backward_rhs_laplacian!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, k, grad_Lφ_x, grad_Lφ_xi)

Chain rule through RHS for Laplacian operator.
"""
function backward_rhs_laplacian!(
        Δdata::Vector{Vector{T}},
        Δeval_point::Vector{T},
        Δb::AbstractVecOrMat{T},
        neighbors::Vector{Int},
        eval_point,
        data::AbstractVector,
        basis::AbstractRadialBasis,
        k::Int,
        grad_Lφ_x,
        grad_Lφ_xi,
    ) where {T}
    num_ops = size(Δb, 2)

    # RBF section: b[i] = ∇²φ(eval_point, xi)
    @inbounds for i in 1:k
        xi = data[neighbors[i]]
        Δdata_i = Δdata[neighbors[i]]

        # Gradient w.r.t. eval_point and xi
        ∇Lφ_x = grad_Lφ_x(eval_point, xi)
        ∇Lφ_xi = grad_Lφ_xi(eval_point, xi)

        # Accumulate across operators with in-place scalar ops
        for op_idx in 1:num_ops
            Δb_val = Δb[i, op_idx]
            for d in eachindex(∇Lφ_x)
                Δeval_point[d] += Δb_val * ∇Lφ_x[d]
                Δdata_i[d] += Δb_val * ∇Lφ_xi[d]
            end
        end
    end

    return nothing
end

"""
    backward_stencil_partial!(Δdata, Δeval_point, Δw, cache, neighbors, eval_point, data, basis, mon, k, dim, grad_Lφ_x, grad_Lφ_xi)

Complete backward pass for a single stencil with Partial operator.

Combines:
1. backward_linear_solve! - compute ΔA, Δb from Δw
2. backward_collocation! - chain ΔA to Δdata
3. backward_rhs_partial! - chain Δb to Δdata and Δeval_point
"""
function backward_stencil_partial!(
        Δdata::Vector{Vector{T}},
        Δeval_point::Vector{T},
        Δw::AbstractVecOrMat{T},
        cache::StencilForwardCache{T},
        neighbors::Vector{Int},
        eval_point,
        data::AbstractVector,
        basis::AbstractRadialBasis,
        mon::MonomialBasis{Dim, Deg},
        k::Int,
        dim::Int,
        grad_Lφ_x,
        grad_Lφ_xi,
    ) where {T, Dim, Deg}
    n = k + cache.nmon

    # Allocate workspace for ΔA and Δb
    ΔA = zeros(T, n, n)
    Δb = zeros(T, n, size(Δw, 2))

    # Step 1: Backprop through linear solve
    backward_linear_solve!(ΔA, Δb, Δw, cache)

    # Step 2: Backprop through collocation matrix
    backward_collocation!(Δdata, ΔA, neighbors, data, basis, mon, k)

    # Step 3: Backprop through RHS
    backward_rhs_partial!(
        Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, dim, k, grad_Lφ_x, grad_Lφ_xi
    )

    return nothing
end

"""
    backward_stencil_laplacian!(Δdata, Δeval_point, Δw, cache, neighbors, eval_point, data, basis, mon, k, grad_Lφ_x, grad_Lφ_xi)

Complete backward pass for a single stencil with Laplacian operator.
"""
function backward_stencil_laplacian!(
        Δdata::Vector{Vector{T}},
        Δeval_point::Vector{T},
        Δw::AbstractVecOrMat{T},
        cache::StencilForwardCache{T},
        neighbors::Vector{Int},
        eval_point,
        data::AbstractVector,
        basis::AbstractRadialBasis,
        mon::MonomialBasis{Dim, Deg},
        k::Int,
        grad_Lφ_x,
        grad_Lφ_xi,
    ) where {T, Dim, Deg}
    n = k + cache.nmon

    # Allocate workspace for ΔA and Δb
    ΔA = zeros(T, n, n)
    Δb = zeros(T, n, size(Δw, 2))

    # Step 1: Backprop through linear solve
    backward_linear_solve!(ΔA, Δb, Δw, cache)

    # Step 2: Backprop through collocation matrix
    backward_collocation!(Δdata, ΔA, neighbors, data, basis, mon, k)

    # Step 3: Backprop through RHS
    backward_rhs_laplacian!(
        Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, k, grad_Lφ_x, grad_Lφ_xi
    )

    return nothing
end
