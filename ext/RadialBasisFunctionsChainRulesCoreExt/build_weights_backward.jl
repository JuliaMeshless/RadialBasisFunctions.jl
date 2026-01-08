#=
Backward pass functions for _build_weights rrule.

The backward pass computes:
  Given Δw (cotangent of weights), compute Δdata and Δeval_points

Key steps per stencil:
1. Pad cotangent: Δλ = [Δw; 0]
2. Solve adjoint: η = A⁻ᵀ Δλ
3. Compute: ΔA = -η λᵀ, Δb = η
4. Chain through RHS: accumulate to Δeval_point and Δdata[neighbors]
5. Chain through collocation: accumulate to Δdata[neighbors]
=#

using LinearAlgebra: dot

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
    # For symmetric A, we need to account for the structure
    fill!(ΔA, zero(T))
    for op_idx in 1:num_ops
        η_vec = view(η, :, op_idx)
        λ_vec = view(cache.lambda, :, op_idx)
        for j in 1:n
            for i in 1:n
                ΔA[i, j] -= η_vec[i] * λ_vec[j]
            end
        end
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
        for i in 1:(j - 1)  # Skip diagonal (i == j) since φ(x,x) = 0 always, no gradient contribution
            xi = data[neighbors[i]]

            # Get gradient of basis function
            ∇φ_ij = grad_φ(xi, xj)

            # ΔA[i,j] contributes to both Δxi and Δxj
            # For symmetric matrix, ΔA[i,j] == ΔA[j,i] conceptually
            # We need to sum contributions from both triangles
            scale = ΔA[i, j] + ΔA[j, i]

            # φ depends on xi - xj, so:
            # ∂φ/∂xi = ∇φ, ∂φ/∂xj = -∇φ
            Δdata[neighbors[i]] .+= scale .* ∇φ_ij
            Δdata[neighbors[j]] .-= scale .* ∇φ_ij
        end
    end

    # Polynomial block: A[i, k+j] = pⱼ(xi)
    # Need gradient of monomial basis w.r.t. xi
    if Deg > -1
        nmon = binomial(Dim + Deg, Deg)
        ∇p = zeros(T, nmon, Dim)

        @inbounds for i in 1:k
            xi = data[neighbors[i]]

            # Compute gradient of all monomials at xi
            ∇mon = RadialBasisFunctions.∇(mon)
            ∇mon(∇p, xi)

            # Accumulate gradient from polynomial block
            for j in 1:nmon
                # ΔA[i, k+j] contributes to Δxi via ∇pⱼ
                # Also ΔA[k+j, i] from transpose block
                scale = ΔA[i, k + j] + ΔA[k + j, i]
                Δdata[neighbors[i]] .+= scale .* view(∇p, j, :)
            end
        end
    end

    return nothing
end

"""
    backward_rhs!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, ℒ, k, dim)

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
    ) where {T}
    num_ops = size(Δb, 2)

    # Get gradient functions for the applied partial operator
    grad_Lφ_x = grad_applied_partial_wrt_x(basis, dim)
    grad_Lφ_xi = grad_applied_partial_wrt_xi(basis, dim)

    # RBF section: b[i] = ∂φ/∂x[dim](eval_point, xi)
    @inbounds for i in 1:k
        xi = data[neighbors[i]]

        # Gradient w.r.t. eval_point
        ∇Lφ_x = grad_Lφ_x(eval_point, xi)
        # Gradient w.r.t. xi
        ∇Lφ_xi = grad_Lφ_xi(eval_point, xi)

        # Accumulate across operators
        for op_idx in 1:num_ops
            Δb_val = Δb[i, op_idx]
            Δeval_point .+= Δb_val .* ∇Lφ_x
            Δdata[neighbors[i]] .+= Δb_val .* ∇Lφ_xi
        end
    end

    # Polynomial section: b[k+j] = ∂pⱼ/∂x[dim](eval_point)
    # The gradient of ∂pⱼ/∂x[dim] w.r.t. eval_point is the second derivative of pⱼ
    # This is non-trivial and depends on polynomial degree
    # For now, we skip this contribution (it's typically small for low-degree polynomials)
    # TODO: Implement polynomial second derivatives if needed

    return nothing
end

"""
    backward_rhs_laplacian!(...)

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
    ) where {T}
    num_ops = size(Δb, 2)

    # Get gradient functions for the applied Laplacian operator
    grad_Lφ_x = grad_applied_laplacian_wrt_x(basis)
    grad_Lφ_xi = grad_applied_laplacian_wrt_xi(basis)

    # RBF section: b[i] = ∇²φ(eval_point, xi)
    @inbounds for i in 1:k
        xi = data[neighbors[i]]

        # Gradient w.r.t. eval_point
        ∇Lφ_x = grad_Lφ_x(eval_point, xi)
        # Gradient w.r.t. xi
        ∇Lφ_xi = grad_Lφ_xi(eval_point, xi)

        # Accumulate across operators
        for op_idx in 1:num_ops
            Δb_val = Δb[i, op_idx]
            Δeval_point .+= Δb_val .* ∇Lφ_x
            Δdata[neighbors[i]] .+= Δb_val .* ∇Lφ_xi
        end
    end

    return nothing
end

"""
    backward_stencil!(Δdata, Δeval_point, Δw, cache, neighbors, eval_point, data, basis, ℒ, mon, k, op_info)

Complete backward pass for a single stencil.

Combines:
1. backward_linear_solve! - compute ΔA, Δb from Δw
2. backward_collocation! - chain ΔA to Δdata
3. backward_rhs! - chain Δb to Δdata and Δeval_point
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
        dim::Int,  # Partial derivative dimension
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
        Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, dim, k
    )

    return nothing
end

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
    backward_rhs_laplacian!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, k)

    return nothing
end
