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

using LinearAlgebra: ldiv!, mul!

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
    # The matrix is symmetric, so A' = A; reuse the cached factorization (O(n²))
    η = ldiv!(cache.A_fact, Δλ)

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
                # In-place accumulation without broadcast allocation
                for d in 1:Dim
                    Δdata_i[d] += scale * ∇p[j, d]
                end
            end
        end
    end

    return nothing
end

"""
    backward_rhs!(Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, k, grad_Lφ_x, grad_Lφ_xi; poly_backward!=nothing)

Chain rule through RHS vector construction for any operator.

RBF section (shared by all operators):
  b[i] = ℒφ(eval_point, xi) → accumulate ∂/∂eval_point and ∂/∂xi

Polynomial section (Partial only — Laplacian gives constants, no gradient):
  b[k+j] = ℒpⱼ(eval_point) → passed as optional `poly_backward!` closure
"""
function backward_rhs!(
        Δdata::Vector{Vector{T}},
        Δeval_point::Vector{T},
        Δb::AbstractVecOrMat{T},
        neighbors::Vector{Int},
        eval_point,
        data::AbstractVector,
        basis::AbstractRadialBasis,
        k::Int,
        grad_Lφ_x,
        grad_Lφ_xi;
        poly_backward!::F = nothing,
    ) where {T, F}
    num_ops = size(Δb, 2)
    n = size(Δb, 1)
    nmon = n - k

    # RBF section: b[i] = ℒφ(eval_point, xi)
    @inbounds for i in 1:k
        xi = data[neighbors[i]]
        Δdata_i = Δdata[neighbors[i]]

        ∇Lφ_x = grad_Lφ_x(eval_point, xi)
        ∇Lφ_xi = grad_Lφ_xi(eval_point, xi)

        for op_idx in 1:num_ops
            Δb_val = Δb[i, op_idx]
            for d in eachindex(∇Lφ_x)
                Δeval_point[d] += Δb_val * ∇Lφ_x[d]
                Δdata_i[d] += Δb_val * ∇Lφ_xi[d]
            end
        end
    end

    # Polynomial section (only for Partial)
    if poly_backward! !== nothing && nmon > 0
        poly_backward!(Δeval_point, Δb, k, nmon, num_ops)
    end

    return nothing
end

"""
    _backward_partial_polynomial_section!(Δeval_point, Δb, k, nmon, dim, num_ops)

Backward pass through the polynomial section of the RHS for Partial operator.

For monomials in 2D with poly_deg=2 (1, x, y, xy, x², y²):
  ∂/∂x gives: 0, 1, 0, y, 2x, 0

The gradients of these w.r.t. eval_point are:
  ∂(0)/∂(x,y) = (0, 0)
  ∂(1)/∂(x,y) = (0, 0)
  ∂(0)/∂(x,y) = (0, 0)
  ∂(y)/∂(x,y) = (0, 1)  -> b[4] contributes to Δeval_point[2]
  ∂(2x)/∂(x,y) = (2, 0) -> b[5] contributes 2 to Δeval_point[1]
  ∂(0)/∂(x,y) = (0, 0)

This is equivalent to computing the mixed second derivatives ∂²pⱼ/∂x[dim]∂x[d].
"""
function _backward_partial_polynomial_section!(
        Δeval_point::Vector{T},
        Δb::AbstractVecOrMat{T},
        k::Int,
        nmon::Int,
        dim::Int,
        num_ops::Int,
    ) where {T}
    D = length(Δeval_point)

    # The contribution depends on spatial dimension and polynomial degree
    # For poly_deg=2, we have known patterns of non-zero second derivatives
    if D == 1
        _backward_partial_poly_1d!(Δeval_point, Δb, k, nmon, dim, num_ops)
    elseif D == 2
        _backward_partial_poly_2d!(Δeval_point, Δb, k, nmon, dim, num_ops)
    elseif D == 3
        _backward_partial_poly_3d!(Δeval_point, Δb, k, nmon, dim, num_ops)
    else
        throw(
            ArgumentError(
                "Polynomial backward pass not implemented for D=$D (only D=1,2,3 supported)",
            ),
        )
    end
    return nothing
end

"""Backward pass for polynomial section in 1D."""
function _backward_partial_poly_1d!(
        Δeval_point::Vector{T},
        Δb::AbstractVecOrMat{T},
        k::Int,
        nmon::Int,
        dim::Int,
        num_ops::Int,
    ) where {T}
    # 1D monomials up to degree 2: 1, x, x²
    # ∂/∂x gives: 0, 1, 2x
    # Second derivatives ∂²/∂x²: 0, 0, 2
    # Only x² term contributes, at index k+3 (if nmon >= 3)
    if nmon >= 3
        @inbounds for op_idx in 1:num_ops
            Δeval_point[1] += Δb[k + 3, op_idx] * 2
        end
    end
    return nothing
end

"""Backward pass for polynomial section in 2D."""
function _backward_partial_poly_2d!(
        Δeval_point::Vector{T},
        Δb::AbstractVecOrMat{T},
        k::Int,
        nmon::Int,
        dim::Int,
        num_ops::Int,
    ) where {T}
    # 2D monomials with poly_deg=2: 1, x, y, xy, x², y² (nmon=6)
    # 2D monomials with poly_deg=1: 1, x, y (nmon=3)
    # 2D monomials with poly_deg=0: 1 (nmon=1)

    if nmon < 4
        # poly_deg <= 1: all second derivatives are zero
        return nothing
    end

    # For poly_deg=2 (nmon=6):
    # ∂/∂x gives: 0, 1, 0, y, 2x, 0
    # ∂/∂y gives: 0, 0, 1, x, 0, 2y
    #
    # Second derivatives for ∂/∂x (dim=1):
    #   ∂(y)/∂y = 1 at index k+4, contributes to Δeval_point[2]
    #   ∂(2x)/∂x = 2 at index k+5, contributes to Δeval_point[1]
    #
    # Second derivatives for ∂/∂y (dim=2):
    #   ∂(x)/∂x = 1 at index k+4, contributes to Δeval_point[1]
    #   ∂(2y)/∂y = 2 at index k+6, contributes to Δeval_point[2]

    if dim == 1  # ∂/∂x operator
        @inbounds for op_idx in 1:num_ops
            Δeval_point[2] += Δb[k + 4, op_idx]  # from xy term: ∂(y)/∂y = 1
            Δeval_point[1] += Δb[k + 5, op_idx] * 2  # from x² term: ∂(2x)/∂x = 2
        end
    elseif dim == 2  # ∂/∂y operator
        @inbounds for op_idx in 1:num_ops
            Δeval_point[1] += Δb[k + 4, op_idx]  # from xy term: ∂(x)/∂x = 1
            Δeval_point[2] += Δb[k + 6, op_idx] * 2  # from y² term: ∂(2y)/∂y = 2
        end
    end

    return nothing
end

"""Backward pass for polynomial section in 3D."""
function _backward_partial_poly_3d!(
        Δeval_point::Vector{T},
        Δb::AbstractVecOrMat{T},
        k::Int,
        nmon::Int,
        dim::Int,
        num_ops::Int,
    ) where {T}
    # 3D monomials with poly_deg=2: 1, x, y, z, xy, xz, yz, x², y², z² (nmon=10)
    # 3D monomials with poly_deg=1: 1, x, y, z (nmon=4)

    if nmon < 5
        # poly_deg <= 1: all second derivatives are zero
        return nothing
    end

    # For poly_deg=2 (nmon=10):
    # Monomial order: 1, x, y, z, xy, xz, yz, x², y², z²
    #                 1  2  3  4  5   6   7   8   9   10
    #
    # ∂/∂x gives: 0, 1, 0, 0, y, z, 0, 2x, 0, 0
    # ∂/∂y gives: 0, 0, 1, 0, x, 0, z, 0, 2y, 0
    # ∂/∂z gives: 0, 0, 0, 1, 0, x, y, 0, 0, 2z

    if dim == 1  # ∂/∂x operator
        @inbounds for op_idx in 1:num_ops
            Δeval_point[2] += Δb[k + 5, op_idx]  # from xy: ∂(y)/∂y = 1
            Δeval_point[3] += Δb[k + 6, op_idx]  # from xz: ∂(z)/∂z = 1
            Δeval_point[1] += Δb[k + 8, op_idx] * 2  # from x²: ∂(2x)/∂x = 2
        end
    elseif dim == 2  # ∂/∂y operator
        @inbounds for op_idx in 1:num_ops
            Δeval_point[1] += Δb[k + 5, op_idx]  # from xy: ∂(x)/∂x = 1
            Δeval_point[3] += Δb[k + 7, op_idx]  # from yz: ∂(z)/∂z = 1
            Δeval_point[2] += Δb[k + 9, op_idx] * 2  # from y²: ∂(2y)/∂y = 2
        end
    elseif dim == 3  # ∂/∂z operator
        @inbounds for op_idx in 1:num_ops
            Δeval_point[1] += Δb[k + 6, op_idx]  # from xz: ∂(x)/∂x = 1
            Δeval_point[2] += Δb[k + 7, op_idx]  # from yz: ∂(y)/∂y = 1
            Δeval_point[3] += Δb[k + 10, op_idx] * 2  # from z²: ∂(2z)/∂z = 2
        end
    end

    return nothing
end

# =============================================================================
# Shape parameter (ε) gradient computation
# =============================================================================

"""
    backward_collocation_ε!(Δε_acc, ΔA, neighbors, data, basis, k)

Compute gradient contribution to shape parameter ε from collocation matrix.

Uses implicit differentiation: Δε += Σᵢⱼ ΔA[i,j] * ∂A[i,j]/∂ε
where A[i,j] = φ(xi, xj) for the RBF block.
"""
function backward_collocation_ε!(
        Δε_acc::Base.RefValue{T},
        ΔA::AbstractMatrix{T},
        neighbors::Vector{Int},
        data::AbstractVector,
        basis::AbstractRadialBasis,
        k::Int,
    ) where {T}
    # RBF block: A[i,j] = φ(xi, xj)
    # Accumulate gradient from upper triangle (matrix is symmetric)
    @inbounds for j in 1:k
        xj = data[neighbors[j]]
        for i in 1:(j - 1)
            xi = data[neighbors[i]]
            # ∂φ/∂ε at this pair
            ∂φ_∂ε_val = ∂φ_∂ε(basis, xi, xj)
            # For symmetric matrix: ΔA[i,j] + ΔA[j,i]
            Δε_acc[] += (ΔA[i, j] + ΔA[j, i]) * ∂φ_∂ε_val
        end
    end
    return nothing
end

"""
    backward_rhs_ε!(Δε_acc, Δb, neighbors, eval_point, data, basis, k, ∂Lφ_∂ε_fn)

Compute gradient contribution to shape parameter ε from RHS.

`∂Lφ_∂ε_fn(x, xi)` returns ∂(ℒφ)/∂ε for the specific operator:
  - Laplacian: `(x, xi) -> ∂Laplacian_φ_∂ε(basis, x, xi)`
  - Partial:   `(x, xi) -> ∂Partial_φ_∂ε(basis, dim, x, xi)`
"""
function backward_rhs_ε!(
        Δε_acc::Base.RefValue{T},
        Δb::AbstractVecOrMat{T},
        neighbors::Vector{Int},
        eval_point,
        data::AbstractVector,
        k::Int,
        ∂Lφ_∂ε_fn::F,
    ) where {T, F}
    num_ops = size(Δb, 2)

    @inbounds for i in 1:k
        xi = data[neighbors[i]]
        ∂Lφ_∂ε_val = ∂Lφ_∂ε_fn(eval_point, xi)
        for op_idx in 1:num_ops
            Δε_acc[] += Δb[i, op_idx] * ∂Lφ_∂ε_val
        end
    end
    return nothing
end

"""
    backward_stencil_with_ε!(Δdata, Δeval_point, Δε_acc, Δw, cache, neighbors, eval_point, data, basis, mon, k, grad_Lφ_x, grad_Lφ_xi; poly_backward!=nothing, ∂Lφ_∂ε_fn=nothing)

Complete backward pass for a single stencil including shape parameter gradient.

Combines:
1. backward_linear_solve! → compute ΔA, Δb from Δw
2. backward_collocation! → chain ΔA to Δdata
3. backward_collocation_ε! → chain ΔA to Δε
4. backward_rhs! → chain Δb to Δdata and Δeval_point
5. backward_rhs_ε! → chain Δb to Δε

Optional closures:
- `poly_backward!`: polynomial section gradient (Partial only)
- `∂Lφ_∂ε_fn`: shape parameter derivative function (IMQ/Gaussian only)
"""
function backward_stencil_with_ε!(
        Δdata::Vector{Vector{T}},
        Δeval_point::Vector{T},
        Δε_acc::Base.RefValue{T},
        Δw::AbstractVecOrMat{T},
        cache::StencilForwardCache{T},
        neighbors::Vector{Int},
        eval_point,
        data::AbstractVector,
        basis::AbstractRadialBasis,
        mon::MonomialBasis{Dim, Deg},
        k::Int,
        grad_Lφ_x,
        grad_Lφ_xi;
        poly_backward!::F1 = nothing,
        ∂Lφ_∂ε_fn::F2 = nothing,
    ) where {T, Dim, Deg, F1, F2}
    n = k + cache.nmon

    ΔA = zeros(T, n, n)
    Δb = zeros(T, n, size(Δw, 2))

    backward_linear_solve!(ΔA, Δb, Δw, cache)
    backward_collocation!(Δdata, ΔA, neighbors, data, basis, mon, k)
    backward_collocation_ε!(Δε_acc, ΔA, neighbors, data, basis, k)
    backward_rhs!(
        Δdata, Δeval_point, Δb, neighbors, eval_point, data, basis, k,
        grad_Lφ_x, grad_Lφ_xi; poly_backward! = poly_backward!
    )
    if ∂Lφ_∂ε_fn !== nothing
        backward_rhs_ε!(Δε_acc, Δb, neighbors, eval_point, data, k, ∂Lφ_∂ε_fn)
    end

    return nothing
end

# =============================================================================
# Shared utilities for AD extensions
# =============================================================================

"""
    extract_stencil_cotangent(ΔW, eval_idx, neighbors, k, num_ops)

Extract cotangent values for a single stencil from a dense/sparse matrix cotangent.
Used by the Enzyme extension.
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

"""
    extract_stencil_cotangent_from_nzval(ΔW_nzval, W, eval_idx, neighbors, k)

Extract cotangent values for a single stencil from sparse matrix nzval gradient.
Used by Mooncake extension where gradients are stored in fdata.nzval.
"""
function extract_stencil_cotangent_from_nzval(
        ΔW_nzval::Vector{T}, W::SparseMatrixCSC, eval_idx::Int, neighbors::Vector{Int}, k::Int
    ) where {T}
    Δw = zeros(T, k, 1)
    for (local_idx, global_idx) in enumerate(neighbors)
        col_start = W.colptr[global_idx]
        col_end = W.colptr[global_idx + 1] - 1
        for pos in col_start:col_end
            if W.rowval[pos] == eval_idx
                Δw[local_idx, 1] = ΔW_nzval[pos]
                break
            end
        end
    end
    return Δw
end
