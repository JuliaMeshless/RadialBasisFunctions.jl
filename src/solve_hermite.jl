"""
Hermite interpolation scheme for handling boundary conditions in RBF stencils.

This module provides Hermite variants of the core stencil building functions
that properly handle Neumann and Robin boundary conditions by modifying
both the collocation matrix and RHS to maintain symmetry and non-singularity.
"""

using LinearAlgebra: dot

"""
Build collocation matrix for Hermite interpolation with boundary conditions.
This is the Hermite variant of _build_collocation_matrix! from solve.jl.

For boundary points with Neumann/Robin conditions, the basis functions are modified:
- Instead of Φ(·,xⱼ), we use B₂Φ(·,xⱼ) where B is the boundary operator
- This maintains matrix symmetry by applying the same operator to rows and columns
"""
function _build_collocation_matrix!(
    A::Symmetric, data::HermiteStencilData, basis::B, mon::MonomialBasis{Dim,Deg}, k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    AA = parent(A)
    N = size(A, 2)

    # Build RBF matrix entries with Hermite modifications
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = _hermite_rbf_entry(i, j, data, basis)
    end

    # Polynomial augmentation with boundary operator modifications
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            _hermite_poly_entry!(a, i, data, mon)
        end
    end

    return nothing
end

"""
Compute single RBF matrix entry for Hermite interpolation.
Handles all combinations of interior/boundary points with appropriate operators.
"""
function _hermite_rbf_entry(
    i::Int, j::Int, data::HermiteStencilData{T}, basis::B
) where {B<:AbstractRadialBasis,T}
    xi, xj = data.data[i], data.data[j]
    is_bound_i = data.is_boundary[i]
    is_bound_j = data.is_boundary[j]

    # Standard case: both interior points
    if !is_bound_i && !is_bound_j
        return basis(xi, xj)
    end

    # Get basis value and gradient
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)

    bc_i = data.boundary_conditions[i]
    bc_j = data.boundary_conditions[j]

    # Cases involving boundary points
    if is_bound_i && !is_bound_j
        # Boundary-Interior: Apply boundary operator to first argument
        ni = data.normals[i]
        if is_dirichlet(bc_i)
            return φ
        else
            # Neumann/Robin: α*φ + β*∂ₙφ
            return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
        end

    elseif !is_bound_i && is_bound_j
        # Interior-Boundary: Apply boundary operator to second argument
        nj = data.normals[j]
        if is_dirichlet(bc_j)
            return φ
        else
            # Neumann/Robin: α*φ + β*∂ₙφ (note sign flip for gradient w.r.t. second arg)
            return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
        end

    else # is_bound_i && is_bound_j
        # Boundary-Boundary: Apply boundary operators to both arguments
        ni = data.normals[i]
        nj = data.normals[j]

        if is_dirichlet(bc_i) && is_dirichlet(bc_j)
            return φ
        elseif is_dirichlet(bc_i) && !is_dirichlet(bc_j)
            return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
        elseif !is_dirichlet(bc_i) && is_dirichlet(bc_j)
            return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
        else
            # Both Neumann/Robin: mixed derivative term
            ∂i∂j_φ = directional∂²(basis, ni, nj)(xi, xj)
            return (
                α(bc_i) * α(bc_j) * φ +
                α(bc_i) * β(bc_j) * dot(nj, -∇φ) +
                β(bc_i) * α(bc_j) * dot(ni, ∇φ) +
                β(bc_i) * β(bc_j) * ∂i∂j_φ
            )
        end
    end
end

"""
Compute polynomial entries for Hermite interpolation.
Applies boundary operators to polynomial basis functions at boundary points.
"""
function _hermite_poly_entry!(
    a::AbstractVector, i::Int, data::HermiteStencilData, mon::MonomialBasis
)
    xi = data.data[i]
    is_bound_i = data.is_boundary[i]

    if !is_bound_i
        # Interior point: standard polynomial evaluation
        mon(a, xi)
    else
        bc_i = data.boundary_conditions[i]
        if is_dirichlet(bc_i)
            # Dirichlet boundary: standard polynomial evaluation
            mon(a, xi)
        else
            # Neumann/Robin: α*P + β*∂ₙP
            ni = data.normals[i]
            nmon = length(a)

            # Evaluate polynomial and its normal derivative
            T = eltype(a)
            poly_vals = zeros(T, nmon)
            deriv_vals = zeros(T, nmon)

            mon(poly_vals, xi)
            ∂_normal(mon, ni)(deriv_vals, xi)

            # Apply boundary condition
            @inbounds for k in 1:nmon
                a[k] = α(bc_i) * poly_vals[k] + β(bc_i) * deriv_vals[k]
            end
        end
    end

    return nothing
end

"""
Build RHS for Hermite interpolation with boundary conditions.
This is the Hermite variant of _build_rhs! from solve.jl.

When the evaluation point (center of stencil) has Neumann/Robin conditions,
the differential operator must be modified according to the boundary operator.
"""
function _build_rhs!(
    b, ℒrbf, ℒmon, data::HermiteStencilData{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}

    # RBF section with Hermite modifications for stencil points
    @inbounds for i in 1:k
        if data.is_boundary[i]
            bc_i = data.boundary_conditions[i]
            if is_dirichlet(bc_i)
                b[i] = ℒrbf(eval_point, data.data[i])
            else
                # Neumann/Robin: Apply boundary operator to RBF operator
                ni = data.normals[i]
                b[i] = (
                    α(bc_i) * ℒrbf(eval_point, data.data[i]) +
                    β(bc_i) * ℒrbf(eval_point, data.data[i], ni)
                )
            end
        else
            # Interior point: standard evaluation
            b[i] = ℒrbf(eval_point, data.data[i])
        end
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        ℒmon(bmono, eval_point)
    end

    return nothing
end

"""
Multi-operator version of Hermite RHS building.
"""
function _build_rhs!(
    b,
    ℒrbf::Tuple,
    ℒmon::Tuple,
    data::HermiteStencilData{TD},
    eval_point::TE,
    basis::B,
    k::K,
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"

    # RBF section with Hermite modifications for stencil points
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            if data.is_boundary[i]
                bc_i = data.boundary_conditions[i]
                if is_dirichlet(bc_i)
                    b[i, j] = ℒ(eval_point, data.data[i])
                else
                    # Neumann/Robin: Apply boundary operator to RBF operator
                    ni = data.normals[i]
                    b[i, j] = (
                        α(bc_i) * ℒ(eval_point, data.data[i]) +
                        β(bc_i) * ℒ(eval_point, data.data[i], ni)
                    )
                end
            else
                # Interior point: standard evaluation
                b[i, j] = ℒ(eval_point, data.data[i])
            end
        end
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        N = size(b, 1)
        for (j, ℒ) in enumerate(ℒmon)
            bmono = view(b, (k + 1):N, j)
            ℒ(bmono, eval_point)
        end
    end

    return nothing
end

"""
Build complete Hermite stencil with boundary conditions.
This is the Hermite variant of _build_stencil! from solve.jl.
"""
function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data::HermiteStencilData{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::Int,
) where {TD,TE,B<:AbstractRadialBasis}
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)

    return (A \ b)[1:k, :]
end

# Note: The SubArray method has been removed to delegate to solve.jl
# When local_data is a SubArray{<:AbstractVector}, it will automatically
# dispatch to the _build_stencil! method in solve.jl since SubArray <: AbstractVector

# _build_rhs_standard! function removed - now using the one from solve.jl

"""
Build weights for Hermite interpolation with sparse matrix construction.
This function follows the solve_hermite.jl philosophy by extending the core
_build_weights function to handle boundary conditions on both stencil points
AND evaluation points.

This is the Hermite variant of _build_weights from solve.jl, optimized for
sparse matrix construction with proper boundary condition handling.
"""
function _build_weights(
    data::Vector{<:AbstractVector},
    eval_points,
    adjl,
    basis::AbstractRadialBasis,
    ℒrbf,
    ℒmon,
    mon::MonomialBasis,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    batch_size::Int=10,
    device=CPU(),
)
    # Use the unified kernel infrastructure with optimized allocation strategy
    boundary_data = (is_boundary, boundary_conditions, normals)
    return _build_weights_unified(
        OptimizedAllocation(),
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        boundary_data;
        batch_size=batch_size,
        device=device,
    )
end
