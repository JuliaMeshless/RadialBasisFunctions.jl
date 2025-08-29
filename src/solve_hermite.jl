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
    A::Symmetric,
    data::AbstractVector,
    basis::B,
    mon::MonomialBasis{Dim,Deg},
    k::K,
    ::HermiteStencil,
    boundary_info::HermiteBoundaryInfo{T}
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg,T}
    
    AA = parent(A)
    N = size(A, 2)
    
    # Build RBF matrix entries with Hermite modifications
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = _hermite_rbf_entry(i, j, data, basis, boundary_info)
    end
    
    # Polynomial augmentation with boundary operator modifications
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            _hermite_poly_entry!(a, i, data, mon, boundary_info)
        end
    end
    
    return nothing
end

"""
Compute single RBF matrix entry for Hermite interpolation.
Handles all combinations of interior/boundary points with appropriate operators.
"""
function _hermite_rbf_entry(
    i::Int, j::Int,
    data::AbstractVector,
    basis::B,
    boundary_info::HermiteBoundaryInfo{T}
) where {B<:AbstractRadialBasis,T}
    
    xi, xj = data[i], data[j]
    is_bound_i = boundary_info.is_boundary[i]
    is_bound_j = boundary_info.is_boundary[j]
    
    # Standard case: both interior points
    if !is_bound_i && !is_bound_j
        return basis(xi, xj)
    end
    
    # Get basis value and gradient
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    
    bc_i = boundary_info.boundary_conditions[i]
    bc_j = boundary_info.boundary_conditions[j]
    
    # Cases involving boundary points
    if is_bound_i && !is_bound_j
        # Boundary-Interior: Apply boundary operator to first argument
        ni = boundary_info.normals[i]
        if is_dirichlet(bc_i)
            return φ
        else
            # Neumann/Robin: α*φ + β*∂ₙφ
            return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
        end
        
    elseif !is_bound_i && is_bound_j
        # Interior-Boundary: Apply boundary operator to second argument
        nj = boundary_info.normals[j]
        if is_dirichlet(bc_j)
            return φ
        else
            # Neumann/Robin: α*φ + β*∂ₙφ (note sign flip for gradient w.r.t. second arg)
            return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
        end
        
    else # is_bound_i && is_bound_j
        # Boundary-Boundary: Apply boundary operators to both arguments
        ni = boundary_info.normals[i]
        nj = boundary_info.normals[j]
        
        if is_dirichlet(bc_i) && is_dirichlet(bc_j)
            return φ
        elseif is_dirichlet(bc_i) && !is_dirichlet(bc_j)
            return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
        elseif !is_dirichlet(bc_i) && is_dirichlet(bc_j)
            return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
        else
            # Both Neumann/Robin: mixed derivative term
            ∂i∂j_φ = directional∂²(basis, ni, nj)(xi, xj)
            return (α(bc_i) * α(bc_j) * φ + 
                   α(bc_i) * β(bc_j) * dot(nj, -∇φ) +
                   β(bc_i) * α(bc_j) * dot(ni, ∇φ) +
                   β(bc_i) * β(bc_j) * ∂i∂j_φ)
        end
    end
end

"""
Compute polynomial entries for Hermite interpolation.
Applies boundary operators to polynomial basis functions at boundary points.
"""
function _hermite_poly_entry!(
    a::AbstractVector,
    i::Int,
    data::AbstractVector,
    mon::MonomialBasis,
    boundary_info::HermiteBoundaryInfo{T}
) where {T}
    
    xi = data[i]
    is_bound_i = boundary_info.is_boundary[i]
    
    if !is_bound_i
        # Interior point: standard polynomial evaluation
        mon(a, xi)
    else
        bc_i = boundary_info.boundary_conditions[i]
        if is_dirichlet(bc_i)
            # Dirichlet boundary: standard polynomial evaluation
            mon(a, xi)
        else
            # Neumann/Robin: α*P + β*∂ₙP
            ni = boundary_info.normals[i]
            nmon = length(a)
            
            # Evaluate polynomial and its normal derivative
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
"""
function _build_rhs!(
    b,
    ℒrbf,
    ℒmon,
    data::AbstractVector{TD},
    eval_point::TE,
    basis::B,
    k::K,
    ::HermiteStencil,
    boundary_info::HermiteBoundaryInfo{T}
) where {TD,TE,B<:AbstractRadialBasis,K<:Int,T}
    
    # RBF section with Hermite modifications
    @inbounds for i in eachindex(data)
        if boundary_info.is_boundary[i]
            bc_i = boundary_info.boundary_conditions[i]
            if is_dirichlet(bc_i)
                b[i] = ℒrbf(eval_point, data[i])
            else
                # Neumann/Robin: Apply boundary operator to RBF operator
                ni = boundary_info.normals[i]
                b[i] = (α(bc_i) * ℒrbf(eval_point, data[i]) + 
                       β(bc_i) * ℒrbf(eval_point, data[i], ni))
            end
        else
            # Interior point: standard evaluation
            b[i] = ℒrbf(eval_point, data[i])
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
    data::AbstractVector{TD},
    eval_point::TE,
    basis::B,
    k::K,
    ::HermiteStencil,
    boundary_info::HermiteBoundaryInfo{T}
) where {TD,TE,B<:AbstractRadialBasis,K<:Int,T}
    
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"
    
    # RBF section with Hermite modifications
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in eachindex(data)
            if boundary_info.is_boundary[i]
                bc_i = boundary_info.boundary_conditions[i]
                if is_dirichlet(bc_i)
                    b[i, j] = ℒ(eval_point, data[i])
                else
                    # Neumann/Robin: Apply boundary operator to RBF operator
                    ni = boundary_info.normals[i]
                    b[i, j] = (α(bc_i) * ℒ(eval_point, data[i]) + 
                              β(bc_i) * ℒ(eval_point, data[i], ni))
                end
            else
                # Interior point: standard evaluation
                b[i, j] = ℒ(eval_point, data[i])
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
    data::AbstractVector{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::Int,
    ::HermiteStencil,
    boundary_info::HermiteBoundaryInfo{T}
) where {TD,TE,B<:AbstractRadialBasis,T}
    
    _build_collocation_matrix!(A, data, basis, mon, k, HermiteStencil(), boundary_info)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k, HermiteStencil(), boundary_info)
    
    return (A \ b)[1:k, :]
end