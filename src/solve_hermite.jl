"""
Hermite interpolation scheme for handling boundary conditions in RBF stencils.

This module provides Hermite variants of the core stencil building functions
that properly handle Neumann and Robin boundary conditions by modifying
both the collocation matrix and RHS to maintain symmetry and non-singularity.
"""

using LinearAlgebra: dot

"""
    _fill_boundary_monomial_rhs!(bmono, mon, eval_point, bc, n)

Fill monomial contribution to RHS with boundary operator (α + β*∂ₙ) applied.

This computes: bmono[i] = α(bc) * pᵢ(x) + β(bc) * ∂ₙpᵢ(x)
where pᵢ are the monomial basis functions.
"""
function _fill_boundary_monomial_rhs!(
    bmono::AbstractVector,
    mon::MonomialBasis,
    eval_point,
    bc::BoundaryCondition,
    n::AbstractVector,
)
    nmon = length(bmono)
    α_val = α(bc)
    β_val = β(bc)

    # Fill with polynomial values scaled by α
    mon(bmono, eval_point)
    @inbounds for idx in 1:nmon
        bmono[idx] *= α_val
    end

    # Add normal derivative contribution scaled by β
    if !iszero(β_val)
        ∂ₙmon = ∂_normal(mon, n)
        deriv_vals = similar(bmono)
        ∂ₙmon(deriv_vals, eval_point)
        @inbounds for idx in 1:nmon
            bmono[idx] += β_val * deriv_vals[idx]
        end
    end

    return nothing
end

"""
Build RHS for Hermite interpolation with boundary conditions.

NOTE: This function is ONLY called for HermiteStencil cases (determined by stencil_type()).
Dirichlet stencils are handled separately in solve_utils.jl via _handle_dirichlet_optimized!.
Internal stencils (no boundary neighbors) use the standard solve.jl implementation.

Two cases handled here:
- Neumann/Robin eval_point: apply boundary operator (α + β*∂ₙ) to all basis functions
- Interior eval_point with boundary neighbors: apply differential operator ℒ to all basis functions
"""
function _build_rhs!(
    b,
    ℒrbf,
    ℒmon,
    data::HermiteStencilData{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::K,
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    eval_idx = findfirst(i -> data.data[i] == eval_point, 1:k)
    eval_idx === nothing && error("Evaluation point not found in stencil data.")
    if !data.is_boundary[eval_idx]
        error("Dispatch to standard _build_rhs! expected for interior eval points.")
    end
    bc_eval = data.boundary_conditions[eval_idx]
    n_eval = data.normals[eval_idx]

    is_dirichlet(bc_eval) && error("Dirichlet boundary should not be handled here.")

    @inbounds for i in 1:k
        xi = data.data[i]
        φ = basis(eval_point, xi)
        ∇φ = ∇(basis)(eval_point, xi)
        b[i] = α(bc_eval) * φ + β(bc_eval) * dot(n_eval, ∇φ)
    end

    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        _fill_boundary_monomial_rhs!(bmono, mon, eval_point, bc_eval, n_eval)
    end

    return nothing
end

"""
Multi-operator version of Hermite RHS building.

NOTE: This function is ONLY called for HermiteStencil cases (determined by stencil_type()).
Dirichlet and purely internal stencils are handled elsewhere.

Two cases handled here:
- Neumann/Robin eval_point: apply boundary operator (α + β*∂ₙ) to all basis functions
- Interior eval_point with boundary neighbors: apply differential operators ℒ to all basis functions
"""
function _build_rhs!(
    b,
    ℒrbf::Tuple,
    ℒmon::Tuple,
    data::HermiteStencilData{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::K,
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"

    eval_idx = findfirst(i -> data.data[i] == eval_point, 1:k)
    eval_idx === nothing && error("Evaluation point not found in stencil data.")
    if !data.is_boundary[eval_idx]
        error("Dispatch to standard _build_rhs! expected for interior eval points.")
    end

    # Neumann/Robin boundary - apply boundary operator (α + β*∂ₙ)
    bc_eval = data.boundary_conditions[eval_idx]
    n_eval = data.normals[eval_idx]

    is_dirichlet(bc_eval) && error("Dirichlet boundary should not be handled here.")
    # Apply boundary operator to each basis function for all operators
    for (j, _) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            xi = data.data[i]
            φ = basis(eval_point, xi)
            ∇φ = ∇(basis)(eval_point, xi)
            b[i, j] = α(bc_eval) * φ + β(bc_eval) * dot(n_eval, ∇φ)
        end
    end

    # Monomial augmentation with boundary operator
    if basis.poly_deg > -1
        N = size(b, 1)
        for j in 1:size(b, 2)
            bmono = view(b, (k + 1):N, j)
            _fill_boundary_monomial_rhs!(bmono, mon, eval_point, bc_eval, n_eval)
        end
    end

    return nothing
end

"""
Hermite stencil builder - extracts data and delegates to standard implementation.

This thin wrapper extracts `data.data` from HermiteStencilData and calls the standard
_build_stencil! from solve.jl. The Hermite-specific logic is in _build_rhs! via dispatch.
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
    _build_collocation_matrix!(A, data.data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)
    return (A \ b)[1:k, :]
end

"""
Core Hermite weight builder with explicit operators.

This is the low-level function that calls _build_weights_unified with OptimizedAllocation.
Use this when you already have ℒrbf and ℒmon computed.
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
    return _build_weights_unified(
        OptimizedAllocation(),
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        (is_boundary, boundary_conditions, normals);
        batch_size=batch_size,
        device=device,
    )
end

"""
High-level Hermite dispatcher for operators with boundary conditions.

This is the public interface that matches the standard operator call pattern.
It computes ℒrbf and ℒmon from the operator and delegates to the core implementation.
"""
function _build_weights(
    ℒ::AbstractOperator,
    data::AbstractVector,
    eval_points::AbstractVector,
    adjl::AbstractVector,
    basis::AbstractRadialBasis,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector},
)
    mon = MonomialBasis(length(first(data)), basis.poly_deg)

    return _build_weights(
        data,
        eval_points,
        adjl,
        basis,
        ℒ(basis),
        ℒ(mon),
        mon,
        is_boundary,
        boundary_conditions,
        normals,
    )
end
