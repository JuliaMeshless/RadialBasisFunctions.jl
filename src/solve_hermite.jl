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

# Boundary point types for dispatch-based operator application
abstract type BoundaryPointType end
struct InteriorPoint <: BoundaryPointType end
struct DirichletPoint <: BoundaryPointType end
struct NeumannRobinPoint <: BoundaryPointType end

"""
Determine the boundary type of a point for dispatch.
"""
@inline function point_type(is_bound::Bool, bc::BoundaryCondition)
    return is_bound ? (is_dirichlet(bc) ? DirichletPoint() : NeumannRobinPoint()) : InteriorPoint()
end

"""
Compute single RBF matrix entry for Hermite interpolation.
Handles all combinations of interior/boundary points with appropriate operators.
Dispatch-based implementation for clarity and maintainability.
"""
function _hermite_rbf_entry(
    i::Int, j::Int, data::HermiteStencilData{T}, basis::B
) where {B<:AbstractRadialBasis,T}
    xi, xj = data.data[i], data.data[j]
    type_i = point_type(data.is_boundary[i], data.boundary_conditions[i])
    type_j = point_type(data.is_boundary[j], data.boundary_conditions[j])

    return _hermite_rbf_entry_dispatch(type_i, type_j, i, j, xi, xj, data, basis)
end

# Dispatch implementations for all 9 combinations of point types

"""
Interior-Interior: Standard RBF evaluation.
"""
function _hermite_rbf_entry_dispatch(
    ::InteriorPoint, ::InteriorPoint, i, j, xi, xj, data, basis
)
    return basis(xi, xj)
end

"""
Interior-Dirichlet: Standard RBF evaluation.
"""
function _hermite_rbf_entry_dispatch(
    ::InteriorPoint, ::DirichletPoint, i, j, xi, xj, data, basis
)
    return basis(xi, xj)
end

"""
Interior-NeumannRobin: Apply boundary operator to second argument.
"""
function _hermite_rbf_entry_dispatch(
    ::InteriorPoint, ::NeumannRobinPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_j = data.boundary_conditions[j]
    nj = data.normals[j]
    return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
end

"""
Dirichlet-Interior: Standard RBF evaluation.
"""
function _hermite_rbf_entry_dispatch(
    ::DirichletPoint, ::InteriorPoint, i, j, xi, xj, data, basis
)
    return basis(xi, xj)
end

"""
Dirichlet-Dirichlet: Standard RBF evaluation.
"""
function _hermite_rbf_entry_dispatch(
    ::DirichletPoint, ::DirichletPoint, i, j, xi, xj, data, basis
)
    return basis(xi, xj)
end

"""
Dirichlet-NeumannRobin: Apply boundary operator to second argument.
"""
function _hermite_rbf_entry_dispatch(
    ::DirichletPoint, ::NeumannRobinPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_j = data.boundary_conditions[j]
    nj = data.normals[j]
    return α(bc_j) * φ + β(bc_j) * dot(nj, -∇φ)
end

"""
NeumannRobin-Interior: Apply boundary operator to first argument.
"""
function _hermite_rbf_entry_dispatch(
    ::NeumannRobinPoint, ::InteriorPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_i = data.boundary_conditions[i]
    ni = data.normals[i]
    return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
end

"""
NeumannRobin-Dirichlet: Apply boundary operator to first argument.
"""
function _hermite_rbf_entry_dispatch(
    ::NeumannRobinPoint, ::DirichletPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_i = data.boundary_conditions[i]
    ni = data.normals[i]
    return α(bc_i) * φ + β(bc_i) * dot(ni, ∇φ)
end

"""
NeumannRobin-NeumannRobin: Apply boundary operators to both arguments.
Includes mixed derivative term for directional derivatives.
"""
function _hermite_rbf_entry_dispatch(
    ::NeumannRobinPoint, ::NeumannRobinPoint, i, j, xi, xj, data, basis
)
    φ = basis(xi, xj)
    ∇φ = ∇(basis)(xi, xj)
    bc_i = data.boundary_conditions[i]
    bc_j = data.boundary_conditions[j]
    ni = data.normals[i]
    nj = data.normals[j]
    ∂i∂j_φ = directional∂²(basis, ni, nj)(xi, xj)

    return (
        α(bc_i) * α(bc_j) * φ +
        α(bc_i) * β(bc_j) * dot(nj, -∇φ) +
        β(bc_i) * α(bc_j) * dot(ni, ∇φ) +
        β(bc_i) * β(bc_j) * ∂i∂j_φ
    )
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
Helper: Apply boundary conditions to RBF operator evaluation.
Handles Dirichlet (identity) and Neumann/Robin (α*φ + β*∂ₙφ) cases.
"""
@inline function _apply_boundary_to_rbf(
    ℒrbf, eval_point, data_point, is_bound::Bool, bc::BoundaryCondition, normal
)
    if !is_bound
        return ℒrbf(eval_point, data_point)
    end

    if is_dirichlet(bc)
        return ℒrbf(eval_point, data_point)
    else
        # Neumann/Robin: α*ℒφ + β*ℒ(∂ₙφ)
        return α(bc) * ℒrbf(eval_point, data_point) + β(bc) * ℒrbf(eval_point, data_point, normal)
    end
end

"""
Helper: Apply boundary conditions to monomial evaluation.
For interior/Dirichlet: apply operator ℒmon to monomial basis
For Neumann/Robin: α*ℒmon(P) + β*ℒmon(∂ₙP)

Note: For interior/Dirichlet evaluation points, the operator ℒmon is applied to the standard
monomial basis. For Neumann/Robin evaluation points at boundaries, the boundary operator
modifies the monomial basis before the differential operator is applied.
"""
@inline function _apply_boundary_to_mono!(
    bmono::AbstractVector, ℒmon, mon::MonomialBasis, eval_point,
    is_bound::Bool, bc::BoundaryCondition, normal, T::Type
)
    if !is_bound || is_dirichlet(bc)
        # Interior or Dirichlet: apply operator to standard monomial basis
        ℒmon(bmono, eval_point)
        return
    end

    # Neumann/Robin case: α*ℒmon(P) + β*ℒmon(∂ₙP)
    # The boundary operator modifies how we evaluate the monomials
    nmon = length(bmono)
    poly_vals = zeros(T, nmon)
    deriv_vals = zeros(T, nmon)

    mon(poly_vals, eval_point)
    ∂_normal(mon, normal)(deriv_vals, eval_point)

    @inbounds for idx in 1:nmon
        bmono[idx] = α(bc) * poly_vals[idx] + β(bc) * deriv_vals[idx]
    end
end

"""
Build RHS for Hermite interpolation with boundary conditions.
This is the Hermite variant of _build_rhs! from solve.jl.

When the evaluation point (center of stencil) has Neumann/Robin conditions,
the differential operator must be modified according to the boundary operator.
This is the comment I forgot to implement before, now added.
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

    # RBF section with Hermite modifications for stencil points
    @inbounds for i in 1:k
        b[i] = _apply_boundary_to_rbf(
            ℒrbf, eval_point, data.data[i],
            data.is_boundary[i], data.boundary_conditions[i], data.normals[i]
        )
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)

        # Find evaluation point index (it must be in the stencil)
        eval_idx = findfirst(i -> data.data[i] == eval_point, 1:k)
        if eval_idx === nothing
            error("Evaluation point not found in stencil data.")
        end
        if is_dirichlet(data.boundary_conditions[eval_idx])
            error("Dirichlet eval nodes should be handled at the higher level (identity row).")
        end

        # Apply boundary conditions to monomial section
        _apply_boundary_to_mono!(
            bmono, ℒmon, mon, eval_point,
            data.is_boundary[eval_idx], data.boundary_conditions[eval_idx],
            data.normals[eval_idx], TD
        )
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
    mon::MonomialBasis,
    k::K,
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"

    # RBF section with Hermite modifications for stencil points
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            b[i, j] = _apply_boundary_to_rbf(
                ℒ, eval_point, data.data[i],
                data.is_boundary[i], data.boundary_conditions[i], data.normals[i]
            )
        end
    end

    # Monomial augmentation
    if basis.poly_deg > -1
        N = size(b, 1)

        # Find evaluation point index (it must be in the stencil)
        eval_idx = findfirst(i -> data.data[i] == eval_point, 1:k)
        if eval_idx === nothing
            error("Evaluation point not found in stencil data.")
        end
        if is_dirichlet(data.boundary_conditions[eval_idx])
            error("Dirichlet eval nodes should be handled at the higher level (identity row).")
        end

        # Apply boundary conditions to monomial section for all operators
        if data.is_boundary[eval_idx]
            # Boundary evaluation point: compute boundary-modified monomials once
            bc_eval = data.boundary_conditions[eval_idx]
            n_eval = data.normals[eval_idx]
            nmon = binomial(dim(mon) + degree(mon), dim(mon))

            # Pre-compute modified monomial values once
            poly_vals = zeros(TD, nmon)
            deriv_vals = zeros(TD, nmon)
            mon(poly_vals, eval_point)
            ∂_normal(mon, n_eval)(deriv_vals, eval_point)

            # Apply to all operators
            for j in 1:length(ℒmon)
                bmono = view(b, (k + 1):N, j)
                @inbounds for idx in 1:nmon
                    bmono[idx] = α(bc_eval) * poly_vals[idx] + β(bc_eval) * deriv_vals[idx]
                end
            end
        else
            # Interior evaluation point: apply each operator directly
            for (j, ℒ_op) in enumerate(ℒmon)
                bmono = view(b, (k + 1):N, j)
                ℒ_op(bmono, eval_point)
            end
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
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, mon, k)

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

"""
Generic Hermite dispatcher for operators.
This eliminates repetitive _build_weights methods across operator files.
All operators that can call ℒ(basis) and ℒ(mon) can use this dispatcher.
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
    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    return _build_weights(
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        is_boundary,
        boundary_conditions,
        normals,
    )
end
