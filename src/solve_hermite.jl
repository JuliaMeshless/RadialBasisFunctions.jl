"""
Hermite interpolation scheme for handling boundary conditions in RBF stencils.

This module provides Hermite variants of the core stencil building functions
that properly handle Neumann and Robin boundary conditions by modifying
both the collocation matrix and RHS to maintain symmetry and non-singularity.
"""

using LinearAlgebra: dot

"""
Build collocation matrix for Hermite interpolation with boundary conditions.

SIMPLIFIED VERSION: Only Dirichlet boundaries are treated specially (as known values).
Neumann/Robin boundaries are unknowns, so we use standard basis functions everywhere.

The matrix is just standard RBF evaluation: φ(xᵢ, xⱼ) for all i,j
This works because:
- Interior points: solving for function values
- Dirichlet boundaries: values are known (handled in global system assembly)
- Neumann/Robin boundaries: solving for function values (BC enforced via RHS)
"""
function _build_collocation_matrix!(
    A::Symmetric,
    data::HermiteStencilData,
    basis::B,
    mon::MonomialBasis{Dim,Deg},
    k::K,
    eval_point::TE,
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg,TE}
    AA = parent(A)
    N = size(A, 2)

    # Build standard RBF matrix entries (no Hermite modifications!)
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = basis(data.data[i], data.data[j])
    end

    # Polynomial augmentation (standard evaluation)
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data.data[i])
        end
    end

    return nothing
end

"""
Build RHS for Hermite interpolation with boundary conditions.

SIMPLIFIED VERSION: 
- Interior eval_point: apply differential operator ℒ to all basis functions
- Dirichlet eval_point: identity row (extract value)
- Neumann/Robin eval_point: apply boundary operator (α + β*∂ₙ) to all basis functions

Since the matrix uses standard basis everywhere, the RHS must match accordingly.
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
    if eval_idx === nothing
        error("Evaluation point not found in stencil data.")
    end

    eval_is_boundary = data.is_boundary[eval_idx]

    # Neumann/Robin boundary - apply boundary operator to all basis functions
    if eval_is_boundary && !is_dirichlet(data.boundary_conditions[eval_idx])
        bc_eval = data.boundary_conditions[eval_idx]
        n_eval = data.normals[eval_idx]

        # Apply boundary operator (α + β*∂ₙ) to each basis function
        @inbounds for i in 1:k
            xi = data.data[i]
            φ = basis(eval_point, xi)
            ∇φ = ∇(basis)(eval_point, xi)
            # Boundary operator: α*φ + β*∂ₙφ
            b[i] = α(bc_eval) * φ + β(bc_eval) * dot(n_eval, ∇φ)
        end

        # Monomial augmentation with boundary operator
        if basis.poly_deg > -1
            N = length(b)
            bmono = view(b, (k + 1):N)
            nmon = length(bmono)

            T = eltype(bmono)
            poly_vals = zeros(T, nmon)
            mon(poly_vals, eval_point)

            ∂ₙmon = ∂_normal(mon, n_eval)
            deriv_vals = zeros(T, nmon)
            ∂ₙmon(deriv_vals, eval_point)

            # Apply boundary condition: α*P + β*∂ₙP
            @inbounds for idx in 1:nmon
                bmono[idx] = α(bc_eval) * poly_vals[idx] + β(bc_eval) * deriv_vals[idx]
            end
        end
        return nothing
    end

    # Interior point - apply differential operator to all basis functions
    @inbounds for i in 1:k
        b[i] = ℒrbf(eval_point, data.data[i])
    end

    # Monomial augmentation with differential operator
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        ℒmon(bmono, eval_point)
    end

    return nothing
end

"""
Multi-operator version of Hermite RHS building (simplified).

SIMPLIFIED VERSION: Same three-case logic as single-operator version.
- Interior eval_point: apply differential operators ℒ to all basis functions
- Dirichlet eval_point: identity row (extract value)
- Neumann/Robin eval_point: apply boundary operator (α + β*∂ₙ) to all basis functions
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
    if eval_idx === nothing
        error("Evaluation point not found in stencil data.")
    end

    eval_is_boundary = data.is_boundary[eval_idx]

    # Case 2: Neumann/Robin boundary - apply boundary operator to all basis functions
    if eval_is_boundary && !is_dirichlet(data.boundary_conditions[eval_idx])
        bc_eval = data.boundary_conditions[eval_idx]
        n_eval = data.normals[eval_idx]

        # Apply boundary operator (α + β*∂ₙ) to each basis function for all operators
        for (j, _) in enumerate(ℒrbf)
            @inbounds for i in 1:k
                xi = data.data[i]
                φ = basis(eval_point, xi)
                ∇φ = ∇(basis)(eval_point, xi)
                # Boundary operator: α*φ + β*∂ₙφ (same for all operators)
                b[i, j] = α(bc_eval) * φ + β(bc_eval) * dot(n_eval, ∇φ)
            end
        end

        # Monomial augmentation with boundary operator
        if basis.poly_deg > -1
            N = size(b, 1)
            nmon = binomial(dim(mon) + degree(mon), dim(mon))

            T = eltype(b)
            poly_vals = zeros(T, nmon)
            mon(poly_vals, eval_point)

            ∂ₙmon = ∂_normal(mon, n_eval)
            deriv_vals = zeros(T, nmon)
            ∂ₙmon(deriv_vals, eval_point)

            # Apply boundary condition: α*P + β*∂ₙP (same for all operators)
            for j in 1:size(b, 2)
                bmono = view(b, (k + 1):N, j)
                @inbounds for idx in 1:nmon
                    bmono[idx] = α(bc_eval) * poly_vals[idx] + β(bc_eval) * deriv_vals[idx]
                end
            end
        end
        return nothing
    end

    # Case 3: Interior point - apply differential operators to all basis functions
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in 1:k
            b[i, j] = ℒ(eval_point, data.data[i])
        end
    end

    # Monomial augmentation with differential operators
    if basis.poly_deg > -1
        N = size(b, 1)
        for (j, ℒ_op) in enumerate(ℒmon)
            bmono = view(b, (k + 1):N, j)
            ℒ_op(bmono, eval_point)
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
    _build_collocation_matrix!(A, data, basis, mon, k, eval_point)
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
