"""
Layer: Public API & Entry Points

This file contains all entry points for the weight building system:
- Main _build_weights function and its overloads
- Routing logic (standard vs Hermite paths)
- Operator application to basis functions
- Parameter validation and defaults

Called by: operators.jl, interpolation.jl
Calls: kernel_exec.jl
Dependencies: types.jl, kernel_exec.jl
"""

using KernelAbstractions: CPU

# ============================================================================
# Entry Points from Operators
# ============================================================================

"""
    _build_weights(ℒ, op)

Entry point from operator construction.
Extracts configuration from operator and routes to appropriate implementation.
"""
function _build_weights(ℒ, op)
    data = op.data
    eval_points = op.eval_points
    adjl = op.adjl
    basis = op.basis
    return _build_weights(ℒ, data, eval_points, adjl, basis)
end

"""
    _build_weights(ℒ, data, eval_points, adjl, basis)

Apply operator to basis functions and route to weight computation.
"""
function _build_weights(ℒ, data, eval_points, adjl, basis)
    dim = length(first(data))

    # Build monomial basis and apply operator
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    return _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
end

# ============================================================================
# Standard Path (No Boundary Conditions)
# ============================================================================

"""
    _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon;
                  batch_size=10, device=CPU())

Build weights for interior-only problems (no boundary conditions).
Creates empty BoundaryData to indicate all interior points.
"""
function _build_weights(
    data, eval_points, adjl, basis, ℒrbf, ℒmon, mon; batch_size::Int=10, device=CPU()
)
    # Create empty boundary data for interior-only case
    TD = eltype(first(data))
    is_boundary = fill(false, length(data))
    boundary_conditions = BoundaryCondition{TD}[]
    normals = similar(data, 0)
    boundary_data = BoundaryData(is_boundary, boundary_conditions, normals)

    return build_weights_kernel(
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

# ============================================================================
# Hermite Path (With Boundary Conditions)
# ============================================================================

"""
    _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon,
                  is_boundary, boundary_conditions, normals;
                  batch_size=10, device=CPU())

Build weights for problems with boundary conditions using Hermite interpolation.
Exact allocation: Dirichlet points get single entry, others get full stencil.
"""
function _build_weights(
    data,
    eval_points,
    adjl,
    basis,
    ℒrbf,
    ℒmon,
    mon,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    batch_size::Int=10,
    device=CPU(),
)
    boundary_data = BoundaryData(is_boundary, boundary_conditions, normals)
    return build_weights_kernel(
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
    _build_weights(ℒ, data, eval_points, adjl, basis,
                  is_boundary, boundary_conditions, normals)

Generic Hermite dispatcher for operators.
Applies operator to basis and routes to Hermite weight computation.

This eliminates repetitive _build_weights methods across operator files.
Note: Type constraint removed to avoid circular dependency with operators.jl
"""
function _build_weights(
    ℒ,  # Any operator type
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
