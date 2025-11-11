"""
    Laplacian <: ScalarValuedOperator

Operator for the sum of the second derivatives w.r.t. each independent variable.
"""
struct Laplacian <: ScalarValuedOperator end
(::Laplacian)(basis) = ∇²(basis)

# convienience constructors
function laplacian(
    data::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {T<:Int,B<:AbstractRadialBasis}
    return RadialBasisOperator(Laplacian(), data, basis; k=k, adjl=adjl)
end

function laplacian(
    data::AbstractVector,
    eval_points::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {T<:Int,B<:AbstractRadialBasis}
    return RadialBasisOperator(Laplacian(), data, eval_points, basis; k=k, adjl=adjl)
end

"""
    function laplacian(data, eval_points, basis, is_boundary, boundary_conditions, normals; k=autoselect_k(data, basis))

Builds a Hermite-compatible `RadialBasisOperator` where the operator is the Laplacian, `Laplacian`.
The additional boundary information enables Hermite interpolation with proper boundary condition handling.
"""
function laplacian(
    data::AbstractVector,
    eval_points::AbstractVector,
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {T<:Int,B<:AbstractRadialBasis}
    return RadialBasisOperator(
        Laplacian(),
        data,
        eval_points,
        basis,
        is_boundary,
        boundary_conditions,
        normals;
        k=k,
        adjl=adjl,
    )
end

# Hermite-compatible method now uses the generic dispatcher in solve_hermite.jl

# pretty printing
print_op(op::Laplacian) = "Laplacian (∇²f)"
