"""
    Partial <: ScalarValuedOperator

Builds an operator for a first order partial derivative.
"""
struct Partial{T<:Int} <: ScalarValuedOperator
    order::T
    dim::T
end
(op::Partial)(basis) = ∂(basis, op.order, op.dim)

# convienience constructors
"""
    function partial(data, order, dim, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the partial derivative, `Partial`, of `order` with respect to `dim`.
"""
function partial(
    data::AbstractVector,
    order::T,
    dim::T,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {T<:Int,B<:AbstractRadialBasis}
    ℒ = Partial(order, dim)
    return RadialBasisOperator(ℒ, data, basis; k=k, adjl=adjl)
end

"""
    function partial(data, eval_points, order, dim, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the partial derivative, `Partial`. The resulting operator will only evaluate at `eval_points`.
"""
function partial(
    data::AbstractVector,
    eval_points::AbstractVector,
    order::T,
    dim::T,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {T<:Int,B<:AbstractRadialBasis}
    ℒ = Partial(order, dim)
    return RadialBasisOperator(ℒ, data, eval_points, basis; k=k, adjl=adjl)
end

"""
    function partial(data, eval_points, order, dim, basis, is_boundary, boundary_conditions, normals; k=autoselect_k(data, basis))

Builds a Hermite-compatible `RadialBasisOperator` where the operator is the partial derivative, `Partial`. 
The additional boundary information enables Hermite interpolation with proper boundary condition handling.
"""
function partial(
    data::AbstractVector,
    eval_points::AbstractVector,
    order::T,
    dim::T,
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {T<:Int,B<:AbstractRadialBasis}
    ℒ = Partial(order, dim)
    # Store boundary information for Hermite dispatch
    return RadialBasisOperator(
        ℒ,
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

function ∂(basis::AbstractBasis, order::T, dim::T) where {T<:Int}
    if order == 1
        return ∂(basis, dim)
    elseif order == 2
        return ∂²(basis, dim)
    else
        throw(
            ArgumentError(
                "Only first and second order derivatives are supported right now. You may use the custom operator.",
            ),
        )
    end
end

# pretty printing
print_op(op::Partial) = "∂ⁿf/∂xᵢ (n = $(op.order), i = $(op.dim))"
