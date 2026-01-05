"""
    Custom{F<:Function} <: AbstractOperator

Custom operator that applies a user-defined function to basis functions.
The function `ℒ` should accept a basis and return a callable `(x, xᵢ) -> value`.
"""
struct Custom{F<:Function} <: AbstractOperator
    ℒ::F
end
(op::Custom)(basis) = op.ℒ(basis)

"""
    custom(data, ℒ, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` with a custom operator function `ℒ`.
The function `ℒ` should accept a basis and return a callable `(x, xᵢ) -> value`.
"""
function custom(
    data::AbstractVector,
    ℒ::Function,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {T<:Int,B<:AbstractRadialBasis}
    return RadialBasisOperator(Custom(ℒ), data, basis; k=k, adjl=adjl)
end

"""
    custom(data, eval_points, ℒ, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` with a custom operator function `ℒ`.
The resulting operator will only evaluate at `eval_points`.
"""
function custom(
    data::AbstractVector,
    eval_points::AbstractVector,
    ℒ::Function,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {T<:Int,B<:AbstractRadialBasis}
    return RadialBasisOperator(Custom(ℒ), data, eval_points, basis; k=k, adjl=adjl)
end

"""
    custom(data, eval_points, ℒ, basis, is_boundary, boundary_conditions, normals; k=autoselect_k(data, basis))

Builds a Hermite-compatible `RadialBasisOperator` with a custom operator function `ℒ`.
The additional boundary information enables Hermite interpolation with proper boundary condition handling.
"""
function custom(
    data::AbstractVector,
    eval_points::AbstractVector,
    ℒ::Function,
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {T<:Int,B<:AbstractRadialBasis}
    return RadialBasisOperator(
        Custom(ℒ),
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

# pretty printing
print_op(op::Custom) = "Custom Operator"
