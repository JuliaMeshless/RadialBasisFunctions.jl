"""
    Directional{Dim,T} <: ScalarValuedOperator

Operator for the directional derivative, or the inner product of the gradient and a direction vector.
"""
struct Directional{Dim,T} <: ScalarValuedOperator
    v::T
end
Directional{Dim}(v) where {Dim} = Directional{Dim,typeof(v)}(v)

"""
    function directional(data, v, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the directional derivative, `Directional`.
"""
function directional(
    data::AbstractVector,
    v::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Directional{Dim}(v)
    return RadialBasisOperator(ℒ, data, basis; k=k, adjl=adjl)
end

"""
    function directional(data, eval_points, v, basis; k=autoselect_k(data, basis))

Builds a `RadialBasisOperator` where the operator is the directional derivative, `Directional`.
"""
function directional(
    data::AbstractVector,
    eval_points::AbstractVector,
    v::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Directional{Dim}(v)
    return RadialBasisOperator(ℒ, data, eval_points, basis; k=k, adjl=adjl)
end

"""
    function directional(data, eval_points, v, basis, is_boundary, boundary_conditions, normals; k=autoselect_k(data, basis))

Builds a Hermite-compatible `RadialBasisOperator` where the operator is the directional derivative, `Directional`.
The additional boundary information enables Hermite interpolation with proper boundary condition handling.
"""
function directional(
    data::AbstractVector,
    eval_points::AbstractVector,
    v::AbstractVector,
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {B<:AbstractRadialBasis,T<:Int}
    Dim = length(first(data))
    ℒ = Directional{Dim}(v)
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

# Helper function to validate directional vector dimensions
function _validate_directional_vector(v, Dim, data)
    if !(length(v) == Dim || length(v) == length(data))
        throw(
            DomainError(
                "The geometrical vector for Directional() should match either the dimension of the input or the number of input points. The geometrical vector length is $(length(v)) while there are $(length(data)) points with a dimension of $Dim",
            ),
        )
    end
end

# Helper function to reduce gradient weights to directional weights
function _reduce_gradient_to_directional(weights, v, Dim, data)
    if length(v) == Dim
        return mapreduce(+, zip(weights, v)) do zipped
            w, vᵢ = zipped
            w * vᵢ
        end
    else
        vv = ntuple(i -> getindex.(v, i), Dim)
        return mapreduce(+, zip(weights, vv)) do zipped
            w, vᵢ = zipped
            Diagonal(vᵢ) * w
        end
    end
end

function _build_weights(ℒ::Directional{Dim}, data, eval_points, adjl, basis) where {Dim}
    v = ℒ.v
    _validate_directional_vector(v, Dim, data)
    weights = _build_weights(Gradient{Dim}(), data, eval_points, adjl, basis)
    return _reduce_gradient_to_directional(weights, v, Dim, data)
end

"""
    _build_weights(ℒ::Directional, data, eval_points, adjl, basis, is_boundary, boundary_conditions, normals)

Hermite-compatible method for building directional derivative weights with boundary condition support.
"""
function _build_weights(
    ℒ::Directional{Dim},
    data::AbstractVector,
    eval_points::AbstractVector,
    adjl::AbstractVector,
    basis::AbstractRadialBasis,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector},
) where {Dim}
    v = ℒ.v
    _validate_directional_vector(v, Dim, data)

    # Build gradient weights using Hermite method
    dim = length(first(data))
    mon = MonomialBasis(dim, basis.poly_deg)
    gradient_op = Gradient{Dim}()
    ℒmon = gradient_op(mon)
    ℒrbf = gradient_op(basis)

    weights = _build_weights(
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

    return _reduce_gradient_to_directional(weights, v, Dim, data)
end

# pretty printing
print_op(op::Directional) = "Directional Derivative (∇f⋅v)"
