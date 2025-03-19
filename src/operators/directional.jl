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

function _build_weights(ℒ::Directional{Dim}, data, eval_points, adjl, basis) where {Dim}
    v = ℒ.v
    if !(length(v) == Dim || length(v) == length(data))
        throw(
            DomainError(
                "The geometrical vector for Directional() should match either the dimension of the input or the number of input points. The geometrical vector length is $(length(v)) while there are $(length(data)) points with a dimension of $Dim",
            ),
        )
    end
    weights = _build_weights(Gradient{Dim}(), data, eval_points, adjl, basis)

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

# pretty printing
print_op(op::Directional) = "Directional Derivative (∇f⋅v)"
