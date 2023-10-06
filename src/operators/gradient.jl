"""
    Gradient <: VectorValuedOperator

Builds an operator for the gradient of a function.
"""
struct Gradient{L<:NTuple} <: VectorValuedOperator
    ℒ::L
end

# convienience constructors
function gradient(
    data::AbstractVector{D},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    sparse=true,
) where {D<:AbstractArray,B<:AbstractRadialBasis,T<:Int}
    f = ntuple(length(first(data))) do dim
        return let dim = dim
            x -> ∂(x, 1, dim)
        end
    end
    ℒ = Gradient(f)
    return RadialBasisOperator(ℒ, data, basis; k=k, sparse=sparse)
end

function gradient(
    data::AbstractVector{D},
    centers::AbstractVector{D},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    sparse=true,
) where {D<:AbstractArray,B<:AbstractRadialBasis,T<:Int}
    f = ntuple(length(first(data))) do dim
        return let dim = dim
            x -> ∂(x, 1, dim)
        end
    end
    ℒ = Gradient(f)
    return RadialBasisOperator(ℒ, data, centers, basis; k=k, sparse=sparse)
end

function RadialBasisOperator(
    ℒ::Gradient,
    data::AbstractVector{D},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    sparse=true,
) where {D<:AbstractArray,T<:Int,B<:AbstractRadialBasis}
    TD = eltype(D)
    adjl = find_neighbors(data, k)
    N = length(adjl)
    weights = ntuple(_ -> _allocate_weights(TD, N, N, k; sparse=sparse), length(ℒ.ℒ))
    return RadialBasisOperator(ℒ, weights, data, data, adjl, basis)
end

function RadialBasisOperator(
    ℒ::Gradient,
    data::AbstractVector{D},
    centers::AbstractVector{D},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    sparse=true,
) where {D<:AbstractArray,T<:Int,B<:AbstractRadialBasis}
    TD = eltype(D)
    adjl = find_neighbors(data, centers, k)
    Na = length(adjl)
    Nd = length(data)
    weights = ntuple(_ -> _allocate_weights(TD, Na, Nd, k; sparse=sparse), length(ℒ.ℒ))
    return RadialBasisOperator(ℒ, weights, data, centers, adjl, basis)
end

Base.size(op::RadialBasisOperator{<:Gradient}) = size(first(op.weights))

# pretty printing
print_op(op::Gradient) = "Gradient (∇)"