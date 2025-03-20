abstract type AbstractOperator end
abstract type ScalarValuedOperator <: AbstractOperator end
abstract type VectorValuedOperator{Dim} <: AbstractOperator end

"""
    struct RadialBasisOperator

Operator of data using a radial basis with potential monomial augmentation.
"""
struct RadialBasisOperator{L,W,D,C,A,B<:AbstractRadialBasis}
    ℒ::L
    weights::W
    data::D
    eval_points::C
    adjl::A
    basis::B
    valid_cache::Base.RefValue{Bool}
    function RadialBasisOperator(
        ℒ::L,
        weights::W,
        data::D,
        eval_points::C,
        adjl::A,
        basis::B,
        cache_status::Bool=false,
    ) where {L,W,D,C,A,B<:AbstractRadialBasis}
        return new{L,W,D,C,A,B}(
            ℒ, weights, data, eval_points, adjl, basis, Ref(cache_status)
        )
    end
end

# convienience constructors
function RadialBasisOperator(
    ℒ,
    data::AbstractVector,
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, k),
) where {T<:Int,B<:AbstractRadialBasis}
    weights = _build_weights(ℒ, data, data, adjl, basis)
    return RadialBasisOperator(ℒ, weights, data, data, adjl, basis, true)
end

function RadialBasisOperator(
    ℒ,
    data::AbstractVector{TD},
    eval_points::AbstractVector{TE},
    basis::B=PHS(3; poly_deg=2);
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {TD,TE,T<:Int,B<:AbstractRadialBasis}
    weights = _build_weights(ℒ, data, eval_points, adjl, basis)
    return RadialBasisOperator(ℒ, weights, data, eval_points, adjl, basis, true)
end

dim(op::RadialBasisOperator) = length(first(op.data))

# caching
invalidate_cache!(op::RadialBasisOperator) = op.valid_cache[] = false
validate_cache!(op::RadialBasisOperator) = op.valid_cache[] = true
is_cache_valid(op::RadialBasisOperator) = op.valid_cache[]

"""
    function (op::RadialBasisOperator)(x)

Evaluate the operator at `x`.
"""
function (op::RadialBasisOperator)(x)
    !is_cache_valid(op) && update_weights!(op)
    return _eval_op(op, x)
end

"""
    function (op::RadialBasisOperator)(y, x)

Evaluate the operator at `x` in-place and store the result in `y`.
"""
function (op::RadialBasisOperator)(y, x)
    !is_cache_valid(op) && update_weights!(op)
    return _eval_op(op, y, x)
end

# dispatches for evaluation
_eval_op(op::RadialBasisOperator, x) = op.weights * x
_eval_op(op::RadialBasisOperator, y, x) = mul!(y, op.weights, x)

function _eval_op(op::RadialBasisOperator{<:VectorValuedOperator}, x)
    return ntuple(i -> op.weights[i] * x, dim(op))
end
function _eval_op(op::RadialBasisOperator{<:VectorValuedOperator}, y, x)
    for i in eachindex(op.weights)
        mul!(y[i], op.weights[i], x)
    end
end

# LinearAlgebra methods
function LinearAlgebra.:⋅(
    op::RadialBasisOperator{<:VectorValuedOperator}, x::AbstractVector
)
    !is_cache_valid(op) && update_weights!(op)
    return sum(op(x))
end

# update weights
function update_weights!(op::RadialBasisOperator)
    op.weights .= _build_weights(op.ℒ, op)
    validate_cache!(op)
    return nothing
end

function update_weights!(op::RadialBasisOperator{<:VectorValuedOperator{Dim}}) where {Dim}
    new_weights = _build_weights(op.ℒ, op)
    for i in 1:Dim
        op.weights[i] .= new_weights[i]
    end
    validate_cache!(op)
    return nothing
end

# pretty printing
function Base.show(io::IO, op::RadialBasisOperator)
    println(io, "RadialBasisOperator")
    println(io, "├─Operator: " * print_op(op.ℒ))
    println(io, "├─Data type: ", typeof(first(op.data)))
    println(io, "├─Number of points: ", length(op.data))
    println(io, "├─Stencil size: ", length(first(op.adjl)))
    return println(
        io,
        "└─Basis: ",
        print_basis(op.basis),
        " with degree $(op.basis.poly_deg) polynomial augmentation",
    )
end
