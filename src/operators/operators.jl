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

# Hermite-compatible constructor
function RadialBasisOperator(
    ℒ,
    data::AbstractVector{TD},
    eval_points::AbstractVector{TE},
    basis::B,
    is_boundary::Vector{Bool},
    boundary_conditions::Vector{<:BoundaryCondition},
    normals::Vector{<:AbstractVector};
    k::T=autoselect_k(data, basis),
    adjl=find_neighbors(data, eval_points, k),
) where {TD,TE,T<:Int,B<:AbstractRadialBasis}
    weights = _build_weights(
        ℒ, data, eval_points, adjl, basis, is_boundary, boundary_conditions, normals
    )
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

# VectorValuedOperator: Scalar field input → Matrix output (N×D)
function _eval_op(
    op::RadialBasisOperator{<:VectorValuedOperator{D}}, x::AbstractVector
) where {D}
    N_eval = length(op.eval_points)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = Matrix{T}(undef, N_eval, D)
    for d in 1:D
        out[:, d] = op.weights[d] * x
    end
    return out
end

# VectorValuedOperator: Vector field input (N×D_in) → 3-tensor output (N×D_in×D)
function _eval_op(
    op::RadialBasisOperator{<:VectorValuedOperator{D}}, x::AbstractMatrix
) where {D}
    N_eval = length(op.eval_points)
    D_in = size(x, 2)
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = Array{T,3}(undef, N_eval, D_in, D)
    for d_out in 1:D, d_in in 1:D_in
        out[:, d_in, d_out] = op.weights[d_out] * view(x, :, d_in)
    end
    return out
end

# VectorValuedOperator: General tensor input → tensor output with extra dimension
function _eval_op(
    op::RadialBasisOperator{<:VectorValuedOperator{D}}, x::AbstractArray
) where {D}
    N_eval = length(op.eval_points)
    trailing_dims = size(x)[2:end]
    T = promote_type(eltype(x), eltype(first(op.weights)))
    out = Array{T}(undef, N_eval, trailing_dims..., D)
    for idx in CartesianIndices(trailing_dims), d in 1:D
        out[:, idx, d] = op.weights[d] * view(x, :, idx)
    end
    return out
end

# In-place: Scalar field → Matrix
function _eval_op(
    op::RadialBasisOperator{<:VectorValuedOperator{D}}, y::AbstractMatrix, x::AbstractVector
) where {D}
    for d in 1:D
        mul!(view(y, :, d), op.weights[d], x)
    end
    return y
end

# In-place: Vector field → 3-tensor
function _eval_op(
    op::RadialBasisOperator{<:VectorValuedOperator{D}},
    y::AbstractArray{<:Any,3},
    x::AbstractMatrix,
) where {D}
    D_in = size(x, 2)
    for d_out in 1:D, d_in in 1:D_in
        mul!(view(y, :, d_in, d_out), op.weights[d_out], view(x, :, d_in))
    end
    return y
end

# LinearAlgebra methods - divergence (dot with gradient operator)
function LinearAlgebra.:⋅(
    op::RadialBasisOperator{<:VectorValuedOperator}, x::AbstractVector
)
    !is_cache_valid(op) && update_weights!(op)
    result = op(x)  # Now Matrix (N×D)
    return vec(sum(result; dims=2))  # Sum across derivative dimension → Vector (N,)
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
