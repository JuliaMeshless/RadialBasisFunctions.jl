function find_neighbors(data::AbstractVector, k::Int)
    tree = KDTree(data)
    adjl, _ = knn(tree, data, k, true)
    return adjl
end

function find_neighbors(data::AbstractVector, eval_points::AbstractVector, k::Int)
    tree = KDTree(data)
    adjl, _ = knn(tree, eval_points, k, true)
    return adjl
end

# Helper function to get vector dimension without scalar indexing
function _get_vector_dim(::Type{SVector{N,T}}) where {N,T}
    return N
end

function _get_vector_dim(data::AbstractVector)
    return _get_vector_dim(eltype(data))
end


"""
    autoselect_k(data::Vector, basis<:AbstractRadialBasis)

See Bayona, 2017 - https://doi.org/10.1016/j.jcp.2016.12.008
"""
function autoselect_k(data::AbstractVector, basis::B) where {B<:AbstractRadialBasis}
    m = basis.poly_deg
    d = _get_vector_dim(data)
    return min(length(data), max(2 * binomial(m + d, d), 2 * d + 1))
end

function reorder_points!(
    x::AbstractVector, adjl::AbstractVector{AbstractVector{T}}, k::T
) where {T<:Int}
    i = symrcm(adjl, ones(T, length(x)) .* k)
    permute!(x, i)
    return nothing
end

function reorder_points!(x::AbstractVector, k::T) where {T<:Int}
    return reorder_points!(x, find_neighbors(x, k), k)
end

function check_poly_deg(poly_deg)
    if poly_deg < -1
        throw(ArgumentError("Augmented Monomial degree must be at least 0 (constant)."))
    end
    return nothing
end

_get_underlying_type(x::AbstractVector) = eltype(x)
_get_underlying_type(x::Number) = typeof(x)
