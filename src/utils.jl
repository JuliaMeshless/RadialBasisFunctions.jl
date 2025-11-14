"""
    find_neighbors(data::AbstractVector, k::Int; metric::Metric=Euclidean())

Find k-nearest neighbors for each point in `data` using the specified distance metric.

# Arguments
- `data::AbstractVector`: Vector of points (each point should be an AbstractVector)
- `k::Int`: Number of nearest neighbors to find
- `metric::Metric`: Distance metric to use (default: Euclidean())

# Returns
- `adjl`: Adjacency list where adjl[i] contains indices of k nearest neighbors of point i
"""
function find_neighbors(data::AbstractVector, k::Int; metric::M=Euclidean()) where {M<:Metric}
    tree = KDTree(data, metric)
    adjl, _ = knn(tree, data, k, true)
    return adjl
end

"""
    find_neighbors(data::AbstractVector, eval_points::AbstractVector, k::Int; metric::Metric=Euclidean())

Find k-nearest neighbors in `data` for each point in `eval_points` using the specified distance metric.

# Arguments
- `data::AbstractVector`: Vector of data points
- `eval_points::AbstractVector`: Vector of evaluation points
- `k::Int`: Number of nearest neighbors to find
- `metric::Metric`: Distance metric to use (default: Euclidean())

# Returns
- `adjl`: Adjacency list where adjl[i] contains indices of k nearest neighbors for eval_points[i]
"""
function find_neighbors(data::AbstractVector, eval_points::AbstractVector, k::Int; metric::M=Euclidean()) where {M<:Metric}
    tree = KDTree(data, metric)
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
