# GPU → CPU conversion for KDTree compatibility
_to_cpu(x::Vector) = x
_to_cpu(x::AbstractVector) = Array(x)

function find_neighbors(data::AbstractVector, k::Int)
    cpu_data = _to_cpu(data)
    tree = KDTree(cpu_data)
    adjl, _ = knn(tree, cpu_data, k, true)
    return adjl
end

function find_neighbors(data::AbstractVector, eval_points::AbstractVector, k::Int)
    cpu_data = _to_cpu(data)
    cpu_eval = _to_cpu(eval_points)
    tree = KDTree(cpu_data)
    adjl, _ = knn(tree, cpu_eval, k, true)
    return adjl
end

# Helper function to get vector dimension without scalar indexing
function _get_vector_dim(::Type{SVector{N, T}}) where {N, T}
    return N
end

function _get_vector_dim(data::AbstractVector)
    return _get_vector_dim(eltype(data))
end

"""
    autoselect_k(data::Vector, basis<:AbstractRadialBasis)

See Bayona, 2017 - https://doi.org/10.1016/j.jcp.2016.12.008
"""
function autoselect_k(data::AbstractVector, basis::B) where {B <: AbstractRadialBasis}
    m = basis.poly_deg
    d = _get_vector_dim(data)
    return min(length(data), max(2 * binomial(m + d, d), 2 * d + 1))
end

function reorder_points!(
        x::AbstractVector, adjl::AbstractVector{AbstractVector{T}}, k::T
    ) where {T <: Int}
    i = symrcm(adjl, ones(T, length(x)) .* k)
    permute!(x, i)
    return nothing
end

function reorder_points!(x::AbstractVector, k::T) where {T <: Int}
    return reorder_points!(x, find_neighbors(x, k), k)
end

function check_poly_deg(poly_deg)
    if poly_deg < -1
        throw(
            ArgumentError(
                "poly_deg must be >= -1 (got $poly_deg). Use poly_deg=2 (default) for quadratic, poly_deg=0 for constant, or poly_deg=-1 to disable polynomial augmentation.",
            ),
        )
    end
    return nothing
end

_get_underlying_type(x::AbstractVector) = eltype(x)
_get_underlying_type(x::Number) = typeof(x)
