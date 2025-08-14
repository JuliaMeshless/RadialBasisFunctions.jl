struct BoundaryType{T<:Real}
    α::T
    β::T
    function BoundaryType(α::A, β::B) where {A,B}
        T = promote_type(A, B)
        new{T}(α, β)
    end
end

# Constructor for array input (backward compatibility)
function BoundaryType(coeffs::AbstractVector)
    length(coeffs) == 2 || throw(ArgumentError("BoundaryType requires exactly 2 coefficients"))
    return BoundaryType(coeffs[1], coeffs[2])
end

α(bt::BoundaryType) = bt.α
β(bt::BoundaryType) = bt.β

function Base.show(io::IO, bt::BoundaryType)
    return print(io, "BoundaryType(α=$(bt.α), β=$(bt.β))")
end

function is_Neumann(bt::BoundaryType)
    return iszero(bt.α) && isone(bt.β)
end
function is_Dirichlet(bt::BoundaryType)
    return isone(bt.α) && iszero(bt.β)
end
function is_Robin(bt::BoundaryType)
    return !iszero(bt.α) && !iszero(bt.β)
end

# Note: Since the new struct is immutable, we can't provide in-place modification functions
# Users should create new BoundaryType instances instead
function _init_boundary_types(::Type{T}, n::Integer) where {T<:Real}
    return [BoundaryType(zero(T), zero(T)) for _ in 1:n]
end

"""Helper constructors for common boundary condition presets."""
function Dirichlet(::Type{T}=Float64) where {T<:Real}
    return BoundaryType(one(T), zero(T))
end
function Neumann(::Type{T}=Float64) where {T<:Real}
    return BoundaryType(zero(T), one(T))
end
function Robin(α::Real, β::Real)
    return BoundaryType(α, β)
end
