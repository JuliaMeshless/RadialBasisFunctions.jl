struct BoundaryType{T<:Real}
    coefficients::Vector{T}
end

α(bt::BoundaryType) = bt.coefficients[1]
β(bt::BoundaryType) = bt.coefficients[2]

function Base.show(io::IO, bt::BoundaryType)
    return print(io, "BoundaryType(α=$(bt.coefficients[1]), β=$(bt.coefficients[2]))")
end

# Classification helpers (exact comparison; adjust if using non-integer floats)
function is_Neumann(bt::BoundaryType)
    return bt.coefficients[1] == zero(bt.coefficients[1]) &&
           bt.coefficients[2] == one(bt.coefficients[2])
end
function is_Dirichlet(bt::BoundaryType)
    return bt.coefficients[1] == one(bt.coefficients[1]) &&
           bt.coefficients[2] == zero(bt.coefficients[2])
end
function is_Robin(bt::BoundaryType)
    return bt.coefficients[1] > zero(bt.coefficients[1]) &&
           bt.coefficients[2] > zero(bt.coefficients[2])
end

function _set_to_zero!(bt::BoundaryType)
    bt.coefficients .= zero(bt.coefficients[1])
    return nothing
end

function _set_to_zero!(boundary_types::AbstractVector{<:BoundaryType})
    for bt in boundary_types
        _set_to_zero!(bt)
    end
    return nothing
end

function _init_boundary_types(::Type{T}, n::Integer) where {T<:Real}
    return [BoundaryType(T[zero(T), zero(T)]) for _ in 1:n]
end

"""In-place helpers for common boundary condition presets."""
function set_Dirichlet!(bt::BoundaryType)
    bt.coefficients[1] = one(bt.coefficients[1])
    bt.coefficients[2] = zero(bt.coefficients[2])
    return nothing
end
function set_Neumann!(bt::BoundaryType)
    bt.coefficients[1] = zero(bt.coefficients[1])
    bt.coefficients[2] = one(bt.coefficients[2])
    return nothing
end
function set_Robin!(bt::BoundaryType, α::Real, β::Real)
    a, b = promote(float(α), float(β))
    bt.coefficients[1] = a
    bt.coefficients[2] = b
    return nothing
end