########################################################################################
# Polyharmonic Spline

"""
   abstract type AbstractPHS <: AbstractRadialBasis

Supertype of all Polyharmonic Splines.
"""
abstract type AbstractPHS <: AbstractRadialBasis end

"""
    function PHS(n::T=3; poly_deg::T=2) where {T<:Int}

Convienience contructor for polyharmonic splines.
"""
function PHS(n::T=3; poly_deg::T=2) where {T<:Int}
    check_poly_deg(poly_deg)
    if iseven(n) || n > 7
        throw(ArgumentError("n must be 1, 3, 5, or 7. (n = $n)"))
    end
    n == 1 && return PHS1(poly_deg)
    n == 3 && return PHS3(poly_deg)
    n == 5 && return PHS5(poly_deg)
    return PHS7(poly_deg)
end

for phs in (:PHS1, :PHS3, :PHS5, :PHS7)
    @eval function $phs(; poly_deg::Int=2)
        return $phs(poly_deg)
    end
end

"""
    struct PHS1{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r``
"""
struct PHS1{T<:Int} <: AbstractPHS
    poly_deg::T
    function PHS1(poly_deg::T) where {T<:Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end

(phs::PHS1)(x, xᵢ) = euclidean(x, xᵢ)
function ∂(::PHS1, dim::Int)
    ∂ℒ(x, xᵢ) = (x[dim] - xᵢ[dim]) / (euclidean(x, xᵢ) + AVOID_INF)
    return ∂ℒ
end
function ∇(::PHS1)
    ∇ℒ(x, xᵢ) = (x .- xᵢ) / euclidean(x, xᵢ)
    return ∇ℒ
end
function ∂²(::PHS1, dim::Int)
    function ∂²ℒ(x, xᵢ)
        return (-(x[dim] - xᵢ[dim])^2 + sqeuclidean(x, xᵢ)) /
               (euclidean(x, xᵢ)^3 + AVOID_INF)
    end
    return ∂²ℒ
end
function ∇²(::PHS1)
    function ∇²ℒ(x, xᵢ)
        return sum(
            (-(x .- xᵢ) .^ 2 .+ sqeuclidean(x, xᵢ)) / (euclidean(x, xᵢ)^3 + AVOID_INF)
        )
    end
    return ∇²ℒ
end

"""
    struct PHS3{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r^3``
"""
struct PHS3{T<:Int} <: AbstractPHS
    poly_deg::T
    function PHS3(poly_deg::T) where {T<:Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end

(phs::PHS3)(x, xᵢ) = euclidean(x, xᵢ)^3
function ∂(::PHS3, dim::Int)
    ∂ℒ(x, xᵢ) = 3 * (x[dim] - xᵢ[dim]) * euclidean(x, xᵢ)
    return ∂ℒ
end
function ∇(::PHS3)
    ∇ℒ(x, xᵢ) = 3 * (x .- xᵢ) * euclidean(x, xᵢ)
    return ∇ℒ
end
function ∂²(::PHS3, dim::Int)
    function ∂²ℒ(x, xᵢ)
        return 3 * (sqeuclidean(x, xᵢ) + (x[dim] - xᵢ[dim])^2) /
               (euclidean(x, xᵢ) + AVOID_INF)
    end
    return ∂²ℒ
end
function ∇²(::PHS3)
    function ∇²ℒ(x, xᵢ)
        return sum(
            3 * (sqeuclidean(x, xᵢ) .+ (x .- xᵢ) .^ 2) / (euclidean(x, xᵢ) + AVOID_INF)
        )
    end
    return ∇²ℒ
end

"""
    struct PHS5{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r^5``
"""
struct PHS5{T<:Int} <: AbstractPHS
    poly_deg::T
    function PHS5(poly_deg::T) where {T<:Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end

(phs::PHS5)(x, xᵢ) = euclidean(x, xᵢ)^5
function ∂(::PHS5, dim::Int)
    ∂ℒ(x, xᵢ) = 5 * (x[dim] - xᵢ[dim]) * euclidean(x, xᵢ)^3
    return ∂ℒ
end
function ∇(::PHS5)
    ∇ℒ(x, xᵢ) = 5 * (x .- xᵢ) * euclidean(x, xᵢ)^3
    return ∇ℒ
end
function ∂²(::PHS5, dim::Int)
    function ∂²ℒ(x, xᵢ)
        return 5 * euclidean(x, xᵢ) * (3 * (x[dim] - xᵢ[dim])^2 + sqeuclidean(x, xᵢ))
    end
    return ∂²ℒ
end
function ∇²(::PHS5)
    function ∇²ℒ(x, xᵢ)
        return sum(5 * euclidean(x, xᵢ) * (3 * (x .- xᵢ) .^ 2 .+ sqeuclidean(x, xᵢ)))
    end
    return ∇²ℒ
end

"""
    struct PHS7{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r^7``
"""
struct PHS7{T<:Int} <: AbstractPHS
    poly_deg::T
    function PHS7(poly_deg::T) where {T<:Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end

(phs::PHS7)(x, xᵢ) = euclidean(x, xᵢ)^7
function ∂(::PHS7, dim::Int)
    ∂ℒ(x, xᵢ) = 7 * (x[dim] - xᵢ[dim]) * euclidean(x, xᵢ)^5
    return ∂ℒ
end
function ∇(::PHS7)
    ∇ℒ(x, xᵢ) = 7 * (x .- xᵢ) * euclidean(x, xᵢ)^5
    return ∇ℒ
end
function ∂²(::PHS7, dim::Int)
    function ∂²ℒ(x, xᵢ)
        return 7 * euclidean(x, xᵢ)^3 * (5 * (x[dim] - xᵢ[dim])^2 + sqeuclidean(x, xᵢ))
    end
    return ∂²ℒ
end
function ∇²(::PHS7)
    function ∇²ℒ(x, xᵢ)
        return sum(7 * euclidean(x, xᵢ)^3 * (5 * (x .- xᵢ) .^ 2 .+ sqeuclidean(x, xᵢ)))
    end
    return ∇²ℒ
end

function Base.show(io::IO, rbf::R) where {R<:AbstractPHS}
    print(io, print_basis(rbf))
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end

print_basis(::PHS1) = "Polyharmonic spline (r¹)"
print_basis(::PHS3) = "Polyharmonic spline (r³)"
print_basis(::PHS5) = "Polyharmonic spline (r⁵)"
print_basis(::PHS7) = "Polyharmonic spline (r⁷)"
