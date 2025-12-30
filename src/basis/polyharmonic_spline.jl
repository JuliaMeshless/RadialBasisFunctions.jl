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

function ∂(::PHS1, dim::Int, ::Nothing=nothing)
    function ∂ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return (x[dim] - xᵢ[dim]) / (r + AVOID_INF)
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -normal[dim] / (r + AVOID_INF) +
                   dot_normal * (x[dim] - xᵢ[dim]) / (r^3 + AVOID_INF)
        end
    end
    return ∂ℒ
end

function ∇(::PHS1, ::Nothing=nothing)
    function ∇ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return (x .- xᵢ) / (r + AVOID_INF)
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -normal / (r + AVOID_INF) +
                   dot_normal * (x .- xᵢ) / (r^3 + AVOID_INF)
        end
    end
    return ∇ℒ
end

function directional∂²(::PHS1, v1::AbstractVector, v2::AbstractVector)
    function directional₂ℒ(x, xᵢ)
        r = euclidean(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -dot_v1_v2 / (r + AVOID_INF) + (dot_v1_r * dot_v2_r) / (r^3 + AVOID_INF)
    end
    return directional₂ℒ
end

function ∂²(::PHS1, dim::Int, ::Nothing=nothing)
    function ∂²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            r² = sqeuclidean(x, xᵢ)
            return (-(x[dim] - xᵢ[dim])^2 + r²) / (r^3 + AVOID_INF)
        else
            n_d = normal[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return (2 * n_d * Δ_d + dot_normal) / (r^3 + AVOID_INF) -
                   3 * Δ_d^2 * dot_normal / (r^5 + AVOID_INF)
        end
    end
    return ∂²ℒ
end

function ∇²(::PHS1, ::Nothing=nothing)
    function ∇²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 2 / (r + AVOID_INF)
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return 2 * dot_normal / (r^3 + AVOID_INF)
        end
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

function ∂(::PHS3, dim::Int, ::Nothing=nothing)
    function ∂ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 3 * (x[dim] - xᵢ[dim]) * r
        else
            n_d = normal[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -3 * (n_d * r + dot_normal * Δ_d / (r + AVOID_INF))
        end
    end
    return ∂ℒ
end

function ∇(::PHS3, ::Nothing=nothing)
    function ∇ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 3 * (x .- xᵢ) * r
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -3 * (normal * r + dot_normal * (x .- xᵢ) / (r + AVOID_INF))
        end
    end
    return ∇ℒ
end

function directional∂²(::PHS3, v1::AbstractVector, v2::AbstractVector)
    function directional₂ℒ(x, xᵢ)
        r = euclidean(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -3 * (dot_v1_v2 * r + dot_v1_r * dot_v2_r / (r + AVOID_INF))
    end
    return directional₂ℒ
end

function ∂²(::PHS3, dim::Int, ::Nothing=nothing)
    function ∂²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 3 * (r + (x[dim] - xᵢ[dim])^2 / (r + AVOID_INF))
        else
            r² = sqeuclidean(x, xᵢ)
            n_d = normal[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -3 *
                   (2 * Δ_d * n_d + dot_normal - dot_normal * Δ_d^2 / (r² + AVOID_INF)) /
                   (r + AVOID_INF)
        end
    end
    return ∂²ℒ
end

function ∇²(::PHS3, ::Nothing=nothing)
    function ∇²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 12 * r
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -12 * dot_normal / (r + AVOID_INF)
        end
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

function ∂(::PHS5, dim::Int, ::Nothing=nothing)
    function ∂ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 5 * (x[dim] - xᵢ[dim]) * r^3
        else
            n_d = normal[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -5 * (n_d * r^3 + 3 * dot_normal * Δ_d * r)
        end
    end
    return ∂ℒ
end

function ∇(::PHS5, ::Nothing=nothing)
    function ∇ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 5 * (x .- xᵢ) * r^3
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -5 * (normal * r^3 + 3 * dot_normal * (x .- xᵢ) * r)
        end
    end
    return ∇ℒ
end

function directional∂²(::PHS5, v1::AbstractVector, v2::AbstractVector)
    function directional₂ℒ(x, xᵢ)
        r = euclidean(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -5 * (dot_v1_v2 * r^3 + 3 * dot_v1_r * dot_v2_r * r)
    end
    return directional₂ℒ
end

function ∂²(::PHS5, dim::Int, ::Nothing=nothing)
    function ∂²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            r² = sqeuclidean(x, xᵢ)
            return 5 * r * (3 * (x[dim] - xᵢ[dim])^2 + r²)
        else
            n_d = normal[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -5 * (6 * n_d * Δ_d * r + 3 * dot_normal * (r + Δ_d^2 / (r + AVOID_INF)))
        end
    end
    return ∂²ℒ
end

function ∇²(::PHS5, ::Nothing=nothing)
    function ∇²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 30 * r^3
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -90 * dot_normal * r
        end
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

function ∂(::PHS7, dim::Int, ::Nothing=nothing)
    function ∂ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 7 * (x[dim] - xᵢ[dim]) * r^5
        else
            n_d = normal[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -7 * (n_d * r^5 + 5 * r^3 * Δ_d * dot_normal)
        end
    end
    return ∂ℒ
end

function ∇(::PHS7, ::Nothing=nothing)
    function ∇ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 7 * (x .- xᵢ) * r^5
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -7 * (normal * r^5 + 5 * dot_normal * (x .- xᵢ) * r^3)
        end
    end
    return ∇ℒ
end

function directional∂²(::PHS7, v1::AbstractVector, v2::AbstractVector)
    function directional₂ℒ(x, xᵢ)
        r = euclidean(x, xᵢ)
        dot_v1_v2 = LinearAlgebra.dot(v1, v2)
        dot_v1_r = LinearAlgebra.dot(v1, x .- xᵢ)
        dot_v2_r = LinearAlgebra.dot(v2, x .- xᵢ)
        return -7 * (dot_v1_v2 * r^5 + 5 * dot_v1_r * dot_v2_r * r^3)
    end
    return directional₂ℒ
end

function ∂²(::PHS7, dim::Int, ::Nothing=nothing)
    function ∂²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            r² = sqeuclidean(x, xᵢ)
            return 7 * r^3 * (5 * (x[dim] - xᵢ[dim])^2 + r²)
        else
            n_d = normal[dim]
            Δ_d = x[dim] - xᵢ[dim]
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -7 * (10 * n_d * Δ_d * r^3 + 5 * dot_normal * (3 * r * Δ_d^2 + r^3))
        end
    end
    return ∂²ℒ
end

function ∇²(::PHS7, ::Nothing=nothing)
    function ∇²ℒ(x, xᵢ, normal=nothing)
        r = euclidean(x, xᵢ)
        if normal === nothing
            return 56 * r^5
        else
            dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
            return -280 * dot_normal * r^3
        end
    end
    return ∇²ℒ
end

# convient constructors using keyword arguments
for phs in (:PHS1, :PHS3, :PHS5, :PHS7)
    @eval function $phs(; poly_deg::Int=2)
        return $phs(poly_deg)
    end
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
