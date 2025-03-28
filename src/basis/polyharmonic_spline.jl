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
function ∂(::PHS1, dim::Int)
    ∂ℒ(x, xᵢ) = (x[dim] - xᵢ[dim]) / (euclidean(x, xᵢ) + AVOID_INF)
    return ∂ℒ
end
function ∂(::PHS1, dim::Int, normal::AbstractVector) #must check
    function ∂₂ℒ(x, xᵢ, normal)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -normal[dim] / (r + AVOID_INF) +
               dot_normal * (x[dim] - xᵢ[dim]) / (r^3 + AVOID_INF)
    end
    return ∂₂ℒ
end
function ∇(::PHS1)
    ∇ℒ(x, xᵢ) = (x .- xᵢ) / euclidean(x, xᵢ)
    return ∇ℒ
end
function ∇(::PHS1, normal::AbstractVector) #must check
    function ∇₂ℒ(x, xᵢ, normal)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -normal / (r + AVOID_INF) + dot_normal * (x .- xᵢ) / (r^3 + AVOID_INF)
    end
    return ∇₂ℒ
end
function directional∂²(::PHS1, v1::AbstractVector, v2::AbstractVector)
    function directional₂ℒ(x, xᵢ)
        return -LinearAlgebra.dot(v1, v2) / (euclidean(x, xᵢ) + AVOID_INF) +
               LinearAlgebra.dot(v1, x .- xᵢ) * LinearAlgebra.dot(v2, x .- xᵢ) /
               (sqeuclidean(x, xᵢ) + AVOID_INF)
    end
    return directional₂ℒ
end
function ∂²(::PHS1, dim::Int)
    function ∂²ℒ(x, xᵢ)
        return (-(x[dim] - xᵢ[dim])^2 + sqeuclidean(x, xᵢ)) /
               (euclidean(x, xᵢ)^3 + AVOID_INF)
    end
    return ∂²ℒ
end
function ∂²(::PHS1, dim::Int, normal::AbstractVector) #must check
    function ∂²ℒ(x, xᵢ, normal)
        n_d = normal[dim]
        Δ_d = x[dim] - xᵢ[dim]
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return (2 * n_d * Δ_d + dot_normal * (1 - 3 * Δ_d / (r^2 + AVOID_INF))) /
               (r^3 + AVOID_INF)
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
function ∇²(::PHS1, normal::AbstractVector)
    function ∇²ℒ(x, xᵢ, normal)
        r = euclidean(x, xᵢ)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        return 2 * dot_normal / (r^3 + AVOID_INF)
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
function ∂(::PHS3, dim::Int, normal::AbstractVector) #must check
    function ∂₂ℒ(x, xᵢ, normal)
        n_d = normal[dim]
        Δ_d = x[dim] - xᵢ[dim]
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -3 * (n_d * r + dot_normal * Δ_d / (r + AVOID_INF))
    end
    return ∂₂ℒ
end
function ∇(::PHS3)
    ∇ℒ(x, xᵢ) = 3 * (x .- xᵢ) * euclidean(x, xᵢ)
    return ∇ℒ
end
function ∇(::PHS3, normal::AbstractVector) #must check
    function ∇₂ℒ(x, xᵢ, normal)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        return -3 * (normal * r .+ dot_normal * (x .- xᵢ) ./ (r + AVOID_INF))
    end
    return ∇₂ℒ
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
function ∂²(::PHS3, dim::Int)
    function ∂²ℒ(x, xᵢ)
        r = euclidean(x, xᵢ)
        return 3 * (r + (x[dim] - xᵢ[dim])^2 / (r + AVOID_INF))
    end
    return ∂²ℒ
end
function ∂²(::PHS3, dim::Int, normal::AbstractVector) #must check
    function ∂²ℒ(x, xᵢ, normal)
        n_d = normal[dim]
        Δ_d = x[dim] - xᵢ[dim]
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -3 * (2 * n_d + dot_normal - dot_normal * Δ_d^2 / (r^2 + AVOID_INF)) /
               (r + AVOID_INF)
    end
    return ∂²ℒ
end
function ∇²(::PHS3)
    function ∇²ℒ(x, xᵢ)
        return 12 * euclidean(x, xᵢ)
        # return sum(
        #     3 * (sqeuclidean(x, xᵢ) .+ (x .- xᵢ) .^ 2) / (euclidean(x, xᵢ) + AVOID_INF)
        # )
    end
    return ∇²ℒ
end
function ∇²(::PHS3, normal::AbstractVector) #must check
    function ∇²ℒ(x, xᵢ, normal)
        return -6 * LinearAlgebra.dot(normal, x .- xᵢ) / (euclidean(x, xᵢ) + AVOID_INF)
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
function ∂(::PHS5, dim::Int, normal::AbstractVector)
    function ∂ℒ(x, xᵢ, normal)
        n_d = normal[dim]
        Δ_d = x[dim] - xᵢ[dim]
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -5 * (n_d * r^3 + 3 * dot_normal * Δ_d * r)
    end
    return ∂ℒ
end
function ∇(::PHS5)
    ∇ℒ(x, xᵢ) = 5 * (x .- xᵢ) * euclidean(x, xᵢ)^3
    return ∇ℒ
end
function ∇(::PHS5, normal::AbstractVector)
    function ∇ℒ(x, xᵢ, normal)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        return -5 * (normal * r^3 + 3 * dot_normal * (x .- xᵢ) * r)
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
function ∂²(::PHS5, dim::Int)
    function ∂²ℒ(x, xᵢ)
        r = euclidean(x, xᵢ)
        r² = sqeuclidean(x, xᵢ)
        return 5 * r * (3 * (x[dim] - xᵢ[dim])^2 + r²)
    end
    return ∂²ℒ
end
function ∂²(::PHS5, dim::Int, normal::AbstractVector) #must check
    function ∂²ℒ(x, xᵢ, normal)
        n_d = normal[dim]
        Δ_d = x[dim] - xᵢ[dim]
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -5 * (6 * n_d * Δ_d * r + 3 * dot_normal * (r + Δ_d^2 / (r + AVOID_INF)))
    end
    return ∂²ℒ
end
function ∂²(::PHS5, dim::Int, normal) #must check
    function ∂²ℒ(x, xᵢ, normal)
        n_d = normal[dim]
        Δ_d = x[dim] - xᵢ[dim]
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -5 * (6 * n_d * Δ_d * r + 3 * dot_normal * (r + Δ_d^2 / (r + AVOID_INF)))
    end
    return ∂²ℒ
end
function ∇²(::PHS5)
    function ∇²ℒ(x, xᵢ)
        return 30 * euclidean(x, xᵢ)^3
        # return sum(5 * euclidean(x, xᵢ) * (3 * (x .- xᵢ) .^ 2 .+ sqeuclidean(x, xᵢ)))
    end
    return ∇²ℒ
end
function ∇²(::PHS5, normal::AbstractVector) #must check
    function ∇²ℒ(x, xᵢ, normal)
        r = euclidean(x, xᵢ)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        return -5 * (18) * dot_normal * r
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
function ∂(::PHS7, dim::Int, normal::AbstractVector)
    function ∂ℒ(x, xᵢ, normal)
        r = euclidean(x, xᵢ)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        Δ_d = x[dim] - xᵢ[dim]
        return -7 * (normal[dim] * r^5 + 5 * r^3 * Δ_d * dot_normal)
    end
    return ∂ℒ
end
function ∇(::PHS7)
    ∇ℒ(x, xᵢ) = 7 * (x .- xᵢ) * euclidean(x, xᵢ)^5
    return ∇ℒ
end
function ∇(::PHS7, normal::AbstractVector)
    function ∇ℒ(x, xᵢ, normal)
        r = euclidean(x, xᵢ)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        return -7 * (r^5 * normal .+ 5 * r^3 * dot_normal * (x .- xᵢ))
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
function ∂²(::PHS7, dim::Int)
    function ∂²ℒ(x, xᵢ)
        return 7 * euclidean(x, xᵢ)^3 * (5 * (x[dim] - xᵢ[dim])^2 + sqeuclidean(x, xᵢ))
    end
    return ∂²ℒ
end
function ∂²(::PHS7, dim::Int, normal::AbstractVector) #must check
    function ∂²ℒ(x, xᵢ, normal)
        n_d = normal[dim]
        Δ_d = x[dim] - xᵢ[dim]
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        r = euclidean(x, xᵢ)
        return -7 * (10 * n_d * dot_normal * r^3 + 5 * dot_normal(3 * r * Δ_d^2 + r^3))
    end
    return ∂²ℒ
end
function ∇²(::PHS7)
    function ∇²ℒ(x, xᵢ)
        return 56 * euclidean(x, xᵢ)^5
        # return sum(7 * euclidean(x, xᵢ)^3 * (5 * (x .- xᵢ) .^ 2 .+ sqeuclidean(x, xᵢ)))
    end
    return ∇²ℒ
end
function ∇²(::PHS7, normal::AbstractVector) #must check
    function ∇²ℒ(x, xᵢ, normal)
        r = euclidean(x, xᵢ)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        return -7 * (40 * dot_normal * r^3)
    end
    return ∇²ℒ
end
function ∇²(::PHS7, normal) #must check
    function ∇²ℒ(x, xᵢ, normal)
        r = euclidean(x, xᵢ)
        dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
        return -7 * (40 * dot_normal * r^3)
    end
    return ∇²ℒ
end

# I'm adding this because of the test.. any better idea?
function ∂_Hermite(phs::AbstractPHS, dim::Int)
    regular_op = ∂(phs, dim)  # Regular derivative operator

    function Hermite_deriv(x, xᵢ, normal=nothing)
        if normal === nothing || all(normal .== 0)
            # Regular derivative case
            return regular_op(x, xᵢ)
        else
            # Neumann derivative case
            neumann_op = ∂(phs, dim, normal)
            return neumann_op(x, xᵢ, normal)  # Must include normal here!
        end
    end

    return Hermite_deriv
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
