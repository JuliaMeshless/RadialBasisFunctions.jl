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
end#==============================================================================##==============================================================================#

#                                    PHS1                                      #

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

# ∂ - first partial derivative
function (op::∂{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return (x[op.dim] - xᵢ[op.dim]) / (r + AVOID_INF)
end

function (op::∂{<:PHS1})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -normal[op.dim] / (r + AVOID_INF) +
           dot_normal * (x[op.dim] - xᵢ[op.dim]) / (r^3 + AVOID_INF)
end

# ∇ - gradient
function (op::∇{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return (x .- xᵢ) / (r + AVOID_INF)
end

function (op::∇{<:PHS1})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -normal / (r + AVOID_INF) + dot_normal * (x .- xᵢ) / (r^3 + AVOID_INF)
end

# D - directional derivative
function (op::D{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return LinearAlgebra.dot(op.v, x .- xᵢ) / (r + AVOID_INF)
end

# D² - directional second derivative
function (op::D²{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    dot_v1_v2 = LinearAlgebra.dot(op.v1, op.v2)
    dot_v1_r = LinearAlgebra.dot(op.v1, x .- xᵢ)
    dot_v2_r = LinearAlgebra.dot(op.v2, x .- xᵢ)
    return -dot_v1_v2 / (r + AVOID_INF) + (dot_v1_r * dot_v2_r) / (r^3 + AVOID_INF)
end

# ∂² - second partial derivative
function (op::∂²{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    r² = sqeuclidean(x, xᵢ)
    return (-(x[op.dim] - xᵢ[op.dim])^2 + r²) / (r^3 + AVOID_INF)
end

function (op::∂²{<:PHS1})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    n_d = normal[op.dim]
    Δ_d = x[op.dim] - xᵢ[op.dim]
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return (2 * n_d * Δ_d + dot_normal) / (r^3 + AVOID_INF) -
           3 * Δ_d^2 * dot_normal / (r^5 + AVOID_INF)
end

# ∇² - Laplacian
function (op::∇²{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 2 / (r + AVOID_INF)
end

function (op::∇²{<:PHS1})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return 2 * dot_normal / (r^3 + AVOID_INF)
end#==============================================================================##==============================================================================#

#                                    PHS3                                      #

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

# ∂ - first partial derivative
function (op::∂{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 3 * (x[op.dim] - xᵢ[op.dim]) * r
end

function (op::∂{<:PHS3})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    n_d = normal[op.dim]
    Δ_d = x[op.dim] - xᵢ[op.dim]
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -3 * (n_d * r + dot_normal * Δ_d / (r + AVOID_INF))
end

# ∇ - gradient
function (op::∇{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 3 * (x .- xᵢ) * r
end

function (op::∇{<:PHS3})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -3 * (normal * r + dot_normal * (x .- xᵢ) / (r + AVOID_INF))
end

# D - directional derivative
function (op::D{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 3 * LinearAlgebra.dot(op.v, x .- xᵢ) * r
end

# D² - directional second derivative
function (op::D²{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    dot_v1_v2 = LinearAlgebra.dot(op.v1, op.v2)
    dot_v1_r = LinearAlgebra.dot(op.v1, x .- xᵢ)
    dot_v2_r = LinearAlgebra.dot(op.v2, x .- xᵢ)
    return -3 * (dot_v1_v2 * r + dot_v1_r * dot_v2_r / (r + AVOID_INF))
end

# ∂² - second partial derivative
function (op::∂²{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 3 * (r + (x[op.dim] - xᵢ[op.dim])^2 / (r + AVOID_INF))
end

function (op::∂²{<:PHS3})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    r² = sqeuclidean(x, xᵢ)
    n_d = normal[op.dim]
    Δ_d = x[op.dim] - xᵢ[op.dim]
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -3 * (2 * Δ_d * n_d + dot_normal - dot_normal * Δ_d^2 / (r² + AVOID_INF)) /
           (r + AVOID_INF)
end

# ∇² - Laplacian
function (op::∇²{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 12 * r
end

function (op::∇²{<:PHS3})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -12 * dot_normal / (r + AVOID_INF)
end#==============================================================================##==============================================================================#

#                                    PHS5                                      #

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

# ∂ - first partial derivative
function (op::∂{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 5 * (x[op.dim] - xᵢ[op.dim]) * r^3
end

function (op::∂{<:PHS5})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    n_d = normal[op.dim]
    Δ_d = x[op.dim] - xᵢ[op.dim]
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -5 * (n_d * r^3 + 3 * dot_normal * Δ_d * r)
end

# ∇ - gradient
function (op::∇{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 5 * (x .- xᵢ) * r^3
end

function (op::∇{<:PHS5})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -5 * (normal * r^3 + 3 * dot_normal * (x .- xᵢ) * r)
end

# D - directional derivative
function (op::D{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 5 * LinearAlgebra.dot(op.v, x .- xᵢ) * r^3
end

# D² - directional second derivative
function (op::D²{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    dot_v1_v2 = LinearAlgebra.dot(op.v1, op.v2)
    dot_v1_r = LinearAlgebra.dot(op.v1, x .- xᵢ)
    dot_v2_r = LinearAlgebra.dot(op.v2, x .- xᵢ)
    return -5 * (dot_v1_v2 * r^3 + 3 * dot_v1_r * dot_v2_r * r)
end

# ∂² - second partial derivative
function (op::∂²{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    r² = sqeuclidean(x, xᵢ)
    return 5 * r * (3 * (x[op.dim] - xᵢ[op.dim])^2 + r²)
end

function (op::∂²{<:PHS5})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    n_d = normal[op.dim]
    Δ_d = x[op.dim] - xᵢ[op.dim]
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -5 * (6 * n_d * Δ_d * r + 3 * dot_normal * (r + Δ_d^2 / (r + AVOID_INF)))
end

# ∇² - Laplacian
function (op::∇²{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 30 * r^3
end

function (op::∇²{<:PHS5})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -90 * dot_normal * r
end#==============================================================================##==============================================================================#

#                                    PHS7                                      #

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

# ∂ - first partial derivative
function (op::∂{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 7 * (x[op.dim] - xᵢ[op.dim]) * r^5
end

function (op::∂{<:PHS7})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    n_d = normal[op.dim]
    Δ_d = x[op.dim] - xᵢ[op.dim]
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -7 * (n_d * r^5 + 5 * r^3 * Δ_d * dot_normal)
end

# ∇ - gradient
function (op::∇{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 7 * (x .- xᵢ) * r^5
end

function (op::∇{<:PHS7})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -7 * (normal * r^5 + 5 * dot_normal * (x .- xᵢ) * r^3)
end

# D - directional derivative
function (op::D{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 7 * LinearAlgebra.dot(op.v, x .- xᵢ) * r^5
end

# D² - directional second derivative
function (op::D²{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    dot_v1_v2 = LinearAlgebra.dot(op.v1, op.v2)
    dot_v1_r = LinearAlgebra.dot(op.v1, x .- xᵢ)
    dot_v2_r = LinearAlgebra.dot(op.v2, x .- xᵢ)
    return -7 * (dot_v1_v2 * r^5 + 5 * dot_v1_r * dot_v2_r * r^3)
end

# ∂² - second partial derivative
function (op::∂²{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    r² = sqeuclidean(x, xᵢ)
    return 7 * r^3 * (5 * (x[op.dim] - xᵢ[op.dim])^2 + r²)
end

function (op::∂²{<:PHS7})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    n_d = normal[op.dim]
    Δ_d = x[op.dim] - xᵢ[op.dim]
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -7 * (10 * n_d * Δ_d * r^3 + 5 * dot_normal * (3 * r * Δ_d^2 + r^3))
end

# ∇² - Laplacian
function (op::∇²{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 56 * r^5
end

function (op::∇²{<:PHS7})(x, xᵢ, normal)
    r = euclidean(x, xᵢ)
    dot_normal = LinearAlgebra.dot(normal, x .- xᵢ)
    return -280 * dot_normal * r^3
end#==============================================================================##==============================================================================#

#                           Keyword Constructors                               #

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
