# Polyharmonic Spline

"""
   abstract type AbstractPHS <: AbstractRadialBasis

Supertype of all Polyharmonic Splines.
"""
abstract type AbstractPHS <: AbstractRadialBasis end

"""
    function PHS(n::T=3; poly_deg::T=2) where {T<:Int}

Convenience constructor for polyharmonic splines.

# Arguments
- `n`: Order of the spline (1, 3, 5, or 7). Higher = smoother.
- `poly_deg`: Polynomial augmentation degree (default: 2 for quadratic).

See also: [`IMQ`](@ref), [`Gaussian`](@ref)
"""
function PHS(n::T = 3; poly_deg::T = 2) where {T <: Int}
    check_poly_deg(poly_deg)
    if iseven(n) || n > 7
        throw(
            ArgumentError(
                "n must be 1, 3, 5, or 7 (got n=$n). Use PHS(3) for cubic or PHS(5) for quintic.",
            ),
        )
    end
    n == 1 && return PHS1(poly_deg)
    n == 3 && return PHS3(poly_deg)
    n == 5 && return PHS5(poly_deg)
    return PHS7(poly_deg)
end #==============================================================================# #==============================================================================#

#                                    PHS1                                      #

"""
    struct PHS1{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r``
"""
struct PHS1{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS1(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS1)(r2) = sqrt(r2)
(phs::PHS1)(x, xᵢ) = phs(sqeuclidean(x, xᵢ))

# ∂ - first partial derivative
function (op::∂{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return (x[op.dim] - xᵢ[op.dim]) / (r + avoid_inf(r))
end

# ∇ - gradient
function (op::∇{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return (x .- xᵢ) / (r + avoid_inf(r))
end

# H - Hessian matrix
function (op::H{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    Δ = x .- xᵢ
    N = length(x)
    T = eltype(x)
    # H[i,j] = δᵢⱼ/r - Δᵢ*Δⱼ/r³
    return StaticArraysCore.SMatrix{N, N, T}(
        ntuple(N * N) do k
            i, j = divrem(k - 1, N) .+ 1
            T(i == j) / (r + avoid_inf(r)) - Δ[i] * Δ[j] / (r^3 + avoid_inf(r))
        end,
    )
end

# ∂² - second partial derivative
function (op::∂²{<:PHS1})(x, xᵢ)
    r = euclidean(x, xᵢ)
    r² = sqeuclidean(x, xᵢ)
    return (-(x[op.dim] - xᵢ[op.dim])^2 + r²) / (r^3 + avoid_inf(r))
end

# ∇² - Laplacian: Δrᵏ = k(k+d−2)·r^(k−2) in d dimensions
function (op::∇²{<:PHS1})(x, xᵢ)
    d = length(x)
    r = euclidean(x, xᵢ)
    return (d - 1) / (r + avoid_inf(r))
end

#==============================================================================#

#                                    PHS3                                      #

"""
    struct PHS3{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r^3``
"""
struct PHS3{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS3(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS3)(r2) = r2 * sqrt(r2)
(phs::PHS3)(x, xᵢ) = phs(sqeuclidean(x, xᵢ))

# ∂ - first partial derivative
function (op::∂{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 3 * (x[op.dim] - xᵢ[op.dim]) * r
end

# ∇ - gradient
function (op::∇{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 3 * (x .- xᵢ) * r
end

# H - Hessian matrix
function (op::H{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    Δ = x .- xᵢ
    N = length(x)
    T = eltype(x)
    # H[i,j] = 3 * (δᵢⱼ * r + Δᵢ*Δⱼ/r)
    return StaticArraysCore.SMatrix{N, N, T}(
        ntuple(N * N) do k
            i, j = divrem(k - 1, N) .+ 1
            3 * (T(i == j) * r + Δ[i] * Δ[j] / (r + avoid_inf(r)))
        end,
    )
end

# ∂² - second partial derivative
function (op::∂²{<:PHS3})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 3 * (r + (x[op.dim] - xᵢ[op.dim])^2 / (r + avoid_inf(r)))
end

# ∇² - Laplacian: Δrᵏ = k(k+d−2)·r^(k−2) in d dimensions
function (op::∇²{<:PHS3})(x, xᵢ)
    d = length(x)
    r = euclidean(x, xᵢ)
    return 3 * (d + 1) * r
end

#==============================================================================#

#                                    PHS5                                      #

"""
    struct PHS5{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r^5``
"""
struct PHS5{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS5(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end
(phs::PHS5)(r2) = r2^2 * sqrt(r2)
(phs::PHS5)(x, xᵢ) = phs(sqeuclidean(x, xᵢ))

# ∂ - first partial derivative
function (op::∂{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 5 * (x[op.dim] - xᵢ[op.dim]) * r^3
end

# ∇ - gradient
function (op::∇{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 5 * (x .- xᵢ) * r^3
end

# H - Hessian matrix
function (op::H{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    Δ = x .- xᵢ
    N = length(x)
    T = eltype(x)
    # H[i,j] = 5 * (δᵢⱼ * r³ + 3 * Δᵢ*Δⱼ * r)
    return StaticArraysCore.SMatrix{N, N, T}(
        ntuple(N * N) do k
            i, j = divrem(k - 1, N) .+ 1
            5 * (T(i == j) * r^3 + 3 * Δ[i] * Δ[j] * r)
        end,
    )
end

# ∂² - second partial derivative
function (op::∂²{<:PHS5})(x, xᵢ)
    r = euclidean(x, xᵢ)
    r² = sqeuclidean(x, xᵢ)
    return 5 * r * (3 * (x[op.dim] - xᵢ[op.dim])^2 + r²)
end

# ∇² - Laplacian: Δrᵏ = k(k+d−2)·r^(k−2) in d dimensions
function (op::∇²{<:PHS5})(x, xᵢ)
    d = length(x)
    r = euclidean(x, xᵢ)
    return 5 * (d + 3) * r^3
end

#==============================================================================#

#                                    PHS7                                      #

"""
    struct PHS7{T<:Int} <: AbstractPHS

Polyharmonic spline radial basis function:``ϕ(r) = r^7``
"""
struct PHS7{T <: Int} <: AbstractPHS
    poly_deg::T
    function PHS7(poly_deg::T) where {T <: Int}
        check_poly_deg(poly_deg)
        return new{T}(poly_deg)
    end
end

(phs::PHS7)(r2) = r2^3 * sqrt(r2)
(phs::PHS7)(x, xᵢ) = phs(sqeuclidean(x, xᵢ))

# ∂ - first partial derivative
function (op::∂{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 7 * (x[op.dim] - xᵢ[op.dim]) * r^5
end

# ∇ - gradient
function (op::∇{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    return 7 * (x .- xᵢ) * r^5
end

# H - Hessian matrix
function (op::H{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    Δ = x .- xᵢ
    N = length(x)
    T = eltype(x)
    # H[i,j] = 7 * (δᵢⱼ * r⁵ + 5 * Δᵢ*Δⱼ * r³)
    return StaticArraysCore.SMatrix{N, N, T}(
        ntuple(N * N) do k
            i, j = divrem(k - 1, N) .+ 1
            7 * (T(i == j) * r^5 + 5 * Δ[i] * Δ[j] * r^3)
        end,
    )
end

# ∂² - second partial derivative
function (op::∂²{<:PHS7})(x, xᵢ)
    r = euclidean(x, xᵢ)
    r² = sqeuclidean(x, xᵢ)
    return 7 * r^3 * (5 * (x[op.dim] - xᵢ[op.dim])^2 + r²)
end

# ∇² - Laplacian: Δrᵏ = k(k+d−2)·r^(k−2) in d dimensions
function (op::∇²{<:PHS7})(x, xᵢ)
    d = length(x)
    r = euclidean(x, xᵢ)
    return 7 * (d + 5) * r^5
end

#==============================================================================#

#                           Keyword Constructors                               #

# convient constructors using keyword arguments
for phs in (:PHS1, :PHS3, :PHS5, :PHS7)
    @eval function $phs(; poly_deg::Int = 2)
        return $phs(poly_deg)
    end
end

function Base.show(io::IO, rbf::R) where {R <: AbstractPHS}
    print(io, print_basis(rbf))
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end

print_basis(::PHS1) = "Polyharmonic spline (r¹)"
print_basis(::PHS3) = "Polyharmonic spline (r³)"
print_basis(::PHS5) = "Polyharmonic spline (r⁵)"
print_basis(::PHS7) = "Polyharmonic spline (r⁷)"
