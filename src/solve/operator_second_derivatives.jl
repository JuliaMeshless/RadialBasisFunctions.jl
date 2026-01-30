#=
Second derivatives of applied operators for RHS backward pass.

For the backward pass through _build_weights, we need:
  ∂/∂x [ℒφ(x, xi)] and ∂/∂xi [ℒφ(x, xi)]

where ℒφ is the operator applied to the basis function (e.g., ∂φ/∂dim for Partial).

These are effectively Hessian-like terms of the basis function.
=#

using Distances: euclidean

# ============================================================================
# PHS3: φ(r) = r³
# First derivative: ∂φ/∂x[dim] = 3 * (x[dim] - xi[dim]) * r
# ============================================================================

"""
    grad_partial_phs3_wrt_x(dim)

Returns a function computing ∂/∂x[j] of [∂φ/∂x[dim]] for PHS3.

Mathematical derivation:
  ∂φ/∂x[dim] = 3 * δ_d * r  where δ_d = x[dim] - xi[dim], r = ||x - xi||

  ∂²φ/∂x[j]∂x[dim] = 3 * (δ_{j,dim} * r + δ_d * δ_j / r)

  For j == dim: 3 * (r + δ_d² / r)
  For j != dim: 3 * δ_d * δ_j / r
"""
function grad_partial_phs3_wrt_x(dim::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r_safe = r + AVOID_INF
        δ = x .- xi
        δ_d = δ[dim]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 3 * (r + δ_d^2 / r_safe)
            else
                result[j] = 3 * δ_d * δ[j] / r_safe
            end
        end
        return result
    end
    return grad_Lφ_x
end

"""
    grad_partial_phs3_wrt_xi(dim)

Returns a function computing ∂/∂xi[j] of [∂φ/∂x[dim]] for PHS3.

By symmetry: ∂/∂xi = -∂/∂x for terms depending on (x - xi).
"""
function grad_partial_phs3_wrt_xi(dim::Int)
    grad_x = grad_partial_phs3_wrt_x(dim)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# PHS3: Laplacian ∇²φ = 12r
# ============================================================================

"""
    grad_laplacian_phs3_wrt_x()

Returns a function computing ∂/∂x[j] of [∇²φ] for PHS3.

Mathematical derivation:
  ∇²φ = 12r
  ∂/∂x[j] [12r] = 12 * δ_j / r
"""
function grad_laplacian_phs3_wrt_x()
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r_safe = r + AVOID_INF
        δ = x .- xi
        return 12 .* δ ./ r_safe
    end
    return grad_Lφ_x
end

"""
    grad_laplacian_phs3_wrt_xi()

Returns a function computing ∂/∂xi[j] of [∇²φ] for PHS3.
"""
function grad_laplacian_phs3_wrt_xi()
    grad_x = grad_laplacian_phs3_wrt_x()
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# PHS1: φ(r) = r
# First derivative: ∂φ/∂x[dim] = (x[dim] - xi[dim]) / r
# ============================================================================

"""
    grad_partial_phs1_wrt_x(dim)

Returns a function computing ∂/∂x[j] of [∂φ/∂x[dim]] for PHS1.

Mathematical derivation:
  ∂φ/∂x[dim] = δ_d / r

  ∂²φ/∂x[j]∂x[dim] = (δ_{j,dim} * r - δ_d * δ_j / r) / r²
                   = δ_{j,dim} / r - δ_d * δ_j / r³

Note: At r=0, the derivative is singular but we return 0 since the RBF value
itself is 0 at r=0, so this term doesn't contribute to the gradient.
"""
function grad_partial_phs1_wrt_x(dim::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        # At r=0, PHS1 and its derivatives are 0, so gradient contribution is 0
        if r < 1.0e-12
            return zero(x)
        end
        δ = x .- xi
        δ_d = δ[dim]
        r3 = r^3

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 1 / r - δ_d^2 / r3
            else
                result[j] = -δ_d * δ[j] / r3
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_partial_phs1_wrt_xi(dim::Int)
    grad_x = grad_partial_phs1_wrt_x(dim)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

"""
    grad_laplacian_phs1_wrt_x()

Returns a function computing ∂/∂x[j] of [∇²φ] for PHS1.

Mathematical derivation:
  ∇²φ = 2/r
  ∂/∂x[j] [2/r] = -2 * δ_j / r³

Note: At r=0, we return 0 to avoid singularity.
"""
function grad_laplacian_phs1_wrt_x()
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        # At r=0, PHS1 Laplacian is singular, return 0 for gradient
        if r < 1.0e-12
            return zero(x)
        end
        r3 = r^3
        δ = x .- xi
        return -2 .* δ ./ r3
    end
    return grad_Lφ_x
end

function grad_laplacian_phs1_wrt_xi()
    grad_x = grad_laplacian_phs1_wrt_x()
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# PHS5: φ(r) = r⁵
# First derivative: ∂φ/∂x[dim] = 5 * (x[dim] - xi[dim]) * r³
# ============================================================================

"""
    grad_partial_phs5_wrt_x(dim)

Returns a function computing ∂/∂x[j] of [∂φ/∂x[dim]] for PHS5.

Mathematical derivation:
  ∂φ/∂x[dim] = 5 * δ_d * r³

  ∂²φ/∂x[j]∂x[dim] = 5 * (δ_{j,dim} * r³ + δ_d * 3r * δ_j)
                   = 5 * (δ_{j,dim} * r³ + 3 * δ_d * δ_j * r)
"""
function grad_partial_phs5_wrt_x(dim::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r3 = r^3
        δ = x .- xi
        δ_d = δ[dim]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 5 * (r3 + 3 * δ_d^2 * r)
            else
                result[j] = 5 * 3 * δ_d * δ[j] * r
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_partial_phs5_wrt_xi(dim::Int)
    grad_x = grad_partial_phs5_wrt_x(dim)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

"""
    grad_laplacian_phs5_wrt_x()

Returns a function computing ∂/∂x[j] of [∇²φ] for PHS5.

Mathematical derivation:
  ∇²φ = 30r³
  ∂/∂x[j] [30r³] = 30 * 3r² * δ_j / r = 90 * r * δ_j
"""
function grad_laplacian_phs5_wrt_x()
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        δ = x .- xi
        return 90 .* r .* δ
    end
    return grad_Lφ_x
end

function grad_laplacian_phs5_wrt_xi()
    grad_x = grad_laplacian_phs5_wrt_x()
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# PHS7: φ(r) = r⁷
# First derivative: ∂φ/∂x[dim] = 7 * (x[dim] - xi[dim]) * r⁵
# ============================================================================

"""
    grad_partial_phs7_wrt_x(dim)

Returns a function computing ∂/∂x[j] of [∂φ/∂x[dim]] for PHS7.

Mathematical derivation:
  ∂φ/∂x[dim] = 7 * δ_d * r⁵

  ∂²φ/∂x[j]∂x[dim] = 7 * (δ_{j,dim} * r⁵ + δ_d * 5r³ * δ_j)
"""
function grad_partial_phs7_wrt_x(dim::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r5 = r^5
        r3 = r^3
        δ = x .- xi
        δ_d = δ[dim]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 7 * (r5 + 5 * δ_d^2 * r3)
            else
                result[j] = 7 * 5 * δ_d * δ[j] * r3
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_partial_phs7_wrt_xi(dim::Int)
    grad_x = grad_partial_phs7_wrt_x(dim)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

"""
    grad_laplacian_phs7_wrt_x()

Returns a function computing ∂/∂x[j] of [∇²φ] for PHS7.

Mathematical derivation:
  ∇²φ = 56r⁵
  ∂/∂x[j] [56r⁵] = 56 * 5r⁴ * δ_j / r = 280 * r³ * δ_j
"""
function grad_laplacian_phs7_wrt_x()
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r3 = r^3
        δ = x .- xi
        return 280 .* r3 .* δ
    end
    return grad_Lφ_x
end

function grad_laplacian_phs7_wrt_xi()
    grad_x = grad_laplacian_phs7_wrt_x()
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# Dispatch functions to get correct second derivative based on operator/basis
# ============================================================================

"""
    grad_applied_partial_wrt_x(basis, dim)

Get gradient of applied partial derivative operator w.r.t. evaluation point.
"""
function grad_applied_partial_wrt_x(::PHS1, dim::Int)
    return grad_partial_phs1_wrt_x(dim)
end

function grad_applied_partial_wrt_x(::PHS3, dim::Int)
    return grad_partial_phs3_wrt_x(dim)
end

function grad_applied_partial_wrt_x(::PHS5, dim::Int)
    return grad_partial_phs5_wrt_x(dim)
end

function grad_applied_partial_wrt_x(::PHS7, dim::Int)
    return grad_partial_phs7_wrt_x(dim)
end

function grad_applied_partial_wrt_xi(::PHS1, dim::Int)
    return grad_partial_phs1_wrt_xi(dim)
end

function grad_applied_partial_wrt_xi(::PHS3, dim::Int)
    return grad_partial_phs3_wrt_xi(dim)
end

function grad_applied_partial_wrt_xi(::PHS5, dim::Int)
    return grad_partial_phs5_wrt_xi(dim)
end

function grad_applied_partial_wrt_xi(::PHS7, dim::Int)
    return grad_partial_phs7_wrt_xi(dim)
end

"""
    grad_applied_laplacian_wrt_x(basis)

Get gradient of applied Laplacian operator w.r.t. evaluation point.
"""
function grad_applied_laplacian_wrt_x(::PHS1)
    return grad_laplacian_phs1_wrt_x()
end

function grad_applied_laplacian_wrt_x(::PHS3)
    return grad_laplacian_phs3_wrt_x()
end

function grad_applied_laplacian_wrt_x(::PHS5)
    return grad_laplacian_phs5_wrt_x()
end

function grad_applied_laplacian_wrt_x(::PHS7)
    return grad_laplacian_phs7_wrt_x()
end

function grad_applied_laplacian_wrt_xi(::PHS1)
    return grad_laplacian_phs1_wrt_xi()
end

function grad_applied_laplacian_wrt_xi(::PHS3)
    return grad_laplacian_phs3_wrt_xi()
end

function grad_applied_laplacian_wrt_xi(::PHS5)
    return grad_laplacian_phs5_wrt_xi()
end

function grad_applied_laplacian_wrt_xi(::PHS7)
    return grad_laplacian_phs7_wrt_xi()
end

# ============================================================================
# IMQ: φ(r) = 1/√(1 + (εr)²)
# Let s = ε²r² + 1, then φ = 1/√s = s^(-1/2)
# First derivative: ∂φ/∂x[dim] = -ε² * δ_d / s^(3/2)
# ============================================================================

"""
    grad_partial_imq_wrt_x(ε, dim)

Returns a function computing ∂/∂x[j] of [∂φ/∂x[dim]] for IMQ.

Mathematical derivation:
  Let s = ε²r² + 1, δ_d = x[dim] - xi[dim]
  ∂φ/∂x[dim] = -ε² * δ_d / s^(3/2)

  ∂²φ/∂x[j]∂x[dim] = -ε² * [δ_{j,dim} / s^(3/2) - δ_d * (3/2) * s^(-5/2) * 2ε² * δ_j]
                   = -ε² * δ_{j,dim} / s^(3/2) + 3ε⁴ * δ_d * δ_j / s^(5/2)

  For j == dim: -ε² / s^(3/2) + 3ε⁴ * δ_d² / s^(5/2)
  For j != dim: 3ε⁴ * δ_d * δ_j / s^(5/2)
"""
function grad_partial_imq_wrt_x(ε::T, dim::Int) where {T}
    ε2 = ε^2
    ε4 = ε^4
    function grad_Lφ_x(x, xi)
        r2 = sqeuclidean(x, xi)
        s = ε2 * r2 + 1
        s32 = sqrt(s^3)
        s52 = sqrt(s^5)
        δ = x .- xi
        δ_d = δ[dim]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = -ε2 / s32 + 3 * ε4 * δ_d^2 / s52
            else
                result[j] = 3 * ε4 * δ_d * δ[j] / s52
            end
        end
        return result
    end
    return grad_Lφ_x
end

"""
    grad_partial_imq_wrt_xi(ε, dim)

Returns a function computing ∂/∂xi[j] of [∂φ/∂x[dim]] for IMQ.

By symmetry: ∂/∂xi = -∂/∂x for terms depending on (x - xi).
"""
function grad_partial_imq_wrt_xi(ε::T, dim::Int) where {T}
    grad_x = grad_partial_imq_wrt_x(ε, dim)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# IMQ: Laplacian ∇²φ = 3ε⁴r²/s^(5/2) - D*ε²/s^(3/2)
# where s = ε²r² + 1, D = dimension
# ============================================================================

"""
    grad_laplacian_imq_wrt_x(ε)

Returns a function computing ∂/∂x[j] of [∇²φ] for IMQ.

Mathematical derivation:
  ∇²φ = sum_i [∂²φ/∂x[i]²] = 3ε⁴r²/s^(5/2) - D*ε²/s^(3/2)

  Let u = 3ε⁴r²/s^(5/2) and v = D*ε²/s^(3/2)

  ∂u/∂x[j] = 3ε⁴ * [2δ_j / s^(5/2) - r² * (5/2) * s^(-7/2) * 2ε² * δ_j]
           = 3ε⁴ * δ_j * [2/s^(5/2) - 5ε²r²/s^(7/2)]
           = δ_j * [6ε⁴/s^(5/2) - 15ε⁶r²/s^(7/2)]

  ∂v/∂x[j] = D*ε² * (-(3/2) * s^(-5/2) * 2ε² * δ_j)
           = -3D*ε⁴ * δ_j / s^(5/2)

  ∂(∇²φ)/∂x[j] = ∂u/∂x[j] - ∂v/∂x[j]
               = δ_j * [6ε⁴/s^(5/2) - 15ε⁶r²/s^(7/2) + 3D*ε⁴/s^(5/2)]
               = δ_j * [(6 + 3D)ε⁴/s^(5/2) - 15ε⁶r²/s^(7/2)]
               = δ_j * [3(D+2)ε⁴/s^(5/2) - 15ε⁶r²/s^(7/2)]
"""
function grad_laplacian_imq_wrt_x(ε::T) where {T}
    ε2 = ε^2
    ε4 = ε^4
    ε6 = ε^6
    function grad_Lφ_x(x, xi)
        D = length(x)
        r2 = sqeuclidean(x, xi)
        s = ε2 * r2 + 1
        s52 = sqrt(s^5)
        s72 = sqrt(s^7)
        δ = x .- xi
        coeff = 3 * (D + 2) * ε4 / s52 - 15 * ε6 * r2 / s72
        return coeff .* δ
    end
    return grad_Lφ_x
end

"""
    grad_laplacian_imq_wrt_xi(ε)

Returns a function computing ∂/∂xi[j] of [∇²φ] for IMQ.
"""
function grad_laplacian_imq_wrt_xi(ε::T) where {T}
    grad_x = grad_laplacian_imq_wrt_x(ε)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# Gaussian: φ(r) = exp(-(εr)²)
# First derivative: ∂φ/∂x[dim] = -2ε² * δ_d * φ
# ============================================================================

"""
    grad_partial_gaussian_wrt_x(ε, dim)

Returns a function computing ∂/∂x[j] of [∂φ/∂x[dim]] for Gaussian.

Mathematical derivation:
  φ = exp(-ε²r²)
  ∂φ/∂x[dim] = -2ε² * δ_d * φ

  ∂²φ/∂x[j]∂x[dim] = -2ε² * [δ_{j,dim} * φ + δ_d * ∂φ/∂x[j]]
                   = -2ε² * [δ_{j,dim} * φ + δ_d * (-2ε² * δ_j * φ)]
                   = -2ε² * φ * [δ_{j,dim} - 2ε² * δ_d * δ_j]
                   = φ * [-2ε² * δ_{j,dim} + 4ε⁴ * δ_d * δ_j]

  For j == dim: φ * (4ε⁴ * δ_d² - 2ε²)
  For j != dim: φ * 4ε⁴ * δ_d * δ_j
"""
function grad_partial_gaussian_wrt_x(ε::T, dim::Int) where {T}
    ε2 = ε^2
    ε4 = ε^4
    function grad_Lφ_x(x, xi)
        r2 = sqeuclidean(x, xi)
        φ = exp(-ε2 * r2)
        δ = x .- xi
        δ_d = δ[dim]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = φ * (4 * ε4 * δ_d^2 - 2 * ε2)
            else
                result[j] = φ * 4 * ε4 * δ_d * δ[j]
            end
        end
        return result
    end
    return grad_Lφ_x
end

"""
    grad_partial_gaussian_wrt_xi(ε, dim)

Returns a function computing ∂/∂xi[j] of [∂φ/∂x[dim]] for Gaussian.

By symmetry: ∂/∂xi = -∂/∂x for terms depending on (x - xi).
"""
function grad_partial_gaussian_wrt_xi(ε::T, dim::Int) where {T}
    grad_x = grad_partial_gaussian_wrt_x(ε, dim)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# Gaussian: Laplacian ∇²φ = (4ε⁴r² - 2ε²D) * φ
# where D = dimension
# ============================================================================

"""
    grad_laplacian_gaussian_wrt_x(ε)

Returns a function computing ∂/∂x[j] of [∇²φ] for Gaussian.

Mathematical derivation:
  ∇²φ = (4ε⁴r² - 2ε²D) * φ

  Let u = 4ε⁴r² - 2ε²D (coefficient)
  ∂u/∂x[j] = 4ε⁴ * 2δ_j = 8ε⁴ * δ_j

  ∂(∇²φ)/∂x[j] = ∂u/∂x[j] * φ + u * ∂φ/∂x[j]
               = 8ε⁴ * δ_j * φ + (4ε⁴r² - 2ε²D) * (-2ε² * δ_j * φ)
               = φ * δ_j * [8ε⁴ - 2ε² * (4ε⁴r² - 2ε²D)]
               = φ * δ_j * [8ε⁴ - 8ε⁶r² + 4ε⁴D]
               = φ * δ_j * 4ε⁴ * [2 + D - 2ε²r²]
"""
function grad_laplacian_gaussian_wrt_x(ε::T) where {T}
    ε2 = ε^2
    ε4 = ε^4
    function grad_Lφ_x(x, xi)
        D = length(x)
        r2 = sqeuclidean(x, xi)
        φ = exp(-ε2 * r2)
        δ = x .- xi
        coeff = 4 * ε4 * (2 + D - 2 * ε2 * r2)
        return φ * coeff .* δ
    end
    return grad_Lφ_x
end

"""
    grad_laplacian_gaussian_wrt_xi(ε)

Returns a function computing ∂/∂xi[j] of [∇²φ] for Gaussian.
"""
function grad_laplacian_gaussian_wrt_xi(ε::T) where {T}
    grad_x = grad_laplacian_gaussian_wrt_x(ε)
    function grad_Lφ_xi(x, xi)
        return -grad_x(x, xi)
    end
    return grad_Lφ_xi
end

# ============================================================================
# Dispatch functions for IMQ and Gaussian
# ============================================================================

function grad_applied_partial_wrt_x(basis::IMQ, dim::Int)
    return grad_partial_imq_wrt_x(basis.ε, dim)
end

function grad_applied_partial_wrt_xi(basis::IMQ, dim::Int)
    return grad_partial_imq_wrt_xi(basis.ε, dim)
end

function grad_applied_laplacian_wrt_x(basis::IMQ)
    return grad_laplacian_imq_wrt_x(basis.ε)
end

function grad_applied_laplacian_wrt_xi(basis::IMQ)
    return grad_laplacian_imq_wrt_xi(basis.ε)
end

function grad_applied_partial_wrt_x(basis::Gaussian, dim::Int)
    return grad_partial_gaussian_wrt_x(basis.ε, dim)
end

function grad_applied_partial_wrt_xi(basis::Gaussian, dim::Int)
    return grad_partial_gaussian_wrt_xi(basis.ε, dim)
end

function grad_applied_laplacian_wrt_x(basis::Gaussian)
    return grad_laplacian_gaussian_wrt_x(basis.ε)
end

function grad_applied_laplacian_wrt_xi(basis::Gaussian)
    return grad_laplacian_gaussian_wrt_xi(basis.ε)
end
