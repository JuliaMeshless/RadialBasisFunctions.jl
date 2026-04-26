#=
Second derivatives of applied operators for RHS backward pass.

For the backward pass through _build_weights, we need:
  ∂/∂x [ℒφ(x, xi)] and ∂/∂xi [ℒφ(x, xi)]

where ℒφ is the operator applied to the basis function (e.g., ∂φ/∂dim for Partial).

These are effectively Hessian-like terms of the basis function.
=#

"""
    negate_grad(grad_fn)

Given a gradient function `grad_fn(x, xi)`, returns `(x, xi) -> -grad_fn(x, xi)`.
All `_wrt_xi` functions are the negation of their `_wrt_x` counterparts by symmetry.
"""
negate_grad(grad_fn) = (x, xi) -> -grad_fn(x, xi)

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

# ============================================================================
# IMQ: Laplacian ∇²φ = 3ε⁴r²/s^(5/2) - D*ε²/s^(3/2)
# where s = ε²r² + 1, D = dimension
# ============================================================================

"""
    grad_laplacian_imq_wrt_x(ε)

Returns a function computing ∂/∂x[j] of [∇²φ] for IMQ.

Mathematical derivation:
  ∇²φ = sum_i [∂²φ/∂x[i]²] = 3ε⁴r²/s^(5/2) - D*ε²/s^(3/2)

  ∂(∇²φ)/∂x[j] = δ_j * [3(D+2)ε⁴/s^(5/2) - 15ε⁶r²/s^(7/2)]
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

  ∂²φ/∂x[j]∂x[dim] = φ * [-2ε² * δ_{j,dim} + 4ε⁴ * δ_d * δ_j]

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

# ============================================================================
# Gaussian: Laplacian ∇²φ = (4ε⁴r² - 2ε²D) * φ
# where D = dimension
# ============================================================================

"""
    grad_laplacian_gaussian_wrt_x(ε)

Returns a function computing ∂/∂x[j] of [∇²φ] for Gaussian.

Mathematical derivation:
  ∇²φ = (4ε⁴r² - 2ε²D) * φ

  ∂(∇²φ)/∂x[j] = φ * δ_j * 4ε⁴ * [2 + D - 2ε²r²]
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

# ============================================================================
# MixedPartial: ∂²φ/(∂x[d1]∂x[d2]) with d1 != d2
# Gradients needed for MixedPartial _build_weights backward pass
# ============================================================================

function grad_mixed_partial_phs1_wrt_x(dim1::Int, dim2::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        if r < 1.0e-12
            return zero(x)
        end
        δ = x .- xi
        δ1 = δ[dim1]
        δ2 = δ[dim2]
        r3 = r^3
        r5 = r^5

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim1
                result[j] = -δ2 / r3 + 3 * δ1^2 * δ2 / r5
            elseif j == dim2
                result[j] = -δ1 / r3 + 3 * δ1 * δ2^2 / r5
            else
                result[j] = 3 * δ1 * δ2 * δ[j] / r5
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_mixed_partial_phs3_wrt_x(dim1::Int, dim2::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r_safe = r + AVOID_INF
        r3_safe = r_safe^3
        δ = x .- xi
        δ1 = δ[dim1]
        δ2 = δ[dim2]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim1
                result[j] = 3 * δ2 * (1 / r_safe - δ1^2 / r3_safe)
            elseif j == dim2
                result[j] = 3 * δ1 * (1 / r_safe - δ2^2 / r3_safe)
            else
                result[j] = -3 * δ1 * δ2 * δ[j] / r3_safe
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_mixed_partial_phs5_wrt_x(dim1::Int, dim2::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r_safe = r + AVOID_INF
        δ = x .- xi
        δ1 = δ[dim1]
        δ2 = δ[dim2]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim1
                result[j] = 15 * δ2 * (r + δ1^2 / r_safe)
            elseif j == dim2
                result[j] = 15 * δ1 * (r + δ2^2 / r_safe)
            else
                result[j] = 15 * δ1 * δ2 * δ[j] / r_safe
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_mixed_partial_phs7_wrt_x(dim1::Int, dim2::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r2 = r^2
        δ = x .- xi
        δ1 = δ[dim1]
        δ2 = δ[dim2]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim1
                result[j] = 35 * δ2 * r * (r2 + 3 * δ1^2)
            elseif j == dim2
                result[j] = 35 * δ1 * r * (r2 + 3 * δ2^2)
            else
                result[j] = 105 * δ1 * δ2 * δ[j] * r
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_mixed_partial_imq_wrt_x(ε::T, dim1::Int, dim2::Int) where {T}
    ε2 = ε^2
    ε4 = ε^4
    ε6 = ε^6
    function grad_Lφ_x(x, xi)
        r2 = sqeuclidean(x, xi)
        s = ε2 * r2 + 1
        s52 = sqrt(s^5)
        s72 = sqrt(s^7)
        δ = x .- xi
        δ1 = δ[dim1]
        δ2 = δ[dim2]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim1
                result[j] = 3 * ε4 * δ2 * (1 / s52 - 5 * ε2 * δ1^2 / s72)
            elseif j == dim2
                result[j] = 3 * ε4 * δ1 * (1 / s52 - 5 * ε2 * δ2^2 / s72)
            else
                result[j] = -15 * ε6 * δ1 * δ2 * δ[j] / s72
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_mixed_partial_gaussian_wrt_x(ε::T, dim1::Int, dim2::Int) where {T}
    ε2 = ε^2
    ε4 = ε^4
    ε6 = ε^6
    function grad_Lφ_x(x, xi)
        r2 = sqeuclidean(x, xi)
        φ = exp(-ε2 * r2)
        δ = x .- xi
        δ1 = δ[dim1]
        δ2 = δ[dim2]

        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim1
                result[j] = 4 * ε4 * φ * δ2 * (1 - 2 * ε2 * δ1^2)
            elseif j == dim2
                result[j] = 4 * ε4 * φ * δ1 * (1 - 2 * ε2 * δ2^2)
            else
                result[j] = -8 * ε6 * φ * δ1 * δ2 * δ[j]
            end
        end
        return result
    end
    return grad_Lφ_x
end

# ============================================================================
# Dispatch functions: map basis type to correct gradient functions
# ============================================================================

# ============================================================================
# Second-order partial derivative: ∂/∂x[j] of [∂²φ/∂x[dim]²]
# These are third-order derivatives needed for Partial(2, dim) backward pass.
# Symmetry: grad_wrt_xi = -grad_wrt_x (same as first-order case, see derivation below)
#
# For φ = φ(||x-xi||), let δ = x-xi. Since ∂δ[d]/∂xi[j] = -δ_{dj},
# ∂/∂xi[j] [∂²φ/∂x[dim]²] = -∂/∂x[j] [∂²φ/∂x[dim]²].
# ============================================================================

function grad_second_partial_phs1_wrt_x(dim::Int)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r < 1.0e-12 && return zero(x)
        r3 = r^3
        r5 = r^5
        δ = x .- xi
        δ_d = δ[dim]
        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = -3 * δ_d / r3 + 3 * δ_d^3 / r5
            else
                result[j] = -δ[j] / r3 + 3 * δ_d^2 * δ[j] / r5
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_second_partial_phs3_wrt_x(dim::Int)
    # ∂²φ/∂x[dim]² = 3(r + δ_d²/r)
    # j==dim: 3(3δ_d/r - δ_d³/r³)   j≠dim: 3δ_j(1/r - δ_d²/r³)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r_safe = r + AVOID_INF
        δ = x .- xi
        δ_d = δ[dim]
        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 3 * (3 * δ_d / r_safe - δ_d^3 / r_safe^3)
            else
                result[j] = 3 * δ[j] * (1 / r_safe - δ_d^2 / r_safe^3)
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_second_partial_phs5_wrt_x(dim::Int)
    # ∂²φ/∂x[dim]² = 5r(r² + 3δ_d²)
    # j==dim: 15δ_d(3r + δ_d²/r)   j≠dim: 15δ_j(r + δ_d²/r)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r_safe = r + AVOID_INF
        δ = x .- xi
        δ_d = δ[dim]
        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 15 * δ_d * (3 * r_safe + δ_d^2 / r_safe)
            else
                result[j] = 15 * δ[j] * (r_safe + δ_d^2 / r_safe)
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_second_partial_phs7_wrt_x(dim::Int)
    # ∂²φ/∂x[dim]² = 7r³(r² + 5δ_d²)
    # j==dim: 105rδ_d(r² + δ_d²)   j≠dim: 35rδ_j(r² + 3δ_d²)
    function grad_Lφ_x(x, xi)
        r = euclidean(x, xi)
        r2 = r^2
        δ = x .- xi
        δ_d = δ[dim]
        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 105 * r * δ_d * (r2 + δ_d^2)
            else
                result[j] = 35 * r * δ[j] * (r2 + 3 * δ_d^2)
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_second_partial_imq_wrt_x(ε::T, dim::Int) where {T}
    ε2 = ε^2
    ε4 = ε^4
    ε6 = ε^6
    # ∂²φ/∂x[dim]² = -ε²/s^(3/2) + 3ε⁴δ_d²/s^(5/2),  s = ε²r²+1
    # j==dim: 9ε⁴δ_d/s^(5/2) - 15ε⁶δ_d³/s^(7/2)
    # j≠dim:  3ε⁴δ_j/s^(5/2) - 15ε⁶δ_d²δ_j/s^(7/2)
    function grad_Lφ_x(x, xi)
        r2 = sqeuclidean(x, xi)
        s = ε2 * r2 + 1
        s52 = sqrt(s^5)
        s72 = sqrt(s^7)
        δ = x .- xi
        δ_d = δ[dim]
        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = 9 * ε4 * δ_d / s52 - 15 * ε6 * δ_d^3 / s72
            else
                result[j] = 3 * ε4 * δ[j] / s52 - 15 * ε6 * δ_d^2 * δ[j] / s72
            end
        end
        return result
    end
    return grad_Lφ_x
end

function grad_second_partial_gaussian_wrt_x(ε::T, dim::Int) where {T}
    ε2 = ε^2
    ε4 = ε^4
    ε6 = ε^6
    # ∂²φ/∂x[dim]² = φ(-2ε² + 4ε⁴δ_d²)
    # j==dim: φδ_d(12ε⁴ - 8ε⁶δ_d²)   j≠dim: φδ_j(4ε⁴ - 8ε⁶δ_d²)
    function grad_Lφ_x(x, xi)
        r2 = sqeuclidean(x, xi)
        φ = exp(-ε2 * r2)
        δ = x .- xi
        δ_d = δ[dim]
        result = similar(x, eltype(x))
        @inbounds for j in eachindex(x)
            if j == dim
                result[j] = φ * δ_d * (12 * ε4 - 8 * ε6 * δ_d^2)
            else
                result[j] = φ * δ[j] * (4 * ε4 - 8 * ε6 * δ_d^2)
            end
        end
        return result
    end
    return grad_Lφ_x
end

"""
    grad_applied_second_partial_wrt_x(basis, dim)

Get gradient of applied second-order partial derivative ∂²φ/∂x[dim]² w.r.t. evaluation point.
Returns a function (x, xi) -> Vector of ∂³φ/∂x[j]∂x[dim]² for each j.
"""
grad_applied_second_partial_wrt_x(::PHS1, dim::Int) = grad_second_partial_phs1_wrt_x(dim)
grad_applied_second_partial_wrt_x(::PHS3, dim::Int) = grad_second_partial_phs3_wrt_x(dim)
grad_applied_second_partial_wrt_x(::PHS5, dim::Int) = grad_second_partial_phs5_wrt_x(dim)
grad_applied_second_partial_wrt_x(::PHS7, dim::Int) = grad_second_partial_phs7_wrt_x(dim)
grad_applied_second_partial_wrt_x(basis::IMQ, dim::Int) = grad_second_partial_imq_wrt_x(basis.ε, dim)
grad_applied_second_partial_wrt_x(basis::Gaussian, dim::Int) = grad_second_partial_gaussian_wrt_x(basis.ε, dim)

"""
    grad_applied_second_partial_wrt_xi(basis, dim)

Gradient w.r.t. data point. By symmetry (φ depends on x-xi), always the negation of _wrt_x.
"""
grad_applied_second_partial_wrt_xi(b, dim::Int) = negate_grad(grad_applied_second_partial_wrt_x(b, dim))

"""
    grad_applied_partial_wrt_x(basis, dim)

Get gradient of applied partial derivative operator w.r.t. evaluation point.
"""
grad_applied_partial_wrt_x(::PHS1, dim::Int) = grad_partial_phs1_wrt_x(dim)
grad_applied_partial_wrt_x(::PHS3, dim::Int) = grad_partial_phs3_wrt_x(dim)
grad_applied_partial_wrt_x(::PHS5, dim::Int) = grad_partial_phs5_wrt_x(dim)
grad_applied_partial_wrt_x(::PHS7, dim::Int) = grad_partial_phs7_wrt_x(dim)
grad_applied_partial_wrt_x(basis::IMQ, dim::Int) = grad_partial_imq_wrt_x(basis.ε, dim)
grad_applied_partial_wrt_x(basis::Gaussian, dim::Int) = grad_partial_gaussian_wrt_x(basis.ε, dim)

"""
    grad_applied_partial_wrt_xi(basis, dim)

Get gradient of applied partial derivative operator w.r.t. data point.
By symmetry, always the negation of the `_wrt_x` version.
"""
grad_applied_partial_wrt_xi(b, dim::Int) = negate_grad(grad_applied_partial_wrt_x(b, dim))

"""
    grad_applied_laplacian_wrt_x(basis)

Get gradient of applied Laplacian operator w.r.t. evaluation point.
"""
grad_applied_laplacian_wrt_x(::PHS1) = grad_laplacian_phs1_wrt_x()
grad_applied_laplacian_wrt_x(::PHS3) = grad_laplacian_phs3_wrt_x()
grad_applied_laplacian_wrt_x(::PHS5) = grad_laplacian_phs5_wrt_x()
grad_applied_laplacian_wrt_x(::PHS7) = grad_laplacian_phs7_wrt_x()
grad_applied_laplacian_wrt_x(basis::IMQ) = grad_laplacian_imq_wrt_x(basis.ε)
grad_applied_laplacian_wrt_x(basis::Gaussian) = grad_laplacian_gaussian_wrt_x(basis.ε)

"""
    grad_applied_mixed_partial_wrt_x(basis, dim1, dim2)

Get gradient of applied mixed partial operator w.r.t. evaluation point.
"""
grad_applied_mixed_partial_wrt_x(::PHS1, dim1::Int, dim2::Int) = grad_mixed_partial_phs1_wrt_x(dim1, dim2)
grad_applied_mixed_partial_wrt_x(::PHS3, dim1::Int, dim2::Int) = grad_mixed_partial_phs3_wrt_x(dim1, dim2)
grad_applied_mixed_partial_wrt_x(::PHS5, dim1::Int, dim2::Int) = grad_mixed_partial_phs5_wrt_x(dim1, dim2)
grad_applied_mixed_partial_wrt_x(::PHS7, dim1::Int, dim2::Int) = grad_mixed_partial_phs7_wrt_x(dim1, dim2)
grad_applied_mixed_partial_wrt_x(basis::IMQ, dim1::Int, dim2::Int) = grad_mixed_partial_imq_wrt_x(basis.ε, dim1, dim2)
grad_applied_mixed_partial_wrt_x(basis::Gaussian, dim1::Int, dim2::Int) = grad_mixed_partial_gaussian_wrt_x(basis.ε, dim1, dim2)

"""
    grad_applied_laplacian_wrt_xi(basis)

Get gradient of applied Laplacian operator w.r.t. data point.
By symmetry, always the negation of the `_wrt_x` version.
"""
grad_applied_laplacian_wrt_xi(b) = negate_grad(grad_applied_laplacian_wrt_x(b))

"""
    grad_applied_mixed_partial_wrt_xi(basis, dim1, dim2)

Get gradient of applied mixed partial operator w.r.t. data point.
By symmetry, always the negation of the `_wrt_x` version.
"""
grad_applied_mixed_partial_wrt_xi(b, dim1::Int, dim2::Int) = negate_grad(grad_applied_mixed_partial_wrt_x(b, dim1, dim2))
