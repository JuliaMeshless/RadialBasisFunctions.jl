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
# ∂/∂x[j] of [∂φ/∂x[dim]]: row `dim` of the basis Hessian functor H(basis)
# ============================================================================

"""
    _hessian_row(basis, x, xᵢ, dim)

Compute ∂/∂x[j] of [∂φ/∂x[dim]]: row `dim` of the Hessian of `φ(x, xᵢ)`
w.r.t. `x`, extracted from the basis Hessian functor `H(basis)`. `x` is a
`StaticVector` (the only point type the backward pass produces); returns an `SVector`.
"""
function _hessian_row(
        basis::AbstractRadialBasis, x::StaticArraysCore.StaticVector{N, T}, xᵢ, dim::Int
    ) where {N, T}
    Hφ = H(basis)(x, xᵢ)
    return StaticArraysCore.SVector{N, T}(ntuple(j -> Hφ[dim, j], Val(N)))
end

"""
    _hessian_row(::PHS1, x, xᵢ, dim)

PHS1 needs exact handling: at r = 0 the derivatives of φ vanish so the gradient
contribution is 0, but the raw `H{PHS1}` functor regularizes with `avoid_inf`
(≈1e16 diagonals at r = 0, and an inexact r³ + 1e-16 denominator that distorts
rows for r ≲ 1e-5), so the row is computed directly.
"""
function _hessian_row(
        ::PHS1, x::StaticArraysCore.StaticVector{N, T}, xᵢ, dim::Int
    ) where {N, T}
    r = euclidean(x, xᵢ)
    r < oftype(r, 1.0e-12) && return zero(StaticArraysCore.SVector{N, T})
    δ = x .- xᵢ
    δ_d = δ[dim]
    r3 = r^3
    return StaticArraysCore.SVector{N, T}(
        ntuple(j -> j == dim ? 1 / r - δ_d^2 / r3 : -δ_d * δ[j] / r3, Val(N))
    )
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
        r_safe = r + avoid_inf(r)
        δ = x .- xi
        return 12 .* δ ./ r_safe
    end
    return grad_Lφ_x
end

# ============================================================================
# PHS1: φ(r) = r
# ============================================================================

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
        if r < oftype(r, 1.0e-12)
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
# ============================================================================

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
# ============================================================================

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
# Dispatch functions: map basis type to correct gradient functions
# ============================================================================

"""
    grad_applied_partial_wrt_x(basis, dim)

Get gradient of applied partial derivative operator w.r.t. evaluation point.
"""
grad_applied_partial_wrt_x(basis::AbstractRadialBasis, dim::Int) =
    (x, xi) -> _hessian_row(basis, x, xi, dim)

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
    grad_applied_laplacian_wrt_xi(basis)

Get gradient of applied Laplacian operator w.r.t. data point.
By symmetry, always the negation of the `_wrt_x` version.
"""
grad_applied_laplacian_wrt_xi(b) = negate_grad(grad_applied_laplacian_wrt_x(b))
