#=
Derivatives of basis functions and applied operators with respect to shape parameter ε.

These functions are used in the backward pass of _build_weights to compute ∂W/∂ε.

For a basis function φ(r; ε) where r = |x - xi|:
- ∂φ/∂ε: derivative of basis function w.r.t. shape parameter
- ∂(ℒφ)/∂ε: derivative of applied operator w.r.t. shape parameter

Currently supported:
- Gaussian: φ(r) = exp(-(εr)²)
- IMQ: φ(r) = 1/√((εr)² + 1)
=#

# =============================================================================
# Gaussian basis: φ(r) = exp(-ε²r²)
# =============================================================================

"""
    ∂φ_∂ε(basis::Gaussian, x, xi)

Derivative of Gaussian basis function w.r.t. shape parameter ε.

φ(r) = exp(-ε²r²)
∂φ/∂ε = -2εr² exp(-ε²r²)
"""
function ∂φ_∂ε(basis::Gaussian, x, xi)
    ε = basis.ε
    r² = sqeuclidean(x, xi)
    return -2 * ε * r² * exp(-ε^2 * r²)
end

"""
    ∂Laplacian_φ_∂ε(basis::Gaussian, x, xi)

Derivative of Laplacian of Gaussian basis w.r.t. shape parameter ε.

∇²φ = (-2ε²D + 4ε⁴r²) exp(-ε²r²), where D = dimension
∂(∇²φ)/∂ε = exp(-ε²r²) [-4εD + 16ε³r² + 4ε³r²D - 8ε⁵r⁴]
"""
function ∂Laplacian_φ_∂ε(basis::Gaussian, x, xi)
    ε = basis.ε
    ε² = ε^2
    ε³ = ε^3
    ε⁵ = ε^5
    r² = sqeuclidean(x, xi)
    D = length(x)
    φ = exp(-ε² * r²)
    return φ * (-4 * ε * D + 16 * ε³ * r² + 4 * ε³ * r² * D - 8 * ε⁵ * r²^2)
end

"""
    ∂Partial_φ_∂ε(basis::Gaussian, dim::Int, x, xi)

Derivative of first partial derivative of Gaussian basis w.r.t. shape parameter ε.

∂φ/∂x_dim = -2ε²(x_dim - xi_dim) exp(-ε²r²)
∂/∂ε[∂φ/∂x_dim] = 4ε(x_dim - xi_dim)(ε²r² - 1) exp(-ε²r²)
"""
function ∂Partial_φ_∂ε(basis::Gaussian, dim::Int, x, xi)
    ε = basis.ε
    ε² = ε^2
    r² = sqeuclidean(x, xi)
    Δ_dim = x[dim] - xi[dim]
    φ = exp(-ε² * r²)
    return 4 * ε * Δ_dim * (ε² * r² - 1) * φ
end

# =============================================================================
# IMQ basis: φ(r) = 1/√(ε²r² + 1)
# =============================================================================

"""
    ∂φ_∂ε(basis::IMQ, x, xi)

Derivative of IMQ basis function w.r.t. shape parameter ε.

φ(r) = (ε²r² + 1)^(-1/2)
∂φ/∂ε = -εr² (ε²r² + 1)^(-3/2)
"""
function ∂φ_∂ε(basis::IMQ, x, xi)
    ε = basis.ε
    r² = sqeuclidean(x, xi)
    s = ε^2 * r² + 1
    return -ε * r² / sqrt(s^3)
end

"""
    ∂Laplacian_φ_∂ε(basis::IMQ, x, xi)

Derivative of Laplacian of IMQ basis w.r.t. shape parameter ε.

Let s = ε²r² + 1, D = dimension
∇²φ = -ε²D/s^(3/2) + 3ε⁴r²/s^(5/2)
∂(∇²φ)/∂ε = ∂/∂ε[-ε²D s^(-3/2) + 3ε⁴r² s^(-5/2)]
"""
function ∂Laplacian_φ_∂ε(basis::IMQ, x, xi)
    ε = basis.ε
    ε² = ε^2
    ε³ = ε^3
    ε⁴ = ε^4
    ε⁵ = ε^5
    r² = sqeuclidean(x, xi)
    r⁴ = r²^2
    D = length(x)
    s = ε² * r² + 1

    # ∂/∂ε[-ε²D s^(-3/2)] = -2εD s^(-3/2) + ε²D (3/2) s^(-5/2) · 2εr²
    #                      = -2εD s^(-3/2) + 3ε³D r² s^(-5/2)
    term1 = -2 * ε * D / sqrt(s^3) + 3 * ε³ * D * r² / sqrt(s^5)

    # ∂/∂ε[3ε⁴r² s^(-5/2)] = 12ε³r² s^(-5/2) + 3ε⁴r² (-5/2) s^(-7/2) · 2εr²
    #                       = 12ε³r² s^(-5/2) - 15ε⁵r⁴ s^(-7/2)
    term2 = 12 * ε³ * r² / sqrt(s^5) - 15 * ε⁵ * r⁴ / sqrt(s^7)

    return term1 + term2
end

"""
    ∂Partial_φ_∂ε(basis::IMQ, dim::Int, x, xi)

Derivative of first partial derivative of IMQ basis w.r.t. shape parameter ε.

∂φ/∂x_dim = ε²(xi_dim - x_dim) s^(-3/2)
∂/∂ε[∂φ/∂x_dim] = 2ε(xi_dim - x_dim) s^(-3/2) + ε²(xi_dim - x_dim)(-3/2)s^(-5/2) · 2εr²
                = (xi_dim - x_dim)[2ε s^(-3/2) - 3ε³r² s^(-5/2)]
"""
function ∂Partial_φ_∂ε(basis::IMQ, dim::Int, x, xi)
    ε = basis.ε
    ε³ = ε^3
    r² = sqeuclidean(x, xi)
    Δ_dim = xi[dim] - x[dim]  # Note: IMQ has opposite sign convention
    s = ε^2 * r² + 1
    return Δ_dim * (2 * ε / sqrt(s^3) - 3 * ε³ * r² / sqrt(s^5))
end

# =============================================================================
# PHS basis: no shape parameter, gradients are zero
# =============================================================================

∂φ_∂ε(::AbstractPHS, x, xi) = zero(eltype(x))
∂Laplacian_φ_∂ε(::AbstractPHS, x, xi) = zero(eltype(x))
∂Partial_φ_∂ε(::AbstractPHS, ::Int, x, xi) = zero(eltype(x))
