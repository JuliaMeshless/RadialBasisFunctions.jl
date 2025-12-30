#=
Second derivatives of applied operators for RHS backward pass.

For the backward pass through _build_weights, we need:
  ∂/∂x [ℒφ(x, xi)] and ∂/∂xi [ℒφ(x, xi)]

where ℒφ is the operator applied to the basis function (e.g., ∂φ/∂dim for Partial).

These are effectively Hessian-like terms of the basis function.
=#

using Distances: euclidean

const AVOID_INF = RadialBasisFunctions.AVOID_INF

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
        if r < 1e-12
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
        if r < 1e-12
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
    get_grad_Lrbf_wrt_x(ℒrbf, basis)

Get the function that computes ∂/∂x[ℒφ(x, xi)] for the given operator and basis.
"""
function get_grad_Lrbf_wrt_x(ℒrbf, basis::PHS1)
    # Detect operator type from the applied operator
    # For now, we support Partial and Laplacian
    return _get_grad_Lrbf_wrt_x_impl(ℒrbf, basis)
end

function get_grad_Lrbf_wrt_x(ℒrbf, basis::PHS3)
    return _get_grad_Lrbf_wrt_x_impl(ℒrbf, basis)
end

function get_grad_Lrbf_wrt_x(ℒrbf, basis::PHS5)
    return _get_grad_Lrbf_wrt_x_impl(ℒrbf, basis)
end

function get_grad_Lrbf_wrt_x(ℒrbf, basis::PHS7)
    return _get_grad_Lrbf_wrt_x_impl(ℒrbf, basis)
end

function get_grad_Lrbf_wrt_xi(ℒrbf, basis::PHS1)
    return _get_grad_Lrbf_wrt_xi_impl(ℒrbf, basis)
end

function get_grad_Lrbf_wrt_xi(ℒrbf, basis::PHS3)
    return _get_grad_Lrbf_wrt_xi_impl(ℒrbf, basis)
end

function get_grad_Lrbf_wrt_xi(ℒrbf, basis::PHS5)
    return _get_grad_Lrbf_wrt_xi_impl(ℒrbf, basis)
end

function get_grad_Lrbf_wrt_xi(ℒrbf, basis::PHS7)
    return _get_grad_Lrbf_wrt_xi_impl(ℒrbf, basis)
end

# Implementation using operator traits
# The applied operator ℒrbf is a closure - we detect type via inspection

function _get_grad_Lrbf_wrt_x_impl(ℒrbf, basis::PHS1)
    # Try to detect if it's a partial or laplacian
    # This is a simplified approach - in practice you may need to store operator info
    error("Operator type detection not yet implemented for PHS1. Pass operator info explicitly.")
end

function _get_grad_Lrbf_wrt_xi_impl(ℒrbf, basis::PHS1)
    error("Operator type detection not yet implemented for PHS1. Pass operator info explicitly.")
end

# For now, provide explicit dispatch on operator types
# These will be called from the backward pass with known operator types

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
