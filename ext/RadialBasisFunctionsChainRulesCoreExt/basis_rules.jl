#=
Differentiation rules for radial basis function evaluations.

Each RBF type (PHS, IMQ, Gaussian) is callable: basis(x, xi) computes φ(||x - xi||).
The analytical gradients are already implemented via the ∇(basis) function, which
returns a function that computes the gradient vector ∇φ at a given (x, xi) pair.

For reverse-mode AD:
  - d/dx[φ(x, xi)] = ∇φ(x, xi)
  - d/dxi[φ(x, xi)] = -∇φ(x, xi)  (by symmetry: φ depends on x - xi)
=#

# PHS basis functions (Polyharmonic Splines)
# Note: PHS ∇ functions accept an optional normal argument, but we use the
# default (nothing) for standard differentiation.

function ChainRulesCore.rrule(basis::PHS1, x::AbstractVector, xi::AbstractVector)
    y = basis(x, xi)

    function phs1_pullback(Δy)
        Δy_real = unthunk(Δy)
        grad_fn = ∇(basis)
        ∇φ = grad_fn(x, xi)
        Δx = Δy_real .* ∇φ
        Δxi = -Δx
        return NoTangent(), Δx, Δxi
    end

    return y, phs1_pullback
end

function ChainRulesCore.rrule(basis::PHS3, x::AbstractVector, xi::AbstractVector)
    y = basis(x, xi)

    function phs3_pullback(Δy)
        Δy_real = unthunk(Δy)
        grad_fn = ∇(basis)
        ∇φ = grad_fn(x, xi)
        Δx = Δy_real .* ∇φ
        Δxi = -Δx
        return NoTangent(), Δx, Δxi
    end

    return y, phs3_pullback
end

function ChainRulesCore.rrule(basis::PHS5, x::AbstractVector, xi::AbstractVector)
    y = basis(x, xi)

    function phs5_pullback(Δy)
        Δy_real = unthunk(Δy)
        grad_fn = ∇(basis)
        ∇φ = grad_fn(x, xi)
        Δx = Δy_real .* ∇φ
        Δxi = -Δx
        return NoTangent(), Δx, Δxi
    end

    return y, phs5_pullback
end

function ChainRulesCore.rrule(basis::PHS7, x::AbstractVector, xi::AbstractVector)
    y = basis(x, xi)

    function phs7_pullback(Δy)
        Δy_real = unthunk(Δy)
        grad_fn = ∇(basis)
        ∇φ = grad_fn(x, xi)
        Δx = Δy_real .* ∇φ
        Δxi = -Δx
        return NoTangent(), Δx, Δxi
    end

    return y, phs7_pullback
end

# IMQ (Inverse Multiquadric) basis function

function ChainRulesCore.rrule(basis::IMQ, x::AbstractVector, xi::AbstractVector)
    y = basis(x, xi)

    function imq_pullback(Δy)
        Δy_real = unthunk(Δy)
        grad_fn = ∇(basis)
        ∇φ = grad_fn(x, xi)
        Δx = Δy_real .* ∇φ
        Δxi = -Δx
        return NoTangent(), Δx, Δxi
    end

    return y, imq_pullback
end

# Gaussian basis function

function ChainRulesCore.rrule(basis::Gaussian, x::AbstractVector, xi::AbstractVector)
    y = basis(x, xi)

    function gaussian_pullback(Δy)
        Δy_real = unthunk(Δy)
        grad_fn = ∇(basis)
        ∇φ = grad_fn(x, xi)
        Δx = Δy_real .* ∇φ
        Δxi = -Δx
        return NoTangent(), Δx, Δxi
    end

    return y, gaussian_pullback
end
