#=
Shared backward pass for Interpolator evaluation.

Computes the gradient of the interpolator output w.r.t. the evaluation point:
  ∂f/∂x = Σᵢ wᵢ ∇φ(x, xᵢ) + Σⱼ wⱼ ∇pⱼ(x)

Used by both Enzyme and Mooncake extensions.
=#

"""
    _interpolator_point_gradient!(Δx, interp::Interpolator, x, Δy)

Accumulate the gradient of `interp(x) * Δy` into `Δx`.

RBF contribution: Σᵢ wᵢ * Δy * ∇φ(x, xᵢ)
Polynomial contribution: Σⱼ wⱼ * Δy * ∇pⱼ(x)
"""
function _interpolator_point_gradient!(Δx, interp::Interpolator, x, Δy)
    # RBF contribution
    grad_fn = ∇(interp.rbf_basis)
    for i in eachindex(interp.rbf_weights)
        ∇φ = grad_fn(x, interp.x[i])
        coeff = interp.rbf_weights[i] * Δy
        for d in eachindex(Δx)
            Δx[d] += coeff * ∇φ[d]
        end
    end

    # Polynomial contribution
    if !isempty(interp.monomial_weights)
        dim = length(x)
        n_terms = length(interp.monomial_weights)
        ∇mon = ∇(interp.monomial_basis)
        ∇p = zeros(eltype(x), n_terms, dim)
        ∇mon(∇p, x)

        for j in eachindex(interp.monomial_weights)
            coeff = interp.monomial_weights[j] * Δy
            for d in 1:dim
                Δx[d] += coeff * ∇p[j, d]
            end
        end
    end

    return nothing
end

"""
    _interpolator_constructor_backward(Δrbf_weights, Δmon_weights, A, k)

Backward pass for the Interpolator constructor w.r.t. `y` (the data values).

Given cotangents of `rbf_weights` and `monomial_weights`, computes the cotangent of `y`
using the implicit function theorem. Since `w = A⁻¹ [y; 0]` and `A` is constant w.r.t. `y`:

    Δy = (A⁻¹ [Δrbf_weights; Δmon_weights])[1:k]

`A` may be the collocation matrix or a factorization of it (both extension rules pass
the cached `BunchKaufman`; symmetric ⟹ self-adjoint, so `A⁻ᵀ = A⁻¹` holds either way).

Used by both the Mooncake and Enzyme extensions.
"""
function _interpolator_constructor_backward(Δrbf_weights, Δmon_weights, A, k)
    Δw = vcat(Δrbf_weights, Δmon_weights)
    Δb = A \ Δw  # A is symmetric ⟹ A⁻ᵀ = A⁻¹
    return Δb[1:k]
end

"""
    _interpolator_weight_cotangents!(Δrbf_w, Δmon_w, interp::Interpolator, x, Δy)

Accumulate the cotangent of `interp(x) * Δy` w.r.t. the interpolator weights:

    Δrbf_w[i] += Δy * φ(x, xᵢ)
    Δmon_w[j] += Δy * pⱼ(x)

Used by the Enzyme extension's Duplicated-Interpolator evaluation rules to deposit
weight cotangents into the shadow Interpolator consumed by the constructor rule.
"""
function _interpolator_weight_cotangents!(Δrbf_w, Δmon_w, interp::Interpolator, x, Δy)
    for i in eachindex(Δrbf_w)
        Δrbf_w[i] += Δy * interp.rbf_basis(x, interp.x[i])
    end
    if !isempty(Δmon_w)
        vals = interp.monomial_basis(x)
        for j in eachindex(Δmon_w)
            Δmon_w[j] += Δy * vals[j]
        end
    end
    return nothing
end
