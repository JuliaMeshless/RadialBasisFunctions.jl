#=
Differentiation rules for Interpolator construction and evaluation.

Construction: Interpolator(x, y, basis)
  Solves A * w = [y; 0] where A is the collocation matrix.
  Forward: build A, factor, solve for w
  Pullback: Δy = (A⁻ᵀ Δw)[1:k] via adjoint solve

Evaluation: interp(x)
  Computes f(x) = Σᵢ wᵢ φ(x, xᵢ) + Σⱼ wⱼ pⱼ(x)
  Pullback returns gradients w.r.t. both evaluation point and weights
=#

using LinearAlgebra: Symmetric, factorize

# ============================================================================
# Interpolator Construction Rule
# ============================================================================

"""
    rrule for Interpolator construction.

Differentiates through w = A \\ [y; 0] via adjoint solve.
Backward: Δy = (A⁻ᵀ Δw)[1:k]
"""
function ChainRulesCore.rrule(
        ::Type{Interpolator},
        x::AbstractVector,
        y::AbstractVector,
        basis::AbstractRadialBasis,
    )
    # Forward pass: build collocation matrix, factor, solve
    dim = length(first(x))
    k = length(x)
    npoly = binomial(dim + basis.poly_deg, basis.poly_deg)
    n = k + npoly
    mon = MonomialBasis(dim, basis.poly_deg)
    T = promote_type(eltype(first(x)), eltype(y))

    A = Symmetric(zeros(T, n, n))
    _build_collocation_matrix!(A, x, basis, mon, k)

    # Factor for reuse in backward pass
    A_factored = factorize(A)
    b = vcat(y, zeros(T, npoly))
    w = A_factored \ b

    interp = Interpolator(x, y, w[1:k], w[(k + 1):end], basis, mon)

    function interpolator_construction_pullback(Δinterp_raw)
        Δinterp = unthunk(Δinterp_raw)

        # Handle NoTangent or ZeroTangent
        if Δinterp isa NoTangent || Δinterp isa ZeroTangent
            return NoTangent(), NoTangent(), ZeroTangent(), NoTangent()
        end

        # Extract weight tangents from Interpolator tangent
        # For Tangent types, use getproperty which handles backing correctly
        Δw_rbf = hasproperty(Δinterp, :rbf_weights) ? getproperty(Δinterp, :rbf_weights) : ZeroTangent()
        Δw_mon = hasproperty(Δinterp, :monomial_weights) ? getproperty(Δinterp, :monomial_weights) : ZeroTangent()

        # Handle ZeroTangent for both components
        if Δw_rbf isa ZeroTangent && Δw_mon isa ZeroTangent
            return NoTangent(), NoTangent(), ZeroTangent(), NoTangent()
        end

        # Combine into full Δw vector
        Δw_rbf_vec = Δw_rbf isa ZeroTangent ? zeros(T, k) : collect(Δw_rbf)
        Δw_mon_vec = Δw_mon isa ZeroTangent ? zeros(T, npoly) : collect(Δw_mon)
        Δw = vcat(Δw_rbf_vec, Δw_mon_vec)

        # Adjoint solve: Δb = A⁻ᵀ Δw
        Δb = A_factored' \ Δw

        # Extract Δy (first k elements of b correspond to y values)
        Δy = Δb[1:k]

        return NoTangent(), NoTangent(), Δy, NoTangent()
    end

    return interp, interpolator_construction_pullback
end

# Convenience wrapper with default basis
function ChainRulesCore.rrule(
        ::Type{Interpolator},
        x::AbstractVector,
        y::AbstractVector,
    )
    return ChainRulesCore.rrule(Interpolator, x, y, PHS())
end

# ============================================================================
# Interpolator Evaluation Rules
# ============================================================================

function ChainRulesCore.rrule(interp::Interpolator, x::AbstractVector)
    y = interp(x)

    function interpolator_pullback(Δy)
        Δy_real = unthunk(Δy)

        # Initialize gradient accumulators
        Δx = zero(x)
        T = eltype(x)

        # RBF contribution: Σᵢ wᵢ ∇φ(x, xᵢ)
        # Gradient w.r.t. weights: ∂y/∂wᵢ = φ(x, xᵢ)
        k = length(interp.rbf_weights)
        Δw_rbf = zeros(T, k)
        grad_fn = ∇(interp.rbf_basis)
        for i in eachindex(interp.rbf_weights)
            φ_val = interp.rbf_basis(x, interp.x[i])
            Δw_rbf[i] = Δy_real * φ_val
            ∇φ = grad_fn(x, interp.x[i])
            Δx = Δx .+ (interp.rbf_weights[i] * Δy_real) .* ∇φ
        end

        # Polynomial contribution: Σⱼ wⱼ ∇pⱼ(x)
        # Gradient w.r.t. weights: ∂y/∂wⱼ = pⱼ(x)
        n_mon = length(interp.monomial_weights)
        Δw_mon = zeros(T, n_mon)
        if !isempty(interp.monomial_weights)
            dim = length(x)

            # Get polynomial values at x
            poly_vals = zeros(T, n_mon)
            interp.monomial_basis(poly_vals, x)

            # Get the gradient operator for the monomial basis
            ∇mon = ∇(interp.monomial_basis)
            ∇p = zeros(T, n_mon, dim)
            ∇mon(∇p, x)

            for j in eachindex(interp.monomial_weights)
                Δw_mon[j] = Δy_real * poly_vals[j]
                Δx = Δx .+ (interp.monomial_weights[j] * Δy_real) .* view(∇p, j, :)
            end
        end

        # Build Tangent for Interpolator with weight gradients
        Δinterp = Tangent{Interpolator}(;
            rbf_weights = Δw_rbf,
            monomial_weights = Δw_mon,
        )

        return Δinterp, Δx
    end

    return y, interpolator_pullback
end

# Batch evaluation: interp([x1, x2, ...]) returns [f(x1), f(x2), ...]
# The pullback needs to accumulate gradients for each input point and weights.
function ChainRulesCore.rrule(interp::Interpolator, xs::Vector{<:AbstractVector})
    ys = interp(xs)

    function interpolator_batch_pullback(Δys)
        Δys_real = unthunk(Δys)
        T = eltype(first(xs))

        # Initialize weight gradient accumulators
        k_rbf = length(interp.rbf_weights)
        n_mon = length(interp.monomial_weights)
        Δw_rbf = zeros(T, k_rbf)
        Δw_mon = zeros(T, n_mon)

        # Compute gradient for each input point
        Δxs = similar(xs)
        grad_fn = ∇(interp.rbf_basis)

        for (i, x) in enumerate(xs)
            Δx = zero(x)
            Δy_i = Δys_real[i]

            # RBF contribution
            for j in eachindex(interp.rbf_weights)
                φ_val = interp.rbf_basis(x, interp.x[j])
                Δw_rbf[j] += Δy_i * φ_val
                ∇φ = grad_fn(x, interp.x[j])
                Δx = Δx .+ (interp.rbf_weights[j] * Δy_i) .* ∇φ
            end

            # Polynomial contribution
            if !isempty(interp.monomial_weights)
                dim = length(x)

                # Get polynomial values at x
                poly_vals = zeros(T, n_mon)
                interp.monomial_basis(poly_vals, x)

                ∇mon = ∇(interp.monomial_basis)
                ∇p = zeros(T, n_mon, dim)
                ∇mon(∇p, x)

                for j in eachindex(interp.monomial_weights)
                    Δw_mon[j] += Δy_i * poly_vals[j]
                    Δx = Δx .+ (interp.monomial_weights[j] * Δy_i) .* view(∇p, j, :)
                end
            end

            Δxs[i] = Δx
        end

        # Build Tangent for Interpolator with weight gradients
        Δinterp = Tangent{Interpolator}(;
            rbf_weights = Δw_rbf,
            monomial_weights = Δw_mon,
        )

        return Δinterp, Δxs
    end

    return ys, interpolator_batch_pullback
end
