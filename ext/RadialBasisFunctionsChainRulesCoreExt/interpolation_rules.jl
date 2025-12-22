#=
Differentiation rules for Interpolator evaluation.

The interpolator computes:
  f(x) = Σᵢ wᵢ φ(x, xᵢ) + Σⱼ wⱼ pⱼ(x)

where φ is the RBF kernel and pⱼ are polynomial basis functions.

For reverse-mode AD, we compute:
  ∂f/∂x = Σᵢ wᵢ ∇φ(x, xᵢ) + Σⱼ wⱼ ∇pⱼ(x)

The weights (wᵢ, wⱼ) and data points (xᵢ) are treated as constants.
=#

function ChainRulesCore.rrule(interp::Interpolator, x::AbstractVector)
    y = interp(x)

    function interpolator_pullback(Δy)
        Δy_real = unthunk(Δy)

        # Initialize gradient accumulator
        Δx = zero(x)

        # RBF contribution: Σᵢ wᵢ ∇φ(x, xᵢ)
        grad_fn = ∇(interp.rbf_basis)
        for i in eachindex(interp.rbf_weights)
            ∇φ = grad_fn(x, interp.x[i])
            Δx = Δx .+ (interp.rbf_weights[i] * Δy_real) .* ∇φ
        end

        # Polynomial contribution: Σⱼ wⱼ ∇pⱼ(x)
        if !isempty(interp.monomial_weights)
            dim = length(x)
            n_terms = length(interp.monomial_weights)

            # Get the gradient operator for the monomial basis
            # ∇(monomial_basis) returns a callable that fills a matrix
            ∇mon = ∇(interp.monomial_basis)
            ∇p = zeros(eltype(x), n_terms, dim)
            ∇mon(∇p, x)

            # Accumulate: Σⱼ wⱼ ∇pⱼ(x)
            for j in eachindex(interp.monomial_weights)
                Δx = Δx .+ (interp.monomial_weights[j] * Δy_real) .* view(∇p, j, :)
            end
        end

        return NoTangent(), Δx
    end

    return y, interpolator_pullback
end

# Batch evaluation: interp([x1, x2, ...]) returns [f(x1), f(x2), ...]
# The pullback needs to accumulate gradients for each input point.
function ChainRulesCore.rrule(
    interp::Interpolator,
    xs::Vector{<:AbstractVector}
)
    ys = interp(xs)

    function interpolator_batch_pullback(Δys)
        Δys_real = unthunk(Δys)

        # Compute gradient for each input point
        Δxs = similar(xs)
        for (i, x) in enumerate(xs)
            Δx = zero(x)

            # RBF contribution
            grad_fn = ∇(interp.rbf_basis)
            for j in eachindex(interp.rbf_weights)
                ∇φ = grad_fn(x, interp.x[j])
                Δx = Δx .+ (interp.rbf_weights[j] * Δys_real[i]) .* ∇φ
            end

            # Polynomial contribution
            if !isempty(interp.monomial_weights)
                dim = length(x)
                n_terms = length(interp.monomial_weights)
                ∇mon = ∇(interp.monomial_basis)
                ∇p = zeros(eltype(x), n_terms, dim)
                ∇mon(∇p, x)

                for k in eachindex(interp.monomial_weights)
                    Δx = Δx .+ (interp.monomial_weights[k] * Δys_real[i]) .* view(∇p, k, :)
                end
            end

            Δxs[i] = Δx
        end

        return NoTangent(), Δxs
    end

    return ys, interpolator_batch_pullback
end
