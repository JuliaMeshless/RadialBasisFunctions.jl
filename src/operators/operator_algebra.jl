# ============================================================================
# Identity Operator
# ============================================================================

"""
    Identity <: AbstractOperator{0}

Identity operator — returns the basis function unchanged. Useful in operator algebra
to represent the function itself (e.g., `Laplacian() + k² * Identity()` for Helmholtz).
"""
struct Identity <: AbstractOperator{0} end
(::Identity)(basis) = basis
print_op(::Identity) = "Identity (f)"

# ============================================================================
# Scaled Operator
# ============================================================================

"""
    ScaledOperator{N, T<:Number, O<:AbstractOperator{N}} <: AbstractOperator{N}

An operator multiplied by a scalar coefficient. Created via `α * op` or `op * α`.
"""
struct ScaledOperator{N, T <: Number, O <: AbstractOperator{N}} <: AbstractOperator{N}
    α::T
    op::O
end
ScaledOperator{N}(α::T, op::O) where {N, T <: Number, O <: AbstractOperator{N}} =
    ScaledOperator{N, T, O}(α, op)

function (s::ScaledOperator)(basis)
    f = s.op(basis)
    return (x, xc) -> s.α * f(x, xc)
end

function (s::ScaledOperator)(basis::MonomialBasis{Dim, Deg}) where {Dim, Deg}
    f = s.op(basis)
    function scaled_basis!(b, x)
        f(b, x)
        b .*= s.α
        return nothing
    end
    return ℒMonomialBasis(Dim, Deg, scaled_basis!)
end

print_op(s::ScaledOperator) = "$(s.α) × $(print_op(s.op))"

Base.:*(α::Number, op::AbstractOperator{N}) where {N} = ScaledOperator{N}(α, op)
Base.:*(op::AbstractOperator{N}, α::Number) where {N} = ScaledOperator{N}(α, op)
Base.:-(op::AbstractOperator{N}) where {N} = ScaledOperator{N}(-1, op)

# ============================================================================
# Operator Algebra on RadialBasisOperator (precomputed weights)
# ============================================================================

for op in (:+, :-)
    @eval function Base.$op(op1::RadialBasisOperator, op2::RadialBasisOperator)
        _check_compatible(op1, op2)
        !is_cache_valid(op1) && update_weights!(op1)
        !is_cache_valid(op2) && update_weights!(op2)
        ℒ = Base.$op(op1.ℒ, op2.ℒ)
        new_weights = _combine_weights(Base.$op, op1.weights, op2.weights)
        return RadialBasisOperator(
            ℒ, new_weights, op1.data, op1.eval_points, op1.adjl, op1.basis, true;
            device = op1.device,
        )
    end
end

for op in (:+, :-)
    @eval function Base.$op(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where {N}
        function additive_ℒ(basis)
            return additive_ℒrbf(x1, x2) = Base.$op(op1(basis)(x1, x2), op2(basis)(x1, x2))
        end
        function additive_ℒ(basis::MonomialBasis{Dim, Deg}) where {Dim, Deg}
            f1 = op1(basis)
            f2 = op2(basis)
            function additive_ℒMon(b, x)
                b .= Base.$op(f1(x), f2(x))
                return nothing
            end
            return ℒMonomialBasis(Dim, Deg, additive_ℒMon)
        end
        return Custom{N}(additive_ℒ)
    end
end

function _check_compatible(op1::RadialBasisOperator, op2::RadialBasisOperator)
    if (length(op1.data) != length(op2.data)) || !all(op1.data .≈ op2.data)
        throw(
            ArgumentError("Can not add operators that were not built with the same data.")
        )
    end
    return if op1.adjl != op2.adjl
        throw(ArgumentError("Can not add operators that do not have the same stencils."))
    end
end

_combine_weights(op, w1, w2) = op(w1, w2)
_combine_weights(op, w1::Tuple, w2::Tuple) = map((a, b) -> op(a, b), w1, w2)
