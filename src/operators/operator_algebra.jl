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
    @eval function Base.$op(op1::AbstractOperator, op2::AbstractOperator)
        function additive_ℒ(basis)
            return additive_ℒrbf(x1, x2) = Base.$op(op1(basis)(x1, x2), op2(basis)(x1, x2))
        end
        function additive_ℒ(basis::MonomialBasis)
            return function additive_ℒMon(b, x)
                b .= Base.$op(op1(basis)(x), op2(basis)(x))
                return nothing
            end
        end
        return Custom(additive_ℒ)
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
