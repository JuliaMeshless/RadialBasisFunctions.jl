#=
Differentiation rules for RadialBasisOperator evaluation.

The core operation `op(x)` computes `weights * x` where `weights` is a sparse matrix.
For reverse-mode AD, the adjoint is simply `weights' * Δy`.

These rules treat the operator weights as constants (they depend on point geometry,
not on the field values being differentiated). This is the typical use case in
PDE solving and optimization where point positions are fixed.
=#

# Scalar-valued operator: y = W * x
# Pullback: Δx = W' * Δy
function ChainRulesCore.rrule(
    ::typeof(_eval_op),
    op::RadialBasisOperator,
    x::AbstractVector
)
    y = _eval_op(op, x)

    function _eval_op_pullback(Δy)
        Δy_unthunked = unthunk(Δy)
        Δx = op.weights' * Δy_unthunked
        return NoTangent(), NoTangent(), Δx
    end

    return y, _eval_op_pullback
end

# Vector-valued operator (Gradient): y[:,d] = W[d] * x for each dimension d
# The weights are stored as a tuple of sparse matrices, one per dimension.
# Pullback: Δx = Σ_d W[d]' * Δy[:,d]
function ChainRulesCore.rrule(
    ::typeof(_eval_op),
    op::RadialBasisOperator{<:VectorValuedOperator{D}},
    x::AbstractVector
) where {D}
    y = _eval_op(op, x)

    function _eval_op_vector_pullback(Δy)
        Δy_unthunked = unthunk(Δy)
        # Use similar to maintain array type (CPU/GPU compatibility)
        Δx = similar(x)
        fill!(Δx, zero(eltype(Δx)))
        for d in 1:D
            Δx .+= op.weights[d]' * view(Δy_unthunked, :, d)
        end
        return NoTangent(), NoTangent(), Δx
    end

    return y, _eval_op_vector_pullback
end

# In-place scalar operator: mul!(y, W, x)
# For completeness, though in-place operations are tricky with AD
function ChainRulesCore.rrule(
    ::typeof(_eval_op),
    op::RadialBasisOperator,
    y::AbstractVector,
    x::AbstractVector
)
    result = _eval_op(op, y, x)

    function _eval_op_inplace_pullback(Δy)
        Δy_unthunked = unthunk(Δy)
        Δx = op.weights' * Δy_unthunked
        # y is mutated, so its tangent is the same as the output tangent
        return NoTangent(), NoTangent(), Δy_unthunked, Δx
    end

    return result, _eval_op_inplace_pullback
end
