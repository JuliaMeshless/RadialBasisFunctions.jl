# ============================================================================
# Identity Operator
# ============================================================================

"""
    Identity <: AbstractOperator{0}

Identity operator — returns the basis function unchanged. Useful in operator algebra
to represent the function itself (e.g., `Laplacian() + k² * Identity()` for Helmholtz).
"""
struct Identity <: AbstractOperator{0} end
(::Identity)(basis::AbstractBasis) = basis
print_op(::Identity) = "Identity (f)"

# ============================================================================
# Action Kernels
# ============================================================================

"""
    SumKernel{Fs<:Tuple}

Callable summing component basis actions. Supports the standard `(x, xᵢ)` form and
the Hermite normal form `(x, xᵢ, normal)`.
"""
struct SumKernel{Fs <: Tuple}
    fs::Fs
end
(k::SumKernel)(x, xᵢ) = sum(f -> f(x, xᵢ), k.fs)
(k::SumKernel)(x, xᵢ, normal) = sum(f -> f(x, xᵢ, normal), k.fs)

"""
    ScaledKernel{T<:Number, F}

Callable scaling a basis action by `α`. Supports the standard `(x, xᵢ)` form and
the Hermite normal form `(x, xᵢ, normal)`.
"""
struct ScaledKernel{T <: Number, F}
    α::T
    f::F
end
(k::ScaledKernel)(x, xᵢ) = k.α * k.f(x, xᵢ)
(k::ScaledKernel)(x, xᵢ, normal) = k.α * k.f(x, xᵢ, normal)

# Normal-form support is determined by the wrapped actions, not the kernel methods
function _supports_normal_form(k::SumKernel, x, n)
    return all(f -> _supports_normal_form(f, x, n), k.fs)
end
_supports_normal_form(k::ScaledKernel, x, n) = _supports_normal_form(k.f, x, n)

# Monomial equivalents: in-place `(b, x)` form, wrapped in ℒMonomialBasis
struct MonomialSumKernel{Fs <: Tuple} <: Function
    fs::Fs
end
function (k::MonomialSumKernel)(b, x)
    cache = similar(b)
    first(k.fs)(b, x)
    for f in Base.tail(k.fs)
        f(cache, x)
        b .+= cache
    end
    return nothing
end

struct MonomialScaledKernel{T <: Number, F} <: Function
    α::T
    f::F
end
function (k::MonomialScaledKernel)(b, x)
    k.f(b, x)
    b .*= k.α
    return nothing
end

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

(s::ScaledOperator)(basis::AbstractBasis) = _scale_action(s.α, s.op(basis))

_scale_action(α, f) = ScaledKernel(α, f)
_scale_action(α, fs::Tuple) = map(f -> ScaledKernel(α, f), fs)

function (s::ScaledOperator)(basis::MonomialBasis{Dim, Deg}) where {Dim, Deg}
    return _scale_mono_action(Dim, Deg, s.α, s.op(basis))
end

function _scale_mono_action(dim, deg, α, f)
    return ℒMonomialBasis(dim, deg, MonomialScaledKernel(α, f))
end
function _scale_mono_action(dim, deg, α, fs::Tuple)
    return map(f -> ℒMonomialBasis(dim, deg, MonomialScaledKernel(α, f)), fs)
end

print_op(s::ScaledOperator) = "$(s.α) × $(print_op(s.op))"

Base.:*(α::Number, op::AbstractOperator{N}) where {N} = ScaledOperator{N}(α, op)
Base.:*(op::AbstractOperator{N}, α::Number) where {N} = ScaledOperator{N}(α, op)
Base.:-(op::AbstractOperator{N}) where {N} = ScaledOperator{N}(-1, op)
Base.:/(op::AbstractOperator{N}, α::Number) where {N} = ScaledOperator{N}(inv(α), op)

# ============================================================================
# Sum Operator
# ============================================================================

"""
    SumOperator{N, Ops<:Tuple{Vararg{AbstractOperator}}} <: AbstractOperator{N}

Sum of operators sharing output rank `N`. Created via `op1 + op2` or `op1 - op2`
(subtraction stores `-1 * op2`); nested sums flatten into a single n-ary term tuple.
"""
struct SumOperator{N, Ops <: Tuple{Vararg{AbstractOperator}}} <: AbstractOperator{N}
    ops::Ops
end
SumOperator{N}(ops::Ops) where {N, Ops <: Tuple{Vararg{AbstractOperator}}} =
    SumOperator{N, Ops}(ops)

_terms(op::AbstractOperator) = (op,)
_terms(s::SumOperator) = s.ops

function Base.:+(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where {N}
    return SumOperator{N}((_terms(op1)..., _terms(op2)...))
end
Base.:-(op1::AbstractOperator{N}, op2::AbstractOperator{N}) where {N} = op1 + (-op2)

(s::SumOperator)(basis::AbstractBasis) = _combine_actions(map(op -> op(basis), s.ops)...)

function (s::SumOperator)(basis::MonomialBasis{Dim, Deg}) where {Dim, Deg}
    return _combine_mono_actions(Dim, Deg, map(op -> op(basis), s.ops)...)
end

function _combine_actions(actions...)
    _check_no_mixed_actions(actions)
    return SumKernel(actions)
end
function _combine_actions(actions::Tuple...)
    n = _check_same_arity(actions)
    return ntuple(i -> SumKernel(map(a -> a[i], actions)), n)
end

function _combine_mono_actions(dim, deg, actions...)
    _check_no_mixed_actions(actions)
    return ℒMonomialBasis(dim, deg, MonomialSumKernel(actions))
end
function _combine_mono_actions(dim, deg, actions::Tuple...)
    n = _check_same_arity(actions)
    return ntuple(
        i -> ℒMonomialBasis(dim, deg, MonomialSumKernel(map(a -> a[i], actions))), n
    )
end

function _check_no_mixed_actions(actions)
    return if any(a -> a isa Tuple, actions)
        throw(
            ArgumentError(
                "Cannot sum operators whose basis actions mix single callables and " *
                    "per-dimension tuples (e.g. a scalar operator such as Laplacian with " *
                    "a gradient-family operator such as Divergence or Jacobian)."
            )
        )
    end
end

function _check_same_arity(actions)
    n = length(first(actions))
    all(a -> length(a) == n, actions) || throw(
        ArgumentError("Cannot sum gradient-family operators with different dimensions.")
    )
    return n
end

print_op(s::SumOperator) = join(map(print_op, s.ops), " + ")

# ============================================================================
# Evaluation of Composed Operators
# ============================================================================

# Rank-0 gradient-family compositions (e.g. Divergence + Divergence) carry tuple
# weights whose contraction semantics are defined by the leading concrete operator;
# rewrap so _eval_op dispatches on that type.
_leading_op(op::AbstractOperator) = op
_leading_op(s::ScaledOperator) = _leading_op(s.op)
_leading_op(s::SumOperator) = _leading_op(first(s.ops))

function _rewrap_leading(op::RadialBasisOperator)
    return RadialBasisOperator(
        _leading_op(op.ℒ), op.weights, op.data, op.eval_points, op.adjl, op.basis,
        is_cache_valid(op); device = op.device,
    )
end

function _eval_op(op::RadialBasisOperator{<:SumOperator{0}}, x::AbstractMatrix)
    return _eval_op(_rewrap_leading(op), x)
end
function _eval_op(op::RadialBasisOperator{<:ScaledOperator{0}}, x::AbstractMatrix)
    return _eval_op(_rewrap_leading(op), x)
end

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

# Catch rank-mismatched operator algebra with a clear error
for op in (:+, :-)
    @eval function Base.$op(op1::AbstractOperator, op2::AbstractOperator)
        throw(
            ArgumentError(
                "Cannot combine operators with different output ranks: " *
                    "$(print_op(op1)) (rank $(output_rank(op1))) and " *
                    "$(print_op(op2)) (rank $(output_rank(op2)))"
            )
        )
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
