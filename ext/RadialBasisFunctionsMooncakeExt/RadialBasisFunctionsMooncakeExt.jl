"""
    RadialBasisFunctionsMooncakeExt

Package extension that imports ChainRulesCore rrules into Mooncake.jl's AD system
using the `@from_rrule` macro. This enables Mooncake.jl to differentiate through
RadialBasisFunctions operations.

This extension requires both ChainRulesCore and Mooncake to be loaded.
"""
module RadialBasisFunctionsMooncakeExt

using RadialBasisFunctions
using ChainRulesCore
using Mooncake

# Import the _eval_op function we need to wrap
import RadialBasisFunctions: _eval_op, RadialBasisOperator, Interpolator
import RadialBasisFunctions: PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: VectorValuedOperator

# Import ChainRulesCore rules into Mooncake using @from_rrule
# The DefaultCtx is used for standard (non-debug) differentiation

# Operator evaluation rules - these are the most commonly differentiated
# Note: @from_rrule requires explicit signatures

# Scalar operator: _eval_op(op, x) -> vector
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_eval_op), RadialBasisOperator, Vector{Float64}}
)

# Vector-valued operator (Gradient): _eval_op(op, x) -> matrix
# This covers gradient, jacobian operators
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_eval_op), RadialBasisOperator{<:VectorValuedOperator}, Vector{Float64}}
)

# Basis function rules for common types (Float64 vectors)
# These enable differentiating through weight computation if needed

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{PHS1, Vector{Float64}, Vector{Float64}}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{PHS3, Vector{Float64}, Vector{Float64}}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{PHS5, Vector{Float64}, Vector{Float64}}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{PHS7, Vector{Float64}, Vector{Float64}}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{IMQ, Vector{Float64}, Vector{Float64}}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{Gaussian, Vector{Float64}, Vector{Float64}}
)

# Interpolator rules
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{Interpolator, Vector{Float64}}
)

end # module
