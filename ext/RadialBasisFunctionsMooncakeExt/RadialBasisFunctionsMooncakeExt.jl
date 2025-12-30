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
using StaticArrays: SVector

# Import the _eval_op function we need to wrap
import RadialBasisFunctions: _eval_op, RadialBasisOperator, Interpolator
import RadialBasisFunctions: PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: VectorValuedOperator
import RadialBasisFunctions: _build_weights, Partial, Laplacian, AbstractRadialBasis

# =============================================================================
# Custom increment_and_get_rdata! for SVector types
# =============================================================================
# Mooncake represents SVector{N,T} tangents as Tangent{@NamedTuple{data::NTuple{N,T}}}
# ChainRulesCore rrules return Vector{SVector{N,T}} tangents
# This method bridges the gap by incrementing fdata in-place

# Generic version that handles any SVector dimension
function Mooncake.increment_and_get_rdata!(
    f::Vector{<:Mooncake.Tangent},
    ::Mooncake.NoRData,
    t::Vector{SVector{N,T}}
) where {N,T}
    for i in eachindex(f, t)
        # Mooncake.Tangent has a `fields` field containing the NamedTuple
        # The NamedTuple has a `data` field with the tuple of values
        old_data = f[i].fields.data
        # Create new tuple with incremented values
        sv = t[i]
        new_data = ntuple(j -> old_data[j] + sv[j], Val(N))
        # Reconstruct the tangent with the same type
        f[i] = typeof(f[i])((data=new_data,))
    end
    return Mooncake.NoRData()
end

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

# _build_weights rules for shape optimization
# These enable differentiating through operator construction w.r.t. point positions

# Partial operator with different PHS types
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS1}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS3}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS5}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS7}
)

# Laplacian operator with different PHS types
Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS1}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS3}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS5}
)

Mooncake.@from_rrule(
    Mooncake.DefaultCtx,
    Tuple{typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS7}
)

end # module
