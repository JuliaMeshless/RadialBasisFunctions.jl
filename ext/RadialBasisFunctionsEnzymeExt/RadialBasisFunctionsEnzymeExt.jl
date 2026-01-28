"""
    RadialBasisFunctionsEnzymeExt

Package extension that provides Enzyme.jl AD support for RadialBasisFunctions.jl
operations using `@import_rrule` to import existing ChainRulesCore rules.

This extension requires both ChainRulesCore and Enzyme to be loaded.

## Current Limitations

Enzyme.jl support for Julia 1.12+ is still in progress. As of January 2026, there are
known issues with custom rules on Julia 1.12. For reliable Enzyme support, use Julia 1.10
or 1.11. Track progress at: https://github.com/EnzymeAD/Enzyme.jl/issues/2699

For Julia 1.10/1.11, the extension imports ChainRulesCore rrules for:
- Operator evaluation: `op(x)` and `_eval_op(op, x)`
- Interpolator evaluation
- Basis functions: PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian

## Performance Note

`@import_rrule` has overhead compared to native Enzyme rules. For performance-critical
applications, consider using Zygote.jl which directly uses the ChainRulesCore rules.
"""
module RadialBasisFunctionsEnzymeExt

using RadialBasisFunctions
using ChainRulesCore
using Enzyme

# Import internal functions
import RadialBasisFunctions: _eval_op, RadialBasisOperator, Interpolator
import RadialBasisFunctions: PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: VectorValuedOperator
import RadialBasisFunctions: _build_weights, Partial, Laplacian

# =============================================================================
# Import ChainRulesCore rrules into Enzyme using @import_rrule
# =============================================================================

# Operator call syntax: op(x)
# These handle the direct operator call, bypassing cache check issues
Enzyme.@import_rrule(RadialBasisOperator, Vector{Float64})
Enzyme.@import_rrule(RadialBasisOperator{<:VectorValuedOperator}, Vector{Float64})

# Operator evaluation: _eval_op(op, x)
Enzyme.@import_rrule(typeof(_eval_op), RadialBasisOperator, Vector{Float64})
Enzyme.@import_rrule(
    typeof(_eval_op), RadialBasisOperator{<:VectorValuedOperator}, Vector{Float64}
)

# Basis function rules for common types (Float64 vectors)
Enzyme.@import_rrule(PHS1, Vector{Float64}, Vector{Float64})
Enzyme.@import_rrule(PHS3, Vector{Float64}, Vector{Float64})
Enzyme.@import_rrule(PHS5, Vector{Float64}, Vector{Float64})
Enzyme.@import_rrule(PHS7, Vector{Float64}, Vector{Float64})
Enzyme.@import_rrule(IMQ, Vector{Float64}, Vector{Float64})
Enzyme.@import_rrule(Gaussian, Vector{Float64}, Vector{Float64})

# Interpolator rules
Enzyme.@import_rrule(Interpolator, Vector{Float64})

# _build_weights rules for shape optimization
# Partial operator with different PHS types
Enzyme.@import_rrule(
    typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS1
)
Enzyme.@import_rrule(
    typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS3
)
Enzyme.@import_rrule(
    typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS5
)
Enzyme.@import_rrule(
    typeof(_build_weights), Partial, AbstractVector, AbstractVector, AbstractVector, PHS7
)

# Laplacian operator with different PHS types
Enzyme.@import_rrule(
    typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS1
)
Enzyme.@import_rrule(
    typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS3
)
Enzyme.@import_rrule(
    typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS5
)
Enzyme.@import_rrule(
    typeof(_build_weights), Laplacian, AbstractVector, AbstractVector, AbstractVector, PHS7
)

end # module
