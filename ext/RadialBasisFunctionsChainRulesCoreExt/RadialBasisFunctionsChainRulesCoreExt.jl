"""
    RadialBasisFunctionsChainRulesCoreExt

Package extension providing ChainRulesCore.jl custom differentiation rules for
RadialBasisFunctions.jl. These rules enable efficient automatic differentiation
with backends like Zygote.jl, Enzyme.jl (via @import_rrule), and others that
support ChainRulesCore.

The rules leverage the analytical derivatives already implemented in the package
(∂, ∇, ∇² methods) rather than relying on AD to trace through the computations.
"""
module RadialBasisFunctionsChainRulesCoreExt

using RadialBasisFunctions
using ChainRulesCore
using LinearAlgebra
using SparseArrays

# Import internal functions we need to extend
import RadialBasisFunctions: _eval_op

# Import types we need
import RadialBasisFunctions: RadialBasisOperator, Interpolator
import RadialBasisFunctions: AbstractRadialBasis, PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: VectorValuedOperator, ScalarValuedOperator
import RadialBasisFunctions: MonomialBasis

# Import the gradient function for basis functions (not exported from main module)
const ∇ = RadialBasisFunctions.∇
const ∂ = RadialBasisFunctions.∂

include("operator_rules.jl")
include("basis_rules.jl")
include("interpolation_rules.jl")

end # module
