"""
    RadialBasisFunctionsChainRulesCoreExt

Package extension providing ChainRulesCore.jl custom differentiation rules for
RadialBasisFunctions.jl. These rules enable efficient automatic differentiation
with backends like Zygote.jl, Enzyme.jl (via @import_rrule), and others that
support ChainRulesCore.

The rules leverage the analytical derivatives already implemented in the package
(∂, ∇, ∇² methods) rather than relying on AD to trace through the computations.

Includes rrules for:
- Basis function evaluation (basis_rules.jl)
- Operator application (operator_rules.jl)
- Interpolator evaluation (interpolation_rules.jl)
- Weight construction for shape optimization (build_weights_*.jl)
"""
module RadialBasisFunctionsChainRulesCoreExt

using RadialBasisFunctions
using ChainRulesCore
using LinearAlgebra
using SparseArrays
using Combinatorics: binomial

# Import internal functions we need to extend
import RadialBasisFunctions: _eval_op, _build_weights
import RadialBasisFunctions: _build_collocation_matrix!, _build_rhs!

# Import types we need
import RadialBasisFunctions: RadialBasisOperator, Interpolator
import RadialBasisFunctions: AbstractRadialBasis, PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: VectorValuedOperator, ScalarValuedOperator
import RadialBasisFunctions: MonomialBasis, BoundaryData
import RadialBasisFunctions: Partial, Laplacian

# Import the gradient function for basis functions (not exported from main module)
const ∇ = RadialBasisFunctions.∇
const ∂ = RadialBasisFunctions.∂

# Existing rules
include("operator_rules.jl")
include("basis_rules.jl")
include("interpolation_rules.jl")

# Shape optimization support: rrules for _build_weights
include("build_weights_cache.jl")
include("operator_second_derivatives.jl")
include("build_weights_backward.jl")
include("build_weights_rrule.jl")

end # module
