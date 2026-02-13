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
- Weight construction for shape optimization (build_weights_rrule.jl)
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
import RadialBasisFunctions: AbstractRadialBasis, PHS, PHS1, PHS3, PHS5, PHS7, IMQ, Gaussian
import RadialBasisFunctions: VectorValuedOperator, ScalarValuedOperator
import RadialBasisFunctions: MonomialBasis, BoundaryData
import RadialBasisFunctions: Partial, Laplacian

# Import backward pass support from main package
import RadialBasisFunctions: StencilForwardCache, WeightsBuildForwardCache
import RadialBasisFunctions: backward_linear_solve!, backward_collocation!
import RadialBasisFunctions: backward_rhs_partial!, backward_rhs_laplacian!
import RadialBasisFunctions: backward_stencil_partial!, backward_stencil_laplacian!
import RadialBasisFunctions: backward_stencil_partial_with_ε!, backward_stencil_laplacian_with_ε!
import RadialBasisFunctions: _forward_with_cache
import RadialBasisFunctions: grad_applied_partial_wrt_x, grad_applied_partial_wrt_xi
import RadialBasisFunctions: grad_applied_laplacian_wrt_x, grad_applied_laplacian_wrt_xi

# Import shape parameter derivative functions
import RadialBasisFunctions: ∂φ_∂ε, ∂Laplacian_φ_∂ε, ∂Partial_φ_∂ε

# Import the gradient function for basis functions (not exported from main module)
const ∇ = RadialBasisFunctions.∇
const ∂ = RadialBasisFunctions.∂

# Existing rules
include("operator_rules.jl")
include("basis_rules.jl")
include("interpolation_rules.jl")

# Shape optimization support: rrules for _build_weights
include("build_weights_rrule.jl")

end # module
