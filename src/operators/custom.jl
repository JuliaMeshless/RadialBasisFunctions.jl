"""
    Custom <: ScalarValuedOperator

Builds an operator for a first order partial derivative.
"""
struct Custom{F<:Function} <: AbstractOperator
    ℒ::F
end
(op::Custom)(basis) = op.ℒ(basis)

# Hermite-compatible method now uses the generic dispatcher in solve_hermite.jl

# pretty printing
print_op(op::Custom) = "Custom Operator"
