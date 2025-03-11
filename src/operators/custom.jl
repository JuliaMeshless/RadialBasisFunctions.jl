"""
    Custom <: ScalarValuedOperator

Builds an operator for a first order partial derivative.
"""
struct Custom{F<:Function} <: AbstractOperator
    ℒ::F
end
(op::Custom)(basis) = op.ℒ(basis)
