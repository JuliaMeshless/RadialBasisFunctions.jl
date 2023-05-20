"""
    Laplacian <: ScalarValuedOperator

Builds an operator for the sum of the second derivatives w.r.t. each independent variable.
"""
struct Laplacian{L<:Function} <: ScalarValuedOperator
    ℒ::L
end

# convienience constructors
function laplacian(
    data::AbstractVector{D}, basis::B=PHS(3; poly_deg=2); k::T=autoselect_k(data, basis)
) where {D<:AbstractArray,T<:Int,B<:AbstractRadialBasis}
    ℒ = Laplacian(∇²)
    return RadialBasisOperator(ℒ, data, basis; k=k)
end

# pretty printing
print_op(op::Laplacian) = "Laplacian (∇² or Δ)"
