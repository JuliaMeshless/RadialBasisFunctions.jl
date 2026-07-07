"""
    abstract type AbstractBasis end
"""
abstract type AbstractBasis end

"""
    abstract type AbstractRadialBasis <: AbstractBasis end
"""
abstract type AbstractRadialBasis <: AbstractBasis end

# Operator functor types - callable structs for RBF differential operators
# These replace closures with proper multiple dispatch
#
# Two basis-differentiation protocols share the names ∂/∇/∂²/∇²/H/∂mixed (load-bearing,
# do not unify): for AbstractRadialBasis they are the functor STRUCTS below, evaluated
# as (x, xᵢ) -> scalar; for MonomialBasis they are factory FUNCTIONS returning an
# ℒMonomialBasis that fills the differentiated monomial vector in place (see
# operators/monomial/monomial.jl). Operator actions such as (op::Partial)(basis) are
# written once and serve both.

"""
    ∂{B<:AbstractRadialBasis}

Partial derivative operator functor. Construct with `∂(basis, dim)`.
"""
struct ∂{B <: AbstractRadialBasis}
    basis::B
    dim::Int
end

"""
    ∇{B<:AbstractRadialBasis}

Gradient operator functor. Construct with `∇(basis)`.
"""
struct ∇{B <: AbstractRadialBasis}
    basis::B
end

"""
    ∂²{B<:AbstractRadialBasis}

Second partial derivative operator functor. Construct with `∂²(basis, dim)`.
"""
struct ∂²{B <: AbstractRadialBasis}
    basis::B
    dim::Int
end

"""
    ∇²{B<:AbstractRadialBasis}

Laplacian operator functor. Construct with `∇²(basis)`.
"""
struct ∇²{B <: AbstractRadialBasis}
    basis::B
end

"""
    H{B<:AbstractRadialBasis}

Hessian operator functor. Construct with `H(basis)`.
Returns the Hessian matrix of the basis function.
"""
struct H{B <: AbstractRadialBasis}
    basis::B
end

"""
    ∂mixed{B<:AbstractRadialBasis}

Mixed partial derivative operator functor. Construct with `∂mixed(basis, dim1, dim2)`.
Computes ∂²φ/(∂x_{dim1} ∂x_{dim2}).
"""
struct ∂mixed{B <: AbstractRadialBasis}
    basis::B
    dim1::Int
    dim2::Int
end

include("polyharmonic_spline.jl")
include("inverse_multiquadric.jl")
include("gaussian.jl")
include("monomial.jl")

# Generic ∂mixed implementation via existing basis operators
function (op::∂mixed)(x, xᵢ)
    if op.dim1 == op.dim2
        return ∂²(op.basis, op.dim1)(x, xᵢ)
    end
    return H(op.basis)(x, xᵢ)[op.dim1, op.dim2]
end
