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

"""
    ∂{B<:AbstractRadialBasis}

Partial derivative operator functor. Construct with `∂(basis, dim)`.
"""
struct ∂{B<:AbstractRadialBasis}
    basis::B
    dim::Int
end

"""
    ∇{B<:AbstractRadialBasis}

Gradient operator functor. Construct with `∇(basis)`.
"""
struct ∇{B<:AbstractRadialBasis}
    basis::B
end

"""
    ∂²{B<:AbstractRadialBasis}

Second partial derivative operator functor. Construct with `∂²(basis, dim)`.
"""
struct ∂²{B<:AbstractRadialBasis}
    basis::B
    dim::Int
end

"""
    ∇²{B<:AbstractRadialBasis}

Laplacian operator functor. Construct with `∇²(basis)`.
"""
struct ∇²{B<:AbstractRadialBasis}
    basis::B
end

"""
    D{B<:AbstractRadialBasis,V}

Directional derivative operator functor. Construct with `D(basis, v)`.
Computes the derivative of the basis function in direction `v`.
"""
struct D{B<:AbstractRadialBasis,V}
    basis::B
    v::V
end

"""
    D²{B<:AbstractRadialBasis,V1,V2}

Directional second derivative operator functor. Construct with `D²(basis, v1, v2)`.
"""
struct D²{B<:AbstractRadialBasis,V1,V2}
    basis::B
    v1::V1
    v2::V2
end

include("polyharmonic_spline.jl")
include("inverse_multiquadric.jl")
include("gaussian.jl")
include("monomial.jl")

# pretty printing
unicode_order(::Val{1}) = ""
unicode_order(::Val{2}) = "²"
unicode_order(::Val{3}) = "³"
unicode_order(::Val{4}) = "⁴"
unicode_order(::Val{5}) = "⁵"
unicode_order(::Val{6}) = "⁶"
unicode_order(::Val{7}) = "⁷"
unicode_order(::Val{8}) = "⁸"
unicode_order(::Val{9}) = "⁹"
