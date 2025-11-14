"""
    abstract type AbstractBasis{M<:Metric} end

Abstract supertype for all basis functions with distance metric type parameter `M`.
"""
abstract type AbstractBasis{M<:Metric} end

"""
    abstract type AbstractRadialBasis{M<:Metric} <: AbstractBasis{M} end

Abstract supertype for radial basis functions with distance metric type parameter `M`.
Radial basis functions depend only on the distance between points, as measured by metric `M`.
"""
abstract type AbstractRadialBasis{M<:Metric} <: AbstractBasis{M} end

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
