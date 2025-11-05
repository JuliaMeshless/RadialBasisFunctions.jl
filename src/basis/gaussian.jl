"""
    struct Gaussian{T,D<:Int} <: AbstractRadialBasis

Gaussian radial basis function:``ϕ(r) = e^{-(ε r)^2}``
"""
struct Gaussian{T,D<:Int} <: AbstractRadialBasis
    ε::T
    poly_deg::D
    function Gaussian(ε::T=1; poly_deg::D=2) where {T,D<:Int}
        if all(ε .< 0)
            throw(ArgumentError("Shape parameter should be > 0. (ε=$ε)"))
        end
        return new{T,D}(ε, poly_deg)
    end
end

(rbf::Gaussian)(x, xᵢ) = exp(-(rbf.ε * euclidean(x, xᵢ))^2)

function ∂(rbf::Gaussian, dim::Int)
    function ∂ℒ(x, xᵢ)
        return -2 * rbf.ε^2 * (x[dim] - xᵢ[dim]) * exp(-rbf.ε^2 * sqeuclidean(x, xᵢ))
    end
    return ∂ℒ
end

function ∇(rbf::Gaussian)
    function ∇ℒ(x, xᵢ)
        return -2 * rbf.ε^2 * (x .- xᵢ) * exp(-rbf.ε^2 * sqeuclidean(x, xᵢ))
    end
    return ∇ℒ
end

function ∂²(rbf::Gaussian, dim::Int)
    function ∂²ℒ(x, xᵢ)
        ε2 = rbf.ε^2
        return (4 * ε2^2 * (x[dim] - xᵢ[dim])^2 - 2 * ε2) * exp(-ε2 * sqeuclidean(x, xᵢ))
    end
    return ∂²ℒ
end

function ∇²(rbf::Gaussian)
    function ∇²ℒ(x, xᵢ)
        ε2 = rbf.ε^2
        return sum((4 * ε2^2 * (x .- xᵢ) .^ 2 .- 2 * ε2) * exp(-ε2 * sqeuclidean(x, xᵢ)))
    end
    return ∇²ℒ
end

function Base.show(io::IO, rbf::Gaussian)
    return show_shape_basis(io, rbf, "Gaussian, exp(-(ε*r)²)")
end

print_basis(rbf::Gaussian) = "Gaussian (ε = $(rbf.ε))"
