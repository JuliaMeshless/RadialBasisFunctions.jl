"""
    struct Gaussian{T,D<:Int,M<:Metric} <: AbstractRadialBasis{M}

Gaussian radial basis function: `ϕ(r) = exp(-(ε·r)²)`

# Fields
- `ε::Real`: Shape parameter (ε > 0)
- `poly_deg::Int`: Degree of polynomial augmentation
- `metric::Metric`: Distance metric (default: Euclidean)
"""
struct Gaussian{T,D<:Int,M<:Metric} <: AbstractRadialBasis{M}
    ε::T
    poly_deg::D
    metric::M
    function Gaussian(ε::T=1; poly_deg::D=2, metric::M=Euclidean()) where {T,D<:Int,M<:Metric}
        if all(ε .< 0)
            throw(ArgumentError("Shape parameter should be > 0. (ε=$ε)"))
        end
        return new{T,D,M}(ε, poly_deg, metric)
    end
end

(rbf::Gaussian)(x, xᵢ) = exp(-(rbf.ε * rbf.metric(x, xᵢ))^2)

# Analytical derivative for Euclidean metric (fast path)
function ∂(rbf::Gaussian{T,D,Euclidean}, dim::Int) where {T,D}
    function ∂ℒ(x, xᵢ)
        return -2 * rbf.ε^2 * (x[dim] - xᵢ[dim]) * exp(-rbf.ε^2 * evaluate(SqEuclidean(), x, xᵢ))
    end
    return ∂ℒ
end

# AD-based derivative for non-Euclidean metrics
function ∂(rbf::Gaussian{T,D,M}, dim::Int) where {T,D,M<:Metric}
    function ∂ℒ(x, xᵢ)
        return ForwardDiff.derivative(t -> rbf(x + t * _unit_vector(x, dim), xᵢ), 0.0)
    end
    return ∂ℒ
end

# Analytical gradient for Euclidean metric
function ∇(rbf::Gaussian{T,D,Euclidean}) where {T,D}
    function ∇ℒ(x, xᵢ)
        return -2 * rbf.ε^2 * (x .- xᵢ) * exp(-rbf.ε^2 * evaluate(SqEuclidean(), x, xᵢ))
    end
    return ∇ℒ
end

# AD-based gradient for non-Euclidean metrics
function ∇(rbf::Gaussian{T,D,M}) where {T,D,M<:Metric}
    function ∇ℒ(x, xᵢ)
        return ForwardDiff.gradient(ξ -> rbf(ξ, xᵢ), x)
    end
    return ∇ℒ
end

# Analytical second partial derivative for Euclidean metric
function ∂²(rbf::Gaussian{T,D,Euclidean}, dim::Int) where {T,D}
    function ∂²ℒ(x, xᵢ)
        ε2 = rbf.ε^2
        return (4 * ε2^2 * (x[dim] - xᵢ[dim])^2 - 2 * ε2) * exp(-ε2 * evaluate(SqEuclidean(), x, xᵢ))
    end
    return ∂²ℒ
end

# AD-based second partial derivative for non-Euclidean metrics
function ∂²(rbf::Gaussian{T,D,M}, dim::Int) where {T,D,M<:Metric}
    function ∂²ℒ(x, xᵢ)
        hess = ForwardDiff.hessian(ξ -> rbf(ξ, xᵢ), x)
        return hess[dim, dim]
    end
    return ∂²ℒ
end

# Analytical Laplacian for Euclidean metric
function ∇²(rbf::Gaussian{T,D,Euclidean}) where {T,D}
    function ∇²ℒ(x, xᵢ)
        ε2 = rbf.ε^2
        return sum((4 * ε2^2 * (x .- xᵢ) .^ 2 .- 2 * ε2) * exp(-ε2 * evaluate(SqEuclidean(), x, xᵢ)))
    end
    return ∇²ℒ
end

# AD-based Laplacian for non-Euclidean metrics
function ∇²(rbf::Gaussian{T,D,M}) where {T,D,M<:Metric}
    function ∇²ℒ(x, xᵢ)
        hess = ForwardDiff.hessian(ξ -> rbf(ξ, xᵢ), x)
        return LinearAlgebra.tr(hess)
    end
    return ∇²ℒ
end

# Helper function for AD (defined in polyharmonic_spline.jl but needed here too)
# Note: This should be moved to basis.jl to avoid duplication
function _unit_vector(x::AbstractVector, dim::Int)
    e = zero(x)
    e = setindex(e, one(eltype(x)), dim)
    return e
end

function Base.show(io::IO, rbf::Gaussian)
    print(io, "Gaussian, exp(-(ε*r)²)")
    print(io, "\n├─Shape factor: ε = $(rbf.ε)")
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end

print_basis(rbf::Gaussian) = "Gaussian (ε = $(rbf.ε))"
