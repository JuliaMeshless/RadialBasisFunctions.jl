########################################################################################
# Inverse Multiquadrics
"""
    struct IMQ{T,D<:Int,M<:Metric} <: AbstractRadialBasis{M}

Inverse Multiquadric radial basis function: `ϕ(r) = 1/√((ε·r)² + 1)`

# Fields
- `ε::Real`: Shape parameter (ε > 0)
- `poly_deg::Int`: Degree of polynomial augmentation
- `metric::Metric`: Distance metric (default: Euclidean)
"""
struct IMQ{T,D<:Int,M<:Metric} <: AbstractRadialBasis{M}
    ε::T
    poly_deg::D
    metric::M
    function IMQ(ε::T=1; poly_deg::D=2, metric::M=Euclidean()) where {T,D<:Int,M<:Metric}
        if all(ε .< 0)
            throw(ArgumentError("Shape parameter should be > 0. (ε=$ε)"))
        end
        return new{T,D,M}(ε, poly_deg, metric)
    end
end

(rbf::IMQ)(x, xᵢ) = 1 / sqrt((rbf.metric(x, xᵢ) * rbf.ε)^2 + 1)

# Analytical derivative for Euclidean metric (fast path)
function ∂(rbf::IMQ{T,D,Euclidean}, dim::Int=1) where {T,D}
    function ∂ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        return (xᵢ[dim] - x[dim]) .* (ε2 / sqrt((ε2 * evaluate(SqEuclidean(), x, xᵢ) + 1)^3))
    end
    return ∂ℒ
end

# AD-based derivative for non-Euclidean metrics
function ∂(rbf::IMQ{T,D,M}, dim::Int=1) where {T,D,M<:Metric}
    function ∂ℒ(x, xᵢ)
        return ForwardDiff.derivative(t -> rbf(x + t * _unit_vector(x, dim), xᵢ), 0.0)
    end
    return ∂ℒ
end

# Analytical gradient for Euclidean metric
function ∇(rbf::IMQ{T,D,Euclidean}) where {T,D}
    function ∇ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        return (xᵢ - x) .* (ε2 / sqrt((ε2 * evaluate(SqEuclidean(), x, xᵢ) + 1)^3))
    end
    return ∇ℒ
end

# AD-based gradient for non-Euclidean metrics
function ∇(rbf::IMQ{T,D,M}) where {T,D,M<:Metric}
    function ∇ℒ(x, xᵢ)
        return ForwardDiff.gradient(ξ -> rbf(ξ, xᵢ), x)
    end
    return ∇ℒ
end

# Analytical second partial derivative for Euclidean metric
function ∂²(rbf::IMQ{T,D,Euclidean}, dim::Int=1) where {T,D}
    function ∂²ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        ε4 = ε2^2
        num1 = 3 * ε4 * (x[dim] - xᵢ[dim])^2
        denom = (ε2 * evaluate(SqEuclidean(), x, xᵢ) + 1)
        return num1 / sqrt(denom^5) - ε2 / sqrt(denom^3)
    end
    return ∂²ℒ
end

# AD-based second partial derivative for non-Euclidean metrics
function ∂²(rbf::IMQ{T,D,M}, dim::Int=1) where {T,D,M<:Metric}
    function ∂²ℒ(x, xᵢ)
        hess = ForwardDiff.hessian(ξ -> rbf(ξ, xᵢ), x)
        return hess[dim, dim]
    end
    return ∂²ℒ
end

# Analytical Laplacian for Euclidean metric
function ∇²(rbf::IMQ{T,D,Euclidean}) where {T,D}
    function ∇²ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        ε4 = ε2^2
        num1 = 3 * ε4 * (x .- xᵢ) .^ 2
        denom = (ε2 * evaluate(SqEuclidean(), x, xᵢ) + 1)
        return sum(num1 / sqrt(denom^5) .- ε2 / sqrt(denom^3))
    end
    return ∇²ℒ
end

# AD-based Laplacian for non-Euclidean metrics
function ∇²(rbf::IMQ{T,D,M}) where {T,D,M<:Metric}
    function ∇²ℒ(x, xᵢ)
        hess = ForwardDiff.hessian(ξ -> rbf(ξ, xᵢ), x)
        return LinearAlgebra.tr(hess)
    end
    return ∇²ℒ
end

function Base.show(io::IO, rbf::IMQ)
    print(io, "Inverse Multiquadrics, 1/sqrt((r*ε)²+1)")
    print(io, "\n├─Shape factor: ε = $(rbf.ε)")
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end

print_basis(rbf::IMQ) = "Inverse Multiquadric (ε = $(rbf.ε))"
