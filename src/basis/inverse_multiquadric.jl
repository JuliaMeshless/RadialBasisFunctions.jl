########################################################################################
# Inverse Multiquadrics
struct IMQ{T,D<:Int} <: AbstractRadialBasis
    ε::T
    poly_deg::D
    function IMQ(ε::T=1; poly_deg::D=2) where {T,D<:Int}
        if all(ε .< 0)
            throw(ArgumentError("Shape parameter should be > 0. (ε=$ε)"))
        end
        return new{T,D}(ε, poly_deg)
    end
end

(rbf::IMQ)(x, xᵢ) = 1 / sqrt((euclidean(x, xᵢ) * rbf.ε)^2 + 1)

function ∂(rbf::IMQ, dim::Int=1)
    function ∂ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        return (xᵢ[dim] - x[dim]) .* (ε2 / sqrt((ε2 * sqeuclidean(x, xᵢ) + 1)^3))
    end
    return ∂ℒ
end

function ∇(rbf::IMQ)
    function ∇ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        return (xᵢ - x) .* (ε2 / sqrt((ε2 * sqeuclidean(x, xᵢ) + 1)^3))
    end
    return ∇ℒ
end

function ∂²(rbf::IMQ, dim::Int=1)
    function ∂²ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        ε4 = ε2^2
        num1 = 3 * ε4 * (x[dim] - xᵢ[dim])^2
        denom = (ε2 * sqeuclidean(x, xᵢ) + 1)
        return num1 / sqrt(denom^5) - ε2 / sqrt(denom^3)
    end
    return ∂²ℒ
end

function ∇²(rbf::IMQ)
    function ∇²ℒ(x, xᵢ)
        ε2 = rbf.ε .^ 2
        ε4 = ε2^2
        num1 = 3 * ε4 * (x .- xᵢ) .^ 2
        denom = (ε2 * sqeuclidean(x, xᵢ) + 1)
        return sum(num1 / sqrt(denom^5) .- ε2 / sqrt(denom^3))
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
