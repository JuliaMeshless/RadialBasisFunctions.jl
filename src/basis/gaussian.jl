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

# ∂ - first partial derivative
function (op::∂{<:Gaussian})(x, xᵢ)
    ε2 = op.basis.ε^2
    return -2 * ε2 * (x[op.dim] - xᵢ[op.dim]) * exp(-ε2 * sqeuclidean(x, xᵢ))
end

# ∇ - gradient
function (op::∇{<:Gaussian})(x, xᵢ)
    ε2 = op.basis.ε^2
    return -2 * ε2 * (x .- xᵢ) * exp(-ε2 * sqeuclidean(x, xᵢ))
end

# D - directional derivative
function (op::D{<:Gaussian})(x, xᵢ)
    ε2 = op.basis.ε^2
    return -2 * ε2 * LinearAlgebra.dot(op.v, x .- xᵢ) * exp(-ε2 * sqeuclidean(x, xᵢ))
end

# ∂² - second partial derivative
function (op::∂²{<:Gaussian})(x, xᵢ)
    ε2 = op.basis.ε^2
    return (4 * ε2^2 * (x[op.dim] - xᵢ[op.dim])^2 - 2 * ε2) * exp(-ε2 * sqeuclidean(x, xᵢ))
end

# ∇² - Laplacian
function (op::∇²{<:Gaussian})(x, xᵢ)
    ε2 = op.basis.ε^2
    return sum((4 * ε2^2 * (x .- xᵢ) .^ 2 .- 2 * ε2) * exp(-ε2 * sqeuclidean(x, xᵢ)))
end

function Base.show(io::IO, rbf::Gaussian)
    print(io, "Gaussian, exp(-(ε*r)²)")
    print(io, "\n├─Shape factor: ε = $(rbf.ε)")
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end

print_basis(rbf::Gaussian) = "Gaussian (ε = $(rbf.ε))"
