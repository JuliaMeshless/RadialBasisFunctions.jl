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

# ∂ - first partial derivative
function (op::∂{<:IMQ})(x, xᵢ)
    ε2 = op.basis.ε^2
    return (xᵢ[op.dim] - x[op.dim]) * (ε2 / sqrt((ε2 * sqeuclidean(x, xᵢ) + 1)^3))
end

# ∇ - gradient
function (op::∇{<:IMQ})(x, xᵢ)
    ε2 = op.basis.ε^2
    return (xᵢ .- x) * (ε2 / sqrt((ε2 * sqeuclidean(x, xᵢ) + 1)^3))
end

# D - directional derivative
function (op::D{<:IMQ})(x, xᵢ)
    ε2 = op.basis.ε^2
    return LinearAlgebra.dot(op.v, xᵢ .- x) * (ε2 / sqrt((ε2 * sqeuclidean(x, xᵢ) + 1)^3))
end

# D² - directional second derivative
function (op::D²{<:IMQ})(x, xᵢ)
    ε2 = op.basis.ε^2
    ε4 = ε2^2
    Δ = x .- xᵢ
    s = ε2 * sqeuclidean(x, xᵢ) + 1
    dot_v1_v2 = LinearAlgebra.dot(op.v1, op.v2)
    dot_v1_r = LinearAlgebra.dot(op.v1, Δ)
    dot_v2_r = LinearAlgebra.dot(op.v2, Δ)
    return 3 * ε4 * dot_v1_r * dot_v2_r / sqrt(s^5) - ε2 * dot_v1_v2 / sqrt(s^3)
end

# H - Hessian matrix
function (op::H{<:IMQ})(x, xᵢ)
    ε2 = op.basis.ε^2
    ε4 = ε2^2
    Δ = x .- xᵢ
    s = ε2 * sqeuclidean(x, xᵢ) + 1
    s3 = sqrt(s^3)
    s5 = sqrt(s^5)
    N = length(x)
    T = eltype(x)
    # H[i,j] = -ε² * δᵢⱼ / s^(3/2) + 3ε⁴ * Δᵢ*Δⱼ / s^(5/2)
    return StaticArraysCore.SMatrix{N,N,T}(
        ntuple(N * N) do k
            i, j = divrem(k - 1, N) .+ 1
            3 * ε4 * Δ[i] * Δ[j] / s5 - ε2 * T(i == j) / s3
        end
    )
end

# ∂² - second partial derivative
function (op::∂²{<:IMQ})(x, xᵢ)
    ε2 = op.basis.ε^2
    ε4 = ε2^2
    num1 = 3 * ε4 * (x[op.dim] - xᵢ[op.dim])^2
    denom = (ε2 * sqeuclidean(x, xᵢ) + 1)
    return num1 / sqrt(denom^5) - ε2 / sqrt(denom^3)
end

# ∇² - Laplacian
function (op::∇²{<:IMQ})(x, xᵢ)
    ε2 = op.basis.ε^2
    ε4 = ε2^2
    num1 = 3 * ε4 * (x .- xᵢ) .^ 2
    denom = (ε2 * sqeuclidean(x, xᵢ) + 1)
    return sum(num1 / sqrt(denom^5) .- ε2 / sqrt(denom^3))
end

function Base.show(io::IO, rbf::IMQ)
    print(io, "Inverse Multiquadrics, 1/sqrt((r*ε)²+1)")
    print(io, "\n├─Shape factor: ε = $(rbf.ε)")
    print(io, "\n└─Polynomial augmentation: degree $(rbf.poly_deg)")
    return nothing
end

print_basis(rbf::IMQ) = "Inverse Multiquadric (ε = $(rbf.ε))"
