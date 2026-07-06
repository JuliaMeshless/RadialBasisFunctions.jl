########################################################################################
# Inverse Multiquadrics

"""
    IMQ(ε=1; poly_deg=2)

Inverse Multiquadric radial basis function: `ϕ(r) = 1/√((εr)² + 1)`

# Arguments
- `ε`: Shape parameter (must be > 0). Smaller values = flatter basis.
- `poly_deg`: Polynomial augmentation degree (default: 2).

See also: [`PHS`](@ref), [`Gaussian`](@ref)
"""
struct IMQ{T, D <: Int} <: AbstractRadialBasis
    ε::T
    poly_deg::D
    function IMQ(ε::T = 1; poly_deg::D = 2) where {T, D <: Int}
        if ε <= 0
            throw(
                ArgumentError(
                    "Shape parameter ε must be > 0 (got ε=$ε). Typical values range from 0.1 to 10.0.",
                ),
            )
        end
        return new{T, D}(ε, poly_deg)
    end
end

(rbf::IMQ)(r2) = 1 / sqrt(rbf.ε^2 * r2 + 1)
(rbf::IMQ)(x, xᵢ) = rbf(sqeuclidean(x, xᵢ))

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
    return StaticArraysCore.SMatrix{N, N, T}(
        ntuple(N * N) do k
            i, j = divrem(k - 1, N) .+ 1
            3 * ε4 * Δ[i] * Δ[j] / s5 - ε2 * T(i == j) / s3
        end,
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
