"""
    Gaussian(ε=1; poly_deg=2)

Gaussian radial basis function: `ϕ(r) = exp(-(εr)²)`

# Arguments
- `ε`: Shape parameter (must be > 0). Smaller values = wider basis.
- `poly_deg`: Polynomial augmentation degree (default: 2).

See also: [`PHS`](@ref), [`IMQ`](@ref)
"""
struct Gaussian{T, D <: Int} <: AbstractRadialBasis
    ε::T
    poly_deg::D
    function Gaussian(ε::T = 1; poly_deg::D = 2) where {T, D <: Int}
        if ε <= 0
            throw(
                ArgumentError(
                    "Shape parameter ε must be > 0 (got ε=$ε). Typical values range from 0.1 to 10.0.",
                ),
            )
        end
        return new{T, D}(ε, poly_deg)
    end
    # AD tangent construction: ε holds a derivative (any sign), so no validation
    global _tangent_basis
    _tangent_basis(::Type{Gaussian{T, D}}, Δε) where {T, D <: Int} = new{T, D}(convert(T, Δε), zero(D))
end

(rbf::Gaussian)(r2) = exp(-rbf.ε^2 * r2)
(rbf::Gaussian)(x, xᵢ) = rbf(sqeuclidean(x, xᵢ))

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

# H - Hessian matrix
function (op::H{<:Gaussian})(x, xᵢ)
    ε2 = op.basis.ε^2
    φ = exp(-ε2 * sqeuclidean(x, xᵢ))
    Δ = x .- xᵢ
    N = length(x)
    T = eltype(x)
    # H[i,j] = (4ε⁴ * Δᵢ*Δⱼ - 2ε² * δᵢⱼ) * φ
    return StaticArraysCore.SMatrix{N, N, T}(
        ntuple(N * N) do k
            i, j = divrem(k - 1, N) .+ 1
            (4 * ε2^2 * Δ[i] * Δ[j] - 2 * ε2 * T(i == j)) * φ
        end,
    )
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
