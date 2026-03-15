module RadialBasisFunctionsLuxCoreExt

using RadialBasisFunctions: AbstractRadialBasis, AbstractPHS, Gaussian, IMQ,
    PHS1, PHS3, PHS5, PHS7
using LuxCore: AbstractLuxLayer, LuxCore
using Random: AbstractRNG

# --- Struct ---

"""
    RBFLayer(in_dims, num_centers, out_dims; kwargs...)
    RBFLayer((in_dims => num_centers) => out_dims; kwargs...)

Radial Basis Function Network layer compatible with Lux.jl.

Requires LuxCore.jl (or Lux.jl) to be loaded.

Architecture: `x ∈ R^d → φᵢ(x) = rbf(εᵢ, ||x - μᵢ||) → y = W * Φ(x) + b`

# Arguments
- `in_dims::Int`: Input dimension
- `num_centers::Int`: Number of RBF centers (hidden neurons)
- `out_dims::Int`: Output dimension

# Keyword Arguments
- `basis_type::Type{<:AbstractRadialBasis}=Gaussian`: RBF type (e.g., `Gaussian`, `IMQ`, `PHS3`)
- `use_bias::Bool=true`: Include bias in output layer
- `learn_centers::Bool=true`: Make centers learnable parameters
- `learn_shape::Bool=true`: Make shape parameters learnable (only for `Gaussian` and `IMQ`)
- `init_centers`: Initialization function `(rng, dims...) -> Array` for centers
- `init_weight`: Initialization function `(rng, dims...) -> Array` for output weights
- `init_shape::Float64=1.0`: Initial value for shape parameter ε
"""
struct RBFLayer{B <: AbstractRadialBasis, F1, F2} <: AbstractLuxLayer
    in_dims::Int
    num_centers::Int
    out_dims::Int
    basis_type::Type{B}
    use_bias::Bool
    learn_centers::Bool
    learn_shape::Bool
    init_centers::F1
    init_weight::F2
    init_shape::Float64
end

_default_init_centers(rng, dims...) = randn(rng, Float32, dims...) .* 0.5f0
_default_init_weight(rng, dims...) =
    (randn(rng, Float32, dims...) .* Float32(sqrt(2.0f0 / sum(dims))))

function RBFLayer(
        in_dims::Int,
        num_centers::Int,
        out_dims::Int;
        basis_type::Type{<:AbstractRadialBasis} = Gaussian,
        use_bias::Bool = true,
        learn_centers::Bool = true,
        learn_shape::Bool = true,
        init_centers = _default_init_centers,
        init_weight = _default_init_weight,
        init_shape::Float64 = 1.0,
    )
    B = basis_type
    init_shape > 0 || throw(ArgumentError("init_shape must be > 0, got $init_shape"))
    in_dims > 0 || throw(ArgumentError("in_dims must be > 0, got $in_dims"))
    num_centers > 0 || throw(ArgumentError("num_centers must be > 0, got $num_centers"))
    out_dims > 0 || throw(ArgumentError("out_dims must be > 0, got $out_dims"))
    return RBFLayer{B, typeof(init_centers), typeof(init_weight)}(
        in_dims, num_centers, out_dims, B, use_bias, learn_centers, learn_shape,
        init_centers, init_weight, init_shape,
    )
end

function RBFLayer(mapping::Pair{Pair{Int, Int}, Int}; kwargs...)
    (in_dims, num_centers), out_dims = mapping
    return RBFLayer(in_dims, num_centers, out_dims; kwargs...)
end

_has_shape_parameter(::Type{<:Gaussian}) = true
_has_shape_parameter(::Type{<:IMQ}) = true
_has_shape_parameter(::Type{<:AbstractRadialBasis}) = false

function Base.show(io::IO, l::RBFLayer)
    print(io, "RBFLayer($(l.in_dims) => $(l.num_centers) => $(l.out_dims)")
    print(io, ", basis_type=$(l.basis_type)")
    l.use_bias || print(io, ", use_bias=false")
    l.learn_centers || print(io, ", learn_centers=false")
    _has_shape_parameter(l.basis_type) && !l.learn_shape && print(io, ", learn_shape=false")
    return print(io, ")")
end

# --- Initialization ---

function LuxCore.initialparameters(rng::AbstractRNG, l::RBFLayer)
    ps = (; weight = l.init_weight(rng, l.out_dims, l.num_centers))
    if l.learn_centers
        ps = (; ps..., centers = l.init_centers(rng, l.in_dims, l.num_centers))
    end
    if _has_shape_parameter(l.basis_type) && l.learn_shape
        ps = (;
            ps...,
            log_shape = fill(_inverse_softplus(Float32(l.init_shape)), l.num_centers),
        )
    end
    if l.use_bias
        ps = (; ps..., bias = zeros(Float32, l.out_dims))
    end
    return ps
end

function LuxCore.initialstates(rng::AbstractRNG, l::RBFLayer)
    st = (;)
    if !l.learn_centers
        st = (; st..., centers = l.init_centers(rng, l.in_dims, l.num_centers))
    end
    if _has_shape_parameter(l.basis_type) && !l.learn_shape
        st = (;
            st...,
            log_shape = fill(_inverse_softplus(Float32(l.init_shape)), l.num_centers),
        )
    end
    return st
end

LuxCore.setup(rng::AbstractRNG, l::RBFLayer) =
    (LuxCore.initialparameters(rng, l), LuxCore.initialstates(rng, l))

# --- Forward pass ---

function (l::RBFLayer)(x::AbstractMatrix, ps, st)
    centers = l.learn_centers ? ps.centers : st.centers
    log_shape = if _has_shape_parameter(l.basis_type)
        l.learn_shape ? ps.log_shape : st.log_shape
    else
        nothing
    end

    D2 = _pairwise_sq_euclidean(centers, x)
    Phi = _rbf_activate(l.basis_type, D2, log_shape)
    y = ps.weight * Phi
    if l.use_bias
        y = y .+ ps.bias
    end
    return y, st
end

function (l::RBFLayer)(x::AbstractVector, ps, st)
    y, st = l(reshape(x, :, 1), ps, st)
    return vec(y), st
end

# --- Pairwise squared Euclidean distance ---

function _pairwise_sq_euclidean(C::AbstractMatrix, X::AbstractMatrix)
    C_sq = sum(abs2, C; dims = 1)
    X_sq = sum(abs2, X; dims = 1)
    D2 = C' * X
    D2 .*= -2
    D2 .+= C_sq'
    D2 .+= X_sq
    return max.(D2, zero(eltype(D2)))
end

# --- Activation functions ---

function _rbf_activate(::Type{<:Gaussian}, D2::AbstractMatrix, log_shape)
    eps2 = _softplus.(log_shape) .^ 2
    return @. exp(-eps2 * D2)
end

function _rbf_activate(::Type{<:IMQ}, D2::AbstractMatrix, log_shape)
    eps2 = _softplus.(log_shape) .^ 2
    return @. 1 / sqrt(eps2 * D2 + 1)
end

_phs_eval(::Type{<:PHS1}, D2) = @. sqrt(D2)
_phs_eval(::Type{<:PHS3}, D2) = @. D2 * sqrt(D2)
_phs_eval(::Type{<:PHS5}, D2) = @. D2^2 * sqrt(D2)
_phs_eval(::Type{<:PHS7}, D2) = @. D2^3 * sqrt(D2)

function _rbf_activate(::Type{B}, D2::AbstractMatrix, ::Nothing) where {B <: AbstractPHS}
    return _phs_eval(B, D2 .+ eps(eltype(D2)))
end

# --- Utilities ---

_softplus(x) = x > 20 ? x : log1p(exp(x))
_inverse_softplus(y) = y > 20 ? y : log(expm1(y))

export RBFLayer

end
