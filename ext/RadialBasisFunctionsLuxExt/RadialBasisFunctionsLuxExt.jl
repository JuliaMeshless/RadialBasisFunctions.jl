module RadialBasisFunctionsLuxExt

using RadialBasisFunctions
using RadialBasisFunctions: RBFLayer, AbstractRadialBasis, AbstractPHS,
    PHS1, PHS3, PHS5, PHS7, Gaussian, IMQ,
    _has_shape_parameter, _default_init_centers, _default_init_weight
using LuxCore
using LuxCore: AbstractLuxLayer
using Random: AbstractRNG

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

end # module
