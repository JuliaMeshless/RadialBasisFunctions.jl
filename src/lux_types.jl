"""
    RBFLayer(in_dims, num_centers, out_dims; kwargs...)
    RBFLayer((in_dims => num_centers) => out_dims; kwargs...)

Radial Basis Function Network layer compatible with Lux.jl.

Requires LuxCore.jl to be loaded for full functionality (parameter initialization,
forward pass). Without LuxCore, only the struct and constructors are available.

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
