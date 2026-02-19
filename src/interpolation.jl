"""
    struct Interpolator

Construct a radial basis interpolation.
"""
struct Interpolator{X, Y, R, M, RB, MB}
    x::X
    y::Y
    rbf_weights::R
    monomial_weights::M
    rbf_basis::RB
    monomial_basis::MB
end

"""
    function Interpolator(x, y, basis::B=PHS())

Construct a radial basis interpolator.

See also: [`regrid`](@ref) for local stencil-based interpolation between point sets.
"""
function Interpolator(x, y, basis::B = PHS()) where {B <: AbstractRadialBasis}
    x = _to_cpu(x)
    dim = length(first(x))
    k = length(x)  # number of data in influence/support domain
    npoly = binomial(dim + basis.poly_deg, basis.poly_deg)
    n = k + npoly
    mon = MonomialBasis(dim, basis.poly_deg)
    data_type = promote_type(eltype(first(x)), eltype(y))
    A = Symmetric(zeros(data_type, n, n))
    _build_collocation_matrix!(A, x, basis, mon, k)
    b = vcat(y, zeros(data_type, npoly))
    w = A \ b
    return Interpolator(x, y, w[1:k], w[(k + 1):end], basis, mon)
end

function (rbfi::Interpolator)(x::T) where {T}
    rbf = zero(eltype(T))
    @inbounds for i in eachindex(rbfi.rbf_weights)
        rbf += rbfi.rbf_weights[i] * rbfi.rbf_basis(x, rbfi.x[i])
    end

    poly = zero(eltype(T))
    if !isempty(rbfi.monomial_weights)
        poly = LinearAlgebra.dot(rbfi.monomial_weights, rbfi.monomial_basis(x))
    end
    return rbf + poly
end

(rbfi::Interpolator)(x::Vector{<:AbstractVector}) = [rbfi(val) for val in x]

# ============================================================================
# Adapt.jl support (GPU array conversion)
# ============================================================================

function Adapt.adapt_structure(to, interp::Interpolator)
    return Interpolator(
        Adapt.adapt(to, interp.x),
        Adapt.adapt(to, interp.y),
        Adapt.adapt(to, interp.rbf_weights),
        Adapt.adapt(to, interp.monomial_weights),
        interp.rbf_basis,
        interp.monomial_basis,
    )
end

# pretty printing
function Base.show(io::IO, op::Interpolator)
    println(io, "Interpolator")
    println(io, "├─Input type: ", typeof(first(op.x)))
    println(io, "├─Output type: ", typeof(first(op.y)))
    println(io, "├─Number of points: ", length(op.x))
    return print(
        io,
        "└─Basis: ",
        print_basis(op.rbf_basis),
        " with degree $(degree(op.monomial_basis)) Monomial",
    )
end
