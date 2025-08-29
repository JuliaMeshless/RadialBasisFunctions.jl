"""
    struct Interpolator

Construct a radial basis interpolation.
"""
struct Interpolator{X,Y,R,M,RB,MB}
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
"""
function Interpolator(x, y, basis::B=PHS()) where {B<:AbstractRadialBasis}
    dim = length(first(x))
    k = length(x)  # number of data in influence/support domain
    npoly = binomial(dim + basis.poly_deg, basis.poly_deg)
    n = k + npoly
    mon = MonomialBasis(dim, basis.poly_deg)
    data_type = promote_type(eltype(first(x)), eltype(y))
    A = Symmetric(zeros(data_type, n, n))
    _build_collocation_matrix!(A, x, basis, mon, k, StandardStencil())
    b = vcat(y, zeros(data_type, npoly))
    w = A \ b
    return Interpolator(x, y, w[1:k], w[(k + 1):end], basis, mon)
end

function (rbfi::Interpolator)(x::T) where {T}
    rbf = zero(eltype(T))
    for i in eachindex(rbfi.rbf_weights)
        rbf += rbfi.rbf_weights[i] * rbfi.rbf_basis(x, rbfi.x[i])
    end

    poly = zero(eltype(T))
    if !isempty(rbfi.monomial_weights)
        val_poly = rbfi.monomial_basis(x)
        for (i, val) in enumerate(val_poly)
            poly += rbfi.monomial_weights[i] * val
        end
    end
    return rbf + poly
end

(rbfi::Interpolator)(x::Vector{<:AbstractVector}) = [rbfi(val) for val in x]

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
