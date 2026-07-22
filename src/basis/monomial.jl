"""
    struct MonomialBasis{Dim,Deg} <: AbstractBasis

`Dim` dimensional monomial basis of order `Deg`.
"""
struct MonomialBasis{Dim, Deg, F <: Function} <: AbstractBasis
    f::F
    function MonomialBasis(dim::T, deg::T) where {T <: Int}
        if deg < 0
            throw(ArgumentError("Monomial basis must have non-negative degree"))
        end
        f = _get_monomial_basis(Val(dim), Val(deg))
        return new{dim, deg, typeof(f)}(f)
    end
end

function (m::MonomialBasis{Dim, Deg})(x) where {Dim, Deg}
    b = ones(_get_underlying_type(x), binomial(Dim + Deg, Dim))
    m.f(b, x)
    return b
end
(m::MonomialBasis)(b, x) = m.f(b, x)

degree(::MonomialBasis{Dim, Deg}) where {Dim, Deg} = Deg
dim(::MonomialBasis{Dim, Deg}) where {Dim, Deg} = Dim

function _constant_monomial_basis(b, x)
    b[1] = one(_get_underlying_type(x))
    return nothing
end

# Degree-0 basis is the constant `1` in every dimension. Explicit `Val{1}` and generic
# `Val{Dim}` methods keep this unambiguous against the `Val{1}, Val{N}` method below.
_get_monomial_basis(::Val{1}, ::Val{0}) = _constant_monomial_basis
_get_monomial_basis(::Val{Dim}, ::Val{0}) where {Dim} = _constant_monomial_basis

function _get_monomial_basis(::Val{1}, ::Val{N}) where {N}
    function basis!(b, x)
        b[1] = one(_get_underlying_type(x))
        if N > 0
            for n in 1:N
                b[n + 1] = only(x)^n
            end
        end
        return nothing
    end
    basis!(b, x::AbstractVector) = basis!(b, x[1])
    return basis!
end

function _get_monomial_basis(::Val{2}, ::Val{1})
    return function basis!(b, x)
        b[1] = one(eltype(x))
        b[2] = x[1]
        b[3] = x[2]
        return nothing
    end
end

function _get_monomial_basis(::Val{2}, ::Val{2})
    return function basis!(b, x)
        b[1] = one(eltype(x))
        b[2] = x[1]
        b[3] = x[2]
        b[4] = x[1] * x[2]
        b[5] = x[1] * x[1]
        b[6] = x[2] * x[2]
        return nothing
    end
end

function _get_monomial_basis(::Val{3}, ::Val{1})
    return function basis!(b, x)
        b[1] = one(eltype(x))
        b[2] = x[1]
        b[3] = x[2]
        b[4] = x[3]
        return nothing
    end
end

function _get_monomial_basis(::Val{3}, ::Val{2})
    return function basis!(b, x)
        b[1] = one(eltype(x))
        b[2] = x[1]
        b[3] = x[2]
        b[4] = x[3]
        b[5] = x[1] * x[2]
        b[6] = x[1] * x[3]
        b[7] = x[2] * x[3]
        b[8] = x[1] * x[1]
        b[9] = x[2] * x[2]
        b[10] = x[3] * x[3]
        return nothing
    end
end

function _get_monomial_basis(::Val{Dim}, ::Val{Deg}) where {Dim, Deg}
    e = multiexponents(Dim + 1, Deg)
    e = map(i -> getindex.(e, i), 1:Dim)
    ids = map(j -> map(i -> findall(x -> x >= i, e[j]), 1:Deg), 1:Dim)
    return build_monomial_basis(ids)
end

# scalar loop: `b[ids[i][k]] *= x[i]` materializes a temporary array per index list
function _monomial_products!(b, ids, x)
    @inbounds for i in eachindex(ids)
        xᵢ = x[i]
        for id in ids[i], j in id
            b[j] *= xᵢ
        end
    end
    return nothing
end

function build_monomial_basis(ids::Vector{Vector{Vector{T}}}) where {T <: Int}
    function basis!(b::AbstractVector{B}, x::AbstractVector) where {B}
        b .= one(B)
        _monomial_products!(b, ids, x)
        return nothing
    end
    return basis!
end

function Base.show(io::IO, ::MonomialBasis{Dim, Deg}) where {Dim, Deg}
    return print(io, "MonomialBasis of degree $(Deg) in $(Dim) dimensions")
end
