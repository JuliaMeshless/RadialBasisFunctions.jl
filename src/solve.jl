function _build_weights(ℒ, op)
    data = op.data
    eval_points = op.eval_points
    adjl = op.adjl
    basis = op.basis
    return _build_weights(ℒ, data, eval_points, adjl, basis)
end

function _build_weights(ℒ, data, eval_points, adjl, basis)
    dim = length(first(data)) # dimension of data

    # build monomial basis and operator
    mon = MonomialBasis(dim, basis.poly_deg)
    ℒmon = ℒ(mon)
    ℒrbf = ℒ(basis)

    return _build_weights(data, eval_points, adjl, basis, ℒrbf, ℒmon, mon)
end

function _build_weights(
    data, eval_points, adjl, basis, ℒrbf, ℒmon, mon; batch_size=10, device=CPU()
)
    # Use the unified kernel infrastructure with standard allocation strategy
    return _build_weights_unified(
        StandardAllocation(),
        data,
        eval_points,
        adjl,
        basis,
        ℒrbf,
        ℒmon,
        mon,
        nothing;
        batch_size=batch_size,
        device=device,
    )
end

function _build_stencil!(
    A::Symmetric,
    b,
    ℒrbf,
    ℒmon,
    data::AbstractVector{TD},
    eval_point::TE,
    basis::B,
    mon::MonomialBasis,
    k::Int,
) where {TD,TE,B<:AbstractRadialBasis}
    _build_collocation_matrix!(A, data, basis, mon, k)
    _build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)
    return (A \ b)[1:k, :]
end

function _build_collocation_matrix!(
    A::Symmetric, data::AbstractVector, basis::B, mon::MonomialBasis{Dim,Deg}, k::K
) where {B<:AbstractRadialBasis,K<:Int,Dim,Deg}
    # radial basis section
    AA = parent(A)
    N = size(A, 2)
    @inbounds for j in 1:k, i in 1:j
        AA[i, j] = basis(data[i], data[j])
    end

    # monomial augmentation
    if Deg > -1
        @inbounds for i in 1:k
            a = view(AA, i, (k + 1):N)
            mon(a, data[i])
        end
    end

    return nothing
end

function _build_rhs!(
    b, ℒrbf, ℒmon, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    # radial basis section
    @inbounds for i in eachindex(data)
        b[i] = ℒrbf(eval_point, data[i])
    end

    # monomial augmentation
    if basis.poly_deg > -1
        N = length(b)
        bmono = view(b, (k + 1):N)
        ℒmon(bmono, eval_point)
    end

    return nothing
end

function _build_rhs!(
    b, ℒrbf::Tuple, ℒmon::Tuple, data::AbstractVector{TD}, eval_point::TE, basis::B, k::K
) where {TD,TE,B<:AbstractRadialBasis,K<:Int}
    @assert size(b, 2) == length(ℒrbf) == length(ℒmon) "b, ℒrbf, ℒmon must have the same length"
    # radial basis section
    for (j, ℒ) in enumerate(ℒrbf)
        @inbounds for i in eachindex(data)
            b[i, j] = ℒ(eval_point, data[i])
        end
    end

    # monomial augmentation
    if basis.poly_deg > -1
        N = size(b, 1)
        for (j, ℒ) in enumerate(ℒmon)
            bmono = view(b, (k + 1):N, j)
            ℒ(bmono, eval_point)
        end
    end

    return nothing
end
