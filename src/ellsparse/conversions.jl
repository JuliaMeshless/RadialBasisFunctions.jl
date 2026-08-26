# ------------------------------------------------------------------------------------------
# Conversions. Sentinel slots are skipped; duplicate column indices within a row are
# summed (sparse() combines by +, Matrix/getindex accumulate), so identity-style rows
# padded with repeated indices collapse exactly. Device matrices route through a host
# copy — conversion targets are host formats.
# ------------------------------------------------------------------------------------------

"""
    coo(A::SellMatrix) -> (I, J, V)

The stored (non-sentinel) entries of `A` as COO triplets, in row order with each row's
slots ascending. Explicit stored zeros are retained; duplicates are **not** combined
here — `sparse(A)` sums them.
"""
function coo(A::SellMatrix{T, C}) where {T, C}
    backend = KernelAbstractions.get_backend(A)
    backend isa CPU || return coo(Adapt.adapt(Array, A))
    S = A.structure
    Ti = eltype(S.colind)
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, length(S.colind))
    sizehint!(J, length(S.colind))
    sizehint!(V, length(S.colind))
    @inbounds for i in 1:S.m
        base, w = _row_base(S.sliceptr, i, C)
        for l in 1:w
            q = base + (l - 1) * C
            j = S.colind[q]
            j == zero(Ti) && continue
            push!(I, i)
            push!(J, Int(j))
            push!(V, A.vals[q])
        end
    end
    return I, J, V
end

function SparseArrays.sparse(A::SellMatrix)
    I, J, V = coo(A)
    m, n = size(A)
    # sparse() sums duplicates, so rows padded with a repeated index collapse to single
    # entries; explicit zeros elsewhere are retained.
    return sparse(I, J, V, m, n)
end

SparseArrays.SparseMatrixCSC(A::SellMatrix) = sparse(A)

function Base.Matrix(A::SellMatrix{T, C}) where {T, C}
    backend = KernelAbstractions.get_backend(A)
    backend isa CPU || return Matrix(Adapt.adapt(Array, A))
    S = A.structure
    Ti = eltype(S.colind)
    M = zeros(T, size(A))
    @inbounds for i in 1:S.m
        base, w = _row_base(S.sliceptr, i, C)
        for l in 1:w
            q = base + (l - 1) * C
            j = S.colind[q]
            j == zero(Ti) && continue
            M[i, Int(j)] += A.vals[q]
        end
    end
    return M
end
