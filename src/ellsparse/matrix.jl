# AbstractMatrix interface — logical shape is (m, n)
Base.size(A::SellMatrix) = (A.structure.m, A.structure.n)
Base.IndexStyle(::Type{<:SellMatrix}) = IndexCartesian()

# Duplicate column indices within a row sum on read, matching the COO/CSC conversion
# semantics (sparse() combines duplicates by +).
function Base.getindex(A::SellMatrix{T, C}, i::Int, j::Int) where {T, C}
    @boundscheck checkbounds(A, i, j)
    S = A.structure
    base, w = _row_base(S.sliceptr, i, C)
    acc = zero(T)
    @inbounds for l in 1:w
        q = base + (l - 1) * C
        if Int(S.colind[q]) == j
            acc += A.vals[q]
        end
    end
    return acc
end

function Base.setindex!(::SellMatrix, v, i::Int, j::Int)
    throw(
        ArgumentError(
            "SellMatrix has a fixed sparsity structure; mutate `parent(A)` to change stored values, or convert with `sparse(A)` for general sparse editing."
        )
    )
end
