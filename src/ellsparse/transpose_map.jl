# Copy a host vector to `backend` via bulk copyto! (no scalar indexing).
function _to_backend(backend, v::Vector)
    dv = KernelAbstractions.allocate(backend, eltype(v), length(v))
    copyto!(dv, v)
    return dv
end

"""
    build_transpose_map(Val(C), colind, sliceptr, m, n, width) -> TransposeMap

Build the column-wise [`TransposeMap`](@ref) of a SELL structure by counting sort:
one linear pass over the flat `colind` (skipping sentinels) counts entries per column,
a cumulative sum forms `offsets`, and a second linear pass in **ascending storage
position** fills `positions` — which is what makes adjoint summation order
deterministic. `rows` is filled alongside unless `C == 1` with uniform `width > 0`,
where the row is recoverable as `(q - 1) ÷ width + 1`.

The sort is a serial host pass: device inputs are copied to the host once and the
finished map is copied back, so the map lives uniformly on `colind`'s backend. Never
scalar-indexes device arrays.
"""
function build_transpose_map(
        ::Val{C}, colind::AbstractVector{Ti}, sliceptr::AbstractVector{Ti},
        m::Int, n::Int, width::Int,
    ) where {C, Ti <: Integer}
    backend = KernelAbstractions.get_backend(colind)
    on_host = backend isa CPU
    colind_h = on_host ? colind : Array(colind)
    sliceptr_h = on_host ? sliceptr : Array(sliceptr)
    counts = zeros(Ti, n + 1)
    @inbounds for q in 1:length(colind_h)
        j = colind_h[q]
        j == zero(Ti) && continue
        counts[j + 1] += one(Ti)
    end
    offsets = Vector{Ti}(undef, n + 1)
    offsets[1] = one(Ti)
    @inbounds for j in 1:n
        offsets[j + 1] = offsets[j] + counts[j + 1]
    end
    nnz_stored = Int(offsets[n + 1]) - 1
    positions = Vector{Ti}(undef, nnz_stored)
    store_rows = !(C == 1 && width > 0)
    rows = store_rows ? Vector{Ti}(undef, nnz_stored) : nothing
    cursor = copy(offsets)
    # Second pass in ascending q. Within slice s, q = sliceptr[s] + (l-1)*C + (r-1), so
    # ascending q is the (l, r) loop below; for C == 1 this is exactly linear iteration
    # of the k × m uniform layout — bit-compatible with a column-major counting sort
    # over that matrix.
    @inbounds for s in 1:cld(m, C)
        base = Int(sliceptr_h[s])
        w = (Int(sliceptr_h[s + 1]) - base) ÷ C
        q = base
        for l in 1:w, r in 1:C
            j = colind_h[q]
            if j != zero(Ti)
                p = Int(cursor[j])
                positions[p] = Ti(q)
                if rows !== nothing
                    rows[p] = Ti((s - 1) * C + r)
                end
                cursor[j] = Ti(p + 1)
            end
            q += 1
        end
    end
    on_host && return TransposeMap(offsets, positions, rows)
    return TransposeMap(
        _to_backend(backend, offsets), _to_backend(backend, positions),
        rows === nothing ? nothing : _to_backend(backend, rows),
    )
end
