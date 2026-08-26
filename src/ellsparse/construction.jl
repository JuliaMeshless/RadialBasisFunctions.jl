# ------------------------------------------------------------------------------------------
# Validating constructors. The mul/adjoint kernels index with @inbounds, so everything
# they assume — matching backends, in-range column indices, index-type range — must be
# rejected here or bad input becomes undefined behavior instead of a clean throw.
# ------------------------------------------------------------------------------------------

"""
    SellMatrix(vals::AbstractMatrix, colind::AbstractMatrix{<:Integer}, n::Integer;
               transpose_map = true)

Zero-copy uniform `C == 1` construction from per-row matrices: `vals[l, i]` is the
`l`-th stored value of row `i` and `colind[l, i]` its column, both `k × m` (column
per logical row). The flat storage aliases both inputs (`parent(A) === vals`); every
slot must hold a real entry (`colind` values in `1:n`, no sentinels).

Set `transpose_map = false` to skip building the [`TransposeMap`](@ref); adjoint and
transpose products then throw until the matrix is rebuilt with a map.
"""
function SellMatrix(
        vals::AbstractMatrix, colind::AbstractMatrix{Ti}, n::Integer;
        transpose_map::Bool = true,
    ) where {Ti <: Integer}
    if size(vals) != size(colind)
        throw(DimensionMismatch("vals is $(size(vals)) but colind is $(size(colind))"))
    end
    backend = KernelAbstractions.get_backend(vals)
    if backend != KernelAbstractions.get_backend(colind)
        throw(
            ArgumentError(
                "vals and colind must live on the same backend; got $backend and $(KernelAbstractions.get_backend(colind))"
            )
        )
    end
    if n > typemax(Ti) || length(colind) >= typemax(Ti)
        throw(ArgumentError("matrix structure exceeds the $Ti index range"))
    end
    # Device-safe bounds check (extrema is a reduction, no scalar indexing).
    if !isempty(colind)
        lo, hi = extrema(colind)
        if lo < 1 || hi > n
            throw(ArgumentError("column indices must lie in 1:$n; got values in $lo:$hi"))
        end
    end
    k, m = size(colind)
    sliceptr = _uniform_sliceptr(backend, Ti, k, m)
    flat = vec(colind)
    tmap = transpose_map ?
        build_transpose_map(Val(1), flat, sliceptr, m, Int(n), k) : nothing
    S = SellStructure{1}(m, Int(n), flat, sliceptr, k, false, tmap)
    return SellMatrix(vals, S)
end

# sliceptr of the uniform C == 1 layout: row i starts at (i - 1) * k + 1
function _uniform_sliceptr(backend, ::Type{Ti}, k::Int, m::Int) where {Ti <: Integer}
    ptr = Vector{Ti}(undef, m + 1)
    @inbounds for s in 0:m
        ptr[s + 1] = Ti(1 + k * s)
    end
    backend isa CPU && return ptr
    return _to_backend(backend, ptr)
end

"""
    SellMatrix(S::SparseMatrixCSC, Val(C); pad = :slice, index_type = Int32,
               transpose_map = true)

Convert a CSC matrix to SELL-C. Row lengths are counted, each slice is padded to the
width of its longest row (`pad = :slice`) or to the global maximum (`pad = :global`),
and entries are filled in CSC column order so each row's column indices come out
ascending. Explicit stored zeros are retained. Host-serial; device conversion goes
through a host matrix.

See also [`sell`](@ref) for a runtime slice height.
"""
function SellMatrix(
        Sp::SparseMatrixCSC, ::Val{C}; pad::Symbol = :slice,
        index_type::Type{Ti} = Int32, transpose_map::Bool = true,
    ) where {C, Ti <: Integer}
    (C isa Int && C >= 1) ||
        throw(ArgumentError("slice height C must be a positive Int; got $C"))
    pad in (:slice, :global) ||
        throw(ArgumentError("pad must be :slice or :global; got $(repr(pad))"))
    m, n = size(Sp)
    rv = rowvals(Sp)
    nzv = nonzeros(Sp)
    rowlen = zeros(Int, m)
    @inbounds for p in 1:length(rv)
        rowlen[rv[p]] += 1
    end
    nsl = cld(m, C)
    widths = Vector{Int}(undef, nsl)
    if pad === :global
        fill!(widths, isempty(rowlen) ? 0 : maximum(rowlen))
    else
        @inbounds for s in 1:nsl
            w = 0
            for i in ((s - 1) * C + 1):min(s * C, m)
                w = max(w, rowlen[i])
            end
            widths[s] = w
        end
    end
    total = C * sum(widths; init = 0)
    if m > typemax(Ti) || n > typemax(Ti) || total >= typemax(Ti)
        throw(ArgumentError("matrix structure exceeds the $Ti index range"))
    end
    sliceptr = Vector{Ti}(undef, nsl + 1)
    sliceptr[1] = one(Ti)
    @inbounds for s in 1:nsl
        sliceptr[s + 1] = sliceptr[s] + Ti(C * widths[s])
    end
    colind = zeros(Ti, total)          # zeros(...) = sentinel-initialized
    vals = zeros(eltype(Sp), total)    # zero padding comes free
    cursors = zeros(Int, m)
    @inbounds for j in 1:n, p in nzrange(Sp, j)
        i = rv[p]
        l = (cursors[i] += 1)
        s = _slice_of(i, C)
        q = Int(sliceptr[s]) + (l - 1) * C + (i - 1) - (s - 1) * C
        colind[q] = Ti(j)
        vals[q] = nzv[p]
    end
    width = (nsl > 0 && all(==(widths[1]), widths)) ? widths[1] : 0
    padded = total != length(rv)
    tmap = transpose_map ?
        build_transpose_map(Val(C), colind, sliceptr, m, n, width) : nothing
    S = SellStructure{C}(m, n, colind, sliceptr, width, padded, tmap)
    return SellMatrix(vals, S)
end

"""
    sell(S::SparseMatrixCSC; slice_height = 1, kwargs...) -> SellMatrix

Runtime-slice-height entry to the CSC constructor, through a `Val` function barrier so
downstream code still specializes on `C`. Keyword arguments are forwarded to
`SellMatrix(S, Val(C); ...)`.
"""
sell(Sp::SparseMatrixCSC; slice_height::Integer = 1, kwargs...) =
    SellMatrix(Sp, Val(Int(slice_height)); kwargs...)

"""
    preferred_slice_height(backend) -> Val

The slice height a backend's row-parallel kernels want: `Val(1)` on CPU (row-contiguous
SIMD), `Val(32)` on device backends (warp-coalesced, the cuSPARSE SELL orientation).
The policy hook for layout decisions at data-movement boundaries; pair with
[`reslice`](@ref) — `Adapt.adapt` itself never changes layout.
"""
preferred_slice_height(::CPU) = Val(1)
preferred_slice_height(::KernelAbstractions.Backend) = Val(32)

"""
    reslice(A::SellMatrix, Val(C₂); pad = :slice) -> SellMatrix

Rebuild `A` with slice height `C₂` — the explicit layout-change operation (`adapt`
never changes layout). Stored entries keep their per-row slot order, and the transpose
map is **remapped in sequence order** rather than rebuilt, so adjoint products of the
resliced matrix keep the original summation order bit-for-bit (up to hardware FMA
differences across devices).

Device matrices round-trip through the host (the map remap is host-serial regardless);
the result lives on `A`'s backend.
"""
function reslice(
        A::SellMatrix{T, C1}, target::Val{C2}; pad::Symbol = :slice,
    ) where {T, C1, C2}
    (C2 isa Int && C2 >= 1) ||
        throw(ArgumentError("slice height C must be a positive Int; got $C2"))
    pad in (:slice, :global) ||
        throw(ArgumentError("pad must be :slice or :global; got $(repr(pad))"))
    backend = KernelAbstractions.get_backend(A)
    backend isa CPU || return _to_backend_matrix(backend, reslice(Adapt.adapt(Array, A), target; pad))
    S = A.structure
    m, n = S.m, S.n
    Ti = eltype(S.colind)
    colind1 = S.colind
    vals1 = A.vals
    # Per-row stored extent: the last non-sentinel slot (slots 1:len are copied
    # verbatim, so any interior sentinel a foreign structure might carry is preserved).
    rowlen = Vector{Int}(undef, m)
    nnz_stored = 0
    @inbounds for i in 1:m
        base, w = _row_base(S.sliceptr, i, C1)
        len = 0
        for l in 1:w
            if colind1[base + (l - 1) * C1] != zero(Ti)
                len = l
                nnz_stored += 1
            end
        end
        rowlen[i] = len
    end
    nsl2 = cld(m, C2)
    widths2 = Vector{Int}(undef, nsl2)
    if pad === :global
        fill!(widths2, isempty(rowlen) ? 0 : maximum(rowlen))
    else
        @inbounds for s in 1:nsl2
            w = 0
            for i in ((s - 1) * C2 + 1):min(s * C2, m)
                w = max(w, rowlen[i])
            end
            widths2[s] = w
        end
    end
    total2 = C2 * sum(widths2; init = 0)
    total2 < typemax(Ti) ||
        throw(ArgumentError("resliced structure exceeds the $Ti index range"))
    sliceptr2 = Vector{Ti}(undef, nsl2 + 1)
    sliceptr2[1] = one(Ti)
    @inbounds for s in 1:nsl2
        sliceptr2[s + 1] = sliceptr2[s] + Ti(C2 * widths2[s])
    end
    colind2 = zeros(Ti, total2)
    vals2 = zeros(T, total2)
    perm = zeros(Int, length(colind1))  # old storage position -> new (0 for padding)
    @inbounds for i in 1:m
        base1, _ = _row_base(S.sliceptr, i, C1)
        base2, _ = _row_base(sliceptr2, i, C2)
        for l in 1:rowlen[i]
            q1 = base1 + (l - 1) * C1
            q2 = base2 + (l - 1) * C2
            colind2[q2] = colind1[q1]
            vals2[q2] = vals1[q1]
            perm[q1] = q2
        end
    end
    width2 = (nsl2 > 0 && all(==(widths2[1]), widths2)) ? widths2[1] : 0
    padded2 = total2 != nnz_stored
    tm = S.tmap
    tmap2 = if tm === nothing
        nothing
    else
        positions2 = Vector{Ti}(undef, length(tm.positions))
        @inbounds for p in 1:length(tm.positions)
            positions2[p] = Ti(perm[Int(tm.positions[p])])
        end
        rows2 = if C2 == 1 && width2 > 0
            nothing
        elseif tm.rows !== nothing
            copy(tm.rows)
        else
            # old map was the uniform C == 1 form; recover rows from old positions
            old_rows = Vector{Ti}(undef, length(tm.positions))
            @inbounds for p in 1:length(tm.positions)
                old_rows[p] = Ti((Int(tm.positions[p]) - 1) ÷ S.width + 1)
            end
            old_rows
        end
        TransposeMap(copy(tm.offsets), positions2, rows2)
    end
    S2 = SellStructure{C2}(m, n, colind2, sliceptr2, width2, padded2, tmap2)
    return SellMatrix(vals2, S2)
end

# Upload a host-resident SellMatrix to `backend`, preserving layout and map sequence.
function _to_backend_matrix(backend, A::SellMatrix{T, C}) where {T, C}
    S = A.structure
    tm = S.tmap
    dev_tmap = tm === nothing ? nothing : TransposeMap(
            _to_backend(backend, tm.offsets), _to_backend(backend, tm.positions),
            tm.rows === nothing ? nothing : _to_backend(backend, tm.rows),
        )
    dev_S = SellStructure{C}(
        S.m, S.n, _to_backend(backend, Vector(S.colind)),
        _to_backend(backend, Vector(S.sliceptr)), S.width, S.padded, dev_tmap,
    )
    return SellMatrix(_to_backend(backend, Vector(vec(A.vals))), dev_S)
end
