"""
    TransposeMap{Ti, VI, VR}

Column-wise adjacency of a SELL structure: for column `j`,
`positions[offsets[j]:offsets[j+1]-1]` are the flat storage positions of the stored
entries whose column index is `j`. The per-column **sequence is the adjoint summation
order** — the determinism anchor: adjoint/transpose products walk this map as a
per-column gather with a fixed order, so results are bitwise-reproducible under any
thread count, with no atomics. [`build_transpose_map`](@ref) produces ascending
positions; [`reslice`](@ref) remaps them preserving the sequence (not re-sorting), so
a layout change never changes the summation order.

`rows` carries the global row of each entry (aligned with `positions`) and is `nothing`
exactly when `C == 1` with a uniform stored width `k`, where the row is recoverable as
`(q - 1) ÷ k + 1`.
"""
struct TransposeMap{
        Ti <: Integer, VI <: AbstractVector{Ti}, VR <: Union{Nothing, AbstractVector{Ti}},
    }
    offsets::VI    # length n + 1; column j owns positions[offsets[j]:offsets[j+1]-1]
    positions::VI  # length nnz_stored, ascending within each column
    rows::VR       # global row per entry; `nothing` iff uniform C == 1
end

"""
    SellStructure{C, Ti, VI, TM}

The immutable sparsity structure of a [`SellMatrix`](@ref) with slice height `C`:
logical size, column indices, slice offsets, and the optional [`TransposeMap`](@ref).
Multiple matrices alias one structure (`===`) via [`with_values`](@ref), which makes
same-pattern families (shared column indices, shared transpose map) first-class.

`colind` is the flat column-index array; the sentinel `Ti(0)` marks padding slots,
including the ghost rows of a last partial slice when `m % C != 0`. The values array of
any matrix over this structure stores `zero(T)` in every padding slot — an invariant
the same-structure `==`/`isapprox` fast paths rely on. `width` is the uniform slice
width when every slice has the same width, else `0`. `padded` is `true` when any
sentinel exists and gates the unguarded (branchless) kernels.
"""
struct SellStructure{
        C, Ti <: Integer, VI <: AbstractVector{Ti}, TM <: Union{Nothing, TransposeMap},
    }
    m::Int
    n::Int
    colind::VI     # flat storage; sentinel Ti(0) = padding
    sliceptr::VI   # length cld(m, C) + 1, 1-based storage offset of each slice
    width::Int     # uniform stored width, else 0
    padded::Bool   # true when any sentinel slot exists
    tmap::TM
    # Trusted constructor: no validation. Used where the structure is known-consistent —
    # the validating constructors, Adapt, and reslice.
    function SellStructure{C}(
            m::Integer, n::Integer, colind::VI, sliceptr::VI, width::Integer,
            padded::Bool, tmap::TM,
        ) where {C, Ti <: Integer, VI <: AbstractVector{Ti}, TM <: Union{Nothing, TransposeMap}}
        (C isa Int && C >= 1) ||
            throw(ArgumentError("slice height C must be a positive Int; got $C"))
        return new{C, Ti, VI, TM}(Int(m), Int(n), colind, sliceptr, Int(width), padded, tmap)
    end
end

"""
    SellMatrix{T, C, V, S} <: AbstractMatrix{T}

SELL-C sparse matrix: a values array over an immutable [`SellStructure`](@ref). See
the [`EllSparse`](@ref) module docstring for the storage layout. `A * x` computes
`y[i] = Σₗ vals[q(i, l)] * x[colind[q(i, l)]]` row-parallel on CPU threads or GPU;
`A' * x` is a deterministic per-column gather through the structure's transpose map.

`vals` is a flat vector of length `sliceptr[end] - 1`, except in the uniform `C == 1`
case where it may be the `k × m` matrix the layout is bit-identical to (zero-copy:
`parent(A) === vals`).
"""
struct SellMatrix{T, C, V <: AbstractVecOrMat{T}, S <: SellStructure{C}} <: AbstractMatrix{T}
    vals::V
    structure::S
    # Trusted constructor: no validation, no map rebuild. Used where the structure is
    # known-consistent — with_values, algebra results, Adapt, and reslice.
    function SellMatrix(vals::V, structure::S) where {
            T, C, V <: AbstractVecOrMat{T}, S <: SellStructure{C},
        }
        return new{T, C, V, S}(vals, structure)
    end
end

"""
    EllMatrix{T, V, S}

Alias for `SellMatrix{T, 1, V, S}`: slice height 1, i.e. row-contiguous ELL storage.
For a uniform stored width `k` the flat storage is bit-identical to a column-major
`k × m` matrix of per-row entries — the cache-friendly CPU orientation.
"""
const EllMatrix{T, V, S} = SellMatrix{T, 1, V, S}

# ------------------------------------------------------------------------------------------
# Index math (load-bearing; see the module docstring for the layout)
#
#   nslices = cld(m, C);  w_s = (sliceptr[s+1] - sliceptr[s]) ÷ C
#   q(s, r, l) = sliceptr[s] + (l - 1) * C + (r - 1);   i = (s - 1) * C + r
#   C == 1, uniform k:  q = (i - 1) * k + l  ≡ column-major [l, i] of a k × m matrix
#   C == 32: fixed l, consecutive r → consecutive q — coalesced device reads
# ------------------------------------------------------------------------------------------

@inline _slice_of(i::Integer, C::Integer) = (Int(i) - 1) ÷ Int(C) + 1

# storage position of row i's slot 1 (add (l - 1) * C for slot l)
@inline function _row_base(sliceptr, i::Integer, C::Integer)
    s = _slice_of(i, C)
    @inbounds base = Int(sliceptr[s]) + (Int(i) - 1) - (s - 1) * Int(C)
    @inbounds w = (Int(sliceptr[s + 1]) - Int(sliceptr[s])) ÷ Int(C)
    return base, w
end

_nslices(S::SellStructure{C}) where {C} = cld(S.m, C)

# ------------------------------------------------------------------------------------------
# Accessors
# ------------------------------------------------------------------------------------------

"""
    slice_height(A::SellMatrix) -> Int

The slice height `C` of the storage layout: `1` is the row-contiguous CPU orientation,
`32` the coalesced device orientation. Change it with [`reslice`](@ref).
"""
slice_height(::SellMatrix{T, C}) where {T, C} = C
slice_height(::SellStructure{C}) where {C} = C

"""
    structure(A::SellMatrix) -> SellStructure

The immutable sparsity structure backing `A`. Matrices from the same family alias one
structure object (compare with `===` or [`same_structure`](@ref)).
"""
structure(A::SellMatrix) = A.structure

"""
    uniform_width(A::SellMatrix) -> Int

The uniform stored width when every slice has the same width, else `0`.
"""
uniform_width(A::SellMatrix) = A.structure.width
uniform_width(S::SellStructure) = S.width

"""
    parent(A::SellMatrix)

The raw values storage backing `A` (zero-copy): the flat vector, or the `k × m` matrix
in the uniform `C == 1` matrix-backed case. The supported handle for reading or
mutating stored values in place.
"""
Base.parent(A::SellMatrix) = A.vals

"""
    values_matrix(A::SellMatrix{T, 1}) -> AbstractMatrix{T}

The `k × m` per-row values matrix of a uniform-width `C == 1` matrix — `A.vals` itself
when matrix-backed, else a zero-copy `reshape` of the flat storage. Errors for `C != 1`
or ragged widths, where no such matrix view exists.
"""
function values_matrix(A::SellMatrix{T, 1}) where {T}
    S = A.structure
    S.width > 0 || throw(
        ArgumentError(
            "values_matrix requires a uniform stored width; this SellMatrix has ragged slice widths"
        )
    )
    v = A.vals
    return v isa AbstractMatrix ? v : reshape(v, S.width, S.m)
end
values_matrix(::SellMatrix{T, C}) where {T, C} = throw(
    ArgumentError("values_matrix requires slice height C == 1; this SellMatrix has C = $C")
)

"""
    with_values(A::SellMatrix, vals::AbstractVecOrMat) -> SellMatrix

A new matrix over the **same** structure object (`===`) with a different values array.
`vals` must match the storage length and live on the same backend. Padding slots of
`vals` must hold `zero(eltype(vals))` — the same-structure `==` fast path relies on it.
"""
function with_values(A::SellMatrix, vals::AbstractVecOrMat)
    length(vals) == length(A.vals) || throw(
        DimensionMismatch(
            "replacement values have length $(length(vals)); the storage holds $(length(A.vals))"
        )
    )
    if KernelAbstractions.get_backend(vals) != KernelAbstractions.get_backend(A.structure.colind)
        throw(
            ArgumentError(
                "replacement values must live on the structure's backend; got $(KernelAbstractions.get_backend(vals)) vs $(KernelAbstractions.get_backend(A.structure.colind))"
            )
        )
    end
    return SellMatrix(vals, A.structure)
end

"""
    same_structure(A::SellMatrix, B::SellMatrix) -> Bool

Whether `A` and `B` share one sparsity structure: identical objects (`===`), or equal
slice height, size, slice offsets, and column indices.
"""
same_structure(A::SellMatrix, B::SellMatrix) = _same_structure(A.structure, B.structure)

function _same_structure(S1::SellStructure{C}, S2::SellStructure{C}) where {C}
    S1 === S2 && return true
    return S1.m == S2.m && S1.n == S2.n && S1.sliceptr == S2.sliceptr &&
        S1.colind == S2.colind
end
_same_structure(::SellStructure, ::SellStructure) = false

# The transpose map is required by adjoint/transpose products; constructing with
# `transpose_map = false` trades that capability for skipping the build.
@inline function _require_tmap(A::SellMatrix)
    tm = A.structure.tmap
    tm === nothing && throw(
        ArgumentError(
            "this SellMatrix was constructed with transpose_map = false; adjoint/transpose products require the transpose map"
        )
    )
    return tm
end

KernelAbstractions.get_backend(A::SellMatrix) = KernelAbstractions.get_backend(A.vals)
