"""
    EllSparse

Self-contained SELL-C (Sliced ELLpack) sparse matrix storage with CPU and GPU
KernelAbstractions kernels. `SellMatrix{T, C}` stores an `m × n` sparse matrix as
horizontal slices of `C` consecutive rows, each slice padded to its own width:

```
         slots (l) →
row 1  ┌ a  b  c ┐
row 2  └ d  e  ⋅ ┘   slice 1 (C = 2 rows, width w₁ = 3; ⋅ = sentinel padding)
row 3  ┌ f  ⋅    ┐
row 4  └ g  h    ┘   slice 2 (width w₂ = 2)
```

Within slice `s` the entry for (local row `r`, slot `l`) lives at flat storage position

    q = sliceptr[s] + (l - 1) * C + (r - 1)

i.e. slot-major within a slice: for a fixed slot, consecutive rows are adjacent in
memory. With `C` matched to the device warp/wavefront size (32), a row-parallel kernel
reads coalesced — this is the cuSPARSE SELL layout (modulo their `-1` padding sentinel
vs `0` here, and 0- vs 1-based offsets). With `C = 1` the layout degenerates to
row-contiguous ELL: `q = rowstart + l - 1`, and for a *uniform* width `k` the flat
storage is bit-identical to a column-major `k × m` matrix whose column `i` holds row
`i`'s entries — the natural cache-friendly CPU layout for stencil-style matrices.

# Format decisions

  - One type covers both orientations: slice height `C` lives in the type domain
    (`SellMatrix{T, C, ...}`), with [`EllMatrix`](@ref) = `SellMatrix{T, 1}` as an
    alias. Layout changes are explicit via [`reslice`](@ref); `Adapt.adapt`
    never changes layout (a data move must not change adjoint summation order).
  - Classic single-slice slot-major ELL is intentionally absent — it was only ever a
    GPU format (slot-major striding is cache-hostile for CPU row applies, where
    `C = 1` wins), and on GPUs `C = 32` strictly dominates it (cuSPARSE itself
    dropped plain ELL in favor of SELL). Its global-max-width padding survives as
    the constructor option `pad = :global`.
  - Structure and values are split: everything except `vals` lives in an immutable
    [`SellStructure`](@ref) that multiple matrices alias (`===`) via
    [`with_values`](@ref), so families of same-pattern matrices (e.g. gradient
    components) share one column-index array and one transpose map.
  - Adjoint/transpose products are deterministic and atomics-free: a precomputed
    [`TransposeMap`](@ref) turns them into per-column gathers with a fixed summation
    order (ascending storage position), stable under any thread count.

# Scoped-out extension points

  - SELL-C-σ row sorting (a row permutation before slicing; zero benefit at uniform
    row length and it complicates AD seams).
  - COO ingestion (construct via `SparseMatrixCSC` for now).
  - Asynchronous batched `mul!` (multi-column applies enqueue all columns, then
    synchronize once).
  - cuSPARSE interop: their SELL uses sentinel `-1` and 0-based offsets; a future CUDA
    extension converts at the boundary.
  - Parallel transpose-map build (a chunked counting sort with per-chunk cursors can
    preserve the ascending-position order, but the build is a once-per-structure host
    pass dwarfed by weight computation — not worth risking the order contract).
"""
module EllSparse

using Adapt: Adapt
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, CPU
using LinearAlgebra: LinearAlgebra, Adjoint, Diagonal, Transpose
using SparseArrays: SparseArrays, AbstractSparseMatrixCSC, SparseMatrixCSC,
    sparse, nzrange, rowvals, nonzeros

export SellMatrix, EllMatrix, SellStructure, TransposeMap
export sell, coo, reslice, preferred_slice_height, build_transpose_map
export with_values, same_structure, slice_height, structure, values_matrix,
    uniform_width, adapt_family

include("structure.jl")
include("transpose_map.jl")
include("construction.jl")
include("matrix.jl")
include("mul.jl")
include("adjoint_mul.jl")
include("conversions.jl")
include("algebra.jl")
include("adapt.jl")

end # module
