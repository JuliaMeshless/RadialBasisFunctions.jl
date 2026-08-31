using LinearAlgebra: Adjoint, Diagonal, Transpose
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC, sparse
using .EllSparse: EllSparse, SellMatrix, with_values, same_structure

"""
    StencilWeights{T, E <: EllSparse.SellMatrix{T}} <: AbstractMatrix{T}

ELL-format stencil weight storage for `RadialBasisOperator`s — a thin policy wrapper
over [`EllSparse.SellMatrix`](@ref RadialBasisFunctions.EllSparse), which owns the
storage layout and the CPU/GPU apply kernels. The wrapper keeps the RBF-specific
policy: sparse-mixing algebra that never densifies, the `\\` guardrail through
`sparse`, the frozen-structure `copyto!` refusal, and the `parent` contract.

`parent(W)[l, i]` is the l-th stencil weight of eval point `i` (column order matches
the adjacency list `adjl`) and the frozen index structure records the data point it
multiplies. The logical size is `(N_eval, N_data)`: `W * x` computes
`y[i] = Σₗ vals[l, i] * x[idx[l, i]]` with one dense length-k dot product per eval
point — row-parallel on CPU threads or GPU.

The index structure is frozen after construction (`update_weights!` rewrites only the
values), and all components of a gradient-family operator share one
`EllSparse.SellStructure` — one index matrix and one precomputed transpose map — which
makes the adjoint apply `W' * x` a deterministic row-parallel gather.
A row may be padded to k slots — a single nonzero at slot 1, zeros elsewhere all
sharing that same in-bounds index — to express an identity row in a fixed-k layout.

Use [`sparse`](@ref) / `SparseMatrixCSC` to convert to a sparse matrix for global
system assembly; duplicate padded indices combine, so padded rows convert to
1-entry identity rows exactly as the previous sparse storage produced.
"""
struct StencilWeights{T, E <: SellMatrix{T}} <: AbstractMatrix{T}
    ell::E
end

# Validating constructor: the weight-assembly and AD-forward paths build through this
# shim; EllSparse validates (dimension match, mixed-backend rejection, Int32 range,
# in-bounds indices — the kernels index with @inbounds).
StencilWeights(vals::AbstractMatrix, idx::AbstractMatrix{Int32}, n_data::Integer) =
    StencilWeights(SellMatrix(vals, idx, Int(n_data)))

# ------------------------------------------------------------------------------------------
# Internal seams — the only supported paths to the storage internals.
# ------------------------------------------------------------------------------------------

# Sibling component over the same structure object (shared index matrix + transpose
# map): the gradient-family assembly seam.
_shared_component(W::StencilWeights, vals::AbstractMatrix) =
    StencilWeights(with_values(W.ell, vals))

# Zero-valued shadow aliasing the primal's frozen structure — matching Enzyme's
# make_zero convention (copy_if_inactive = Val(false)) for guaranteed-inactive arrays;
# nothing ever writes shadow indices.
_zero_shadow(W::StencilWeights) = StencilWeights(with_values(W.ell, zero(parent(W.ell))))

# The k × N_eval neighbor-index matrix view of the frozen structure (zero-copy
# reshape). Host stencil-major weights only — the AD gathers and tests that read
# per-stencil indices are structurally host-pinned.
function _neighbor_matrix(W::StencilWeights)
    S = EllSparse.structure(W.ell)
    return reshape(S.colind, S.width, S.m)
end

# ------------------------------------------------------------------------------------------
# Device orientation policy. EllSparse's `adapt` never changes layout; RBF applies the
# reslice-to-preferred-orientation policy at the OPERATOR-level Adapt boundary: reslice
# on host to the destination backend's preferred slice height (C = 1 CPU, C = 32
# device — coalesced), then upload. The reslice remaps the transpose map preserving
# sequence order, so device adjoints keep the CPU summation order.
# ------------------------------------------------------------------------------------------

_slice_val(::SellMatrix{T, C}) where {T, C} = Val(C)

# Probe which backend `to` adapts arrays onto (an empty upload — no data movement).
_dest_backend(to, ::Type{T}) where {T} =
    KernelAbstractions.get_backend(Adapt.adapt(to, Vector{T}(undef, 0)))

_with_layout(ell::SellMatrix{T, C}, ::Val{C}) where {T, C} = ell
_with_layout(ell::SellMatrix, ::Val{C}) where {C} = EllSparse.reslice(ell, Val(C))

_adapt_preferring_layout(to, ell::SellMatrix{T}) where {T} = Adapt.adapt(
    to, _with_layout(ell, EllSparse.preferred_slice_height(_dest_backend(to, T)))
)

# Reslice a same-structure family, re-aliasing every component onto ONE resliced
# structure so the sharing survives the layout change.
function _reslice_family(ells::NTuple{N, SellMatrix}, ::Val{C}) where {N, C}
    EllSparse.slice_height(first(ells)) == C && return ells
    resliced = map(e -> EllSparse.reslice(e, Val(C)), ells)
    first_r = first(resliced)
    return ntuple(i -> i == 1 ? first_r : with_values(first_r, parent(resliced[i])), N)
end

function _adapt_family_preferring_layout(to, ells::NTuple{N, SellMatrix}) where {N}
    T = eltype(first(ells))
    C = EllSparse.preferred_slice_height(_dest_backend(to, T))
    return EllSparse.adapt_family(to, _reslice_family(ells, C))
end

# Array interface — logical shape is (N_eval, N_data)
Base.size(W::StencilWeights) = size(W.ell)
Base.IndexStyle(::Type{<:StencilWeights}) = IndexCartesian()

"""
    parent(W::StencilWeights)

For host-resident weights — the only kind construction and AD produce — return the
dense `k × N_eval` stencil-major weight value matrix backing `W` (zero-copy): the
supported handle for reading or mutating weight values in place and for AD losses over
built weights (e.g. `sum(parent(W) .^ 2)`). Device-adapted weights return the
device-layout values array — treat it as opaque storage and go through `copyto!`,
`sparse`, or `Matrix` instead.
"""
Base.parent(W::StencilWeights) = parent(W.ell)

Base.getindex(W::StencilWeights, i::Int, j::Int) = W.ell[i, j]

function Base.setindex!(::StencilWeights, v, i::Int, j::Int)
    throw(
        ArgumentError(
            "StencilWeights has a fixed stencil structure; mutate `parent(W)` to change " *
                "weight values, or convert with `sparse(W)` for general sparse editing."
        )
    )
end

Base.copy(W::StencilWeights) = StencilWeights(copy(W.ell))

# The index structure is frozen after construction, so copyto! transfers values only —
# and refuses sources with a different stencil structure rather than rewriting indices
# (which may be aliased by operators derived through the algebra methods). A destination
# with a different orientation or backend (a device-adapted operator's weights) takes
# the orientation-aware path instead.
function Base.copyto!(dest::StencilWeights, src::StencilWeights)
    if size(dest) != size(src)
        throw(DimensionMismatch("destination and source StencilWeights differ in shape"))
    end
    same_layout = EllSparse.slice_height(dest.ell) == EllSparse.slice_height(src.ell) &&
        KernelAbstractions.get_backend(dest.ell) == KernelAbstractions.get_backend(src.ell)
    same_layout || return _copyvalues!(dest, src)
    if length(parent(dest.ell)) != length(parent(src.ell))
        throw(DimensionMismatch("destination and source StencilWeights differ in shape"))
    end
    _check_same_stencils(dest, src)
    copyto!(parent(dest.ell), parent(src.ell))
    return dest
end

# Orientation-aware value transfer (update_weights!/copyto! on device-adapted
# operators): permute the host stencil-major values into the destination's layout on
# host, then one bulk upload — never a scalar-indexing device loop. Structure identity
# is guaranteed by the frozen-adjl invariant (both sides describe the same stencils);
# sizes are checked, index contents are trusted.
function _copyvalues!(dest::StencilWeights, src::StencilWeights)
    if !(KernelAbstractions.get_backend(src.ell) isa CPU)
        throw(
            ArgumentError(
                "copyto! into a StencilWeights with a different layout expects a " *
                    "host-built source (weights are always constructed on host)"
            )
        )
    end
    resliced = _with_layout(src.ell, _slice_val(dest.ell))
    if length(parent(dest.ell)) != length(parent(resliced))
        throw(DimensionMismatch("destination and source StencilWeights differ in shape"))
    end
    copyto!(vec(parent(dest.ell)), Vector(vec(parent(resliced))))
    return dest
end

function Base.:(==)(A::StencilWeights, B::StencilWeights)
    return A.ell == B.ell
end

Base.isapprox(A::StencilWeights, B::StencilWeights; kwargs...) =
    isapprox(A.ell, B.ell; kwargs...)

# Algebra — in-type whenever both operands share the stencil structure. Operator algebra
# guarantees this via `_check_compatible` (equal adjl), so `+`/`-`/scaling never densify.
function _check_same_stencils(A::StencilWeights, B::StencilWeights)
    return same_structure(A.ell, B.ell) || throw(
        ArgumentError(
            "StencilWeights operands have different stencil index structures; " *
                "convert with `sparse` for general sparse arithmetic."
        )
    )
end

function Base.:+(A::StencilWeights, B::StencilWeights)
    _check_same_stencils(A, B)
    return StencilWeights(A.ell + B.ell)
end

function Base.:-(A::StencilWeights, B::StencilWeights)
    _check_same_stencils(A, B)
    return StencilWeights(A.ell - B.ell)
end

Base.:-(W::StencilWeights) = StencilWeights(-W.ell)

# Mixing with sparse weights (e.g. combining with a VirtualPartial operator) stays
# sparse — the generic AbstractMatrix fallback would silently densify to N_eval × N_data.
Base.:+(A::StencilWeights, B::AbstractSparseMatrixCSC) = sparse(A) + B
Base.:+(A::AbstractSparseMatrixCSC, B::StencilWeights) = A + sparse(B)
Base.:-(A::StencilWeights, B::AbstractSparseMatrixCSC) = sparse(A) - B
Base.:-(A::AbstractSparseMatrixCSC, B::StencilWeights) = A - sparse(B)
Base.:*(α::Number, W::StencilWeights) = StencilWeights(α * W.ell)
Base.:*(W::StencilWeights, α::Number) = α * W
Base.:/(W::StencilWeights, α::Number) = StencilWeights(W.ell / α)

# Diagonal(v) * W scales logical row i, i.e. column i of the value matrix
function Base.:*(D::Diagonal, W::StencilWeights)
    if length(D.diag) != size(W, 1)
        throw(DimensionMismatch("Diagonal has $(length(D.diag)) rows, W has $(size(W, 1))"))
    end
    return StencilWeights(D * W.ell)
end

# Generic AbstractMatrix `\` would densify and LU-factorize; route through sparse instead
Base.:\(W::StencilWeights, b::AbstractVecOrMat) = sparse(W) \ b

# Conversions (device-resident weights route through a host copy inside EllSparse)
Base.Matrix(W::StencilWeights) = Matrix(W.ell)

# sparse() sums duplicates, so padded columns (all slots share one index)
# collapse to single identity entries; explicit zeros elsewhere are retained, matching
# the previous COO-built storage bit-for-bit.
SparseArrays.sparse(W::StencilWeights) = sparse(W.ell)
SparseArrays.SparseMatrixCSC(W::StencilWeights) = sparse(W)

Adapt.adapt_structure(to, W::StencilWeights) = StencilWeights(Adapt.adapt(to, W.ell))

KernelAbstractions.get_backend(W::StencilWeights) = KernelAbstractions.get_backend(W.ell)

# ------------------------------------------------------------------------------------------
# Apply surface: full mul!/* forwarding to the EllSparse kernels. Operator evaluation
# must reach these only through `_eval_op`/`mul!` — the AD extensions register rules on
# those seams so Enzyme never traces the threaded kernel (see #150). Every signature is
# forwarded explicitly: a missed one would silently fall back to the generic
# AbstractMatrix mul! — near-identical numerics, catastrophic perf, and it would bypass
# the Enzyme rule seams. Guarded by dispatch-integrity tests.
# ------------------------------------------------------------------------------------------

LinearAlgebra.mul!(
    y::AbstractVector, W::StencilWeights, x::AbstractVector, α::Number, β::Number,
) = LinearAlgebra.mul!(y, W.ell, x, α, β)
LinearAlgebra.mul!(y::AbstractVector, W::StencilWeights, x::AbstractVector) =
    LinearAlgebra.mul!(y, W.ell, x, true, false)
LinearAlgebra.mul!(
    Y::AbstractMatrix, W::StencilWeights, X::AbstractMatrix, α::Number, β::Number,
) = LinearAlgebra.mul!(Y, W.ell, X, α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, W::StencilWeights, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, W.ell, X, true, false)

Base.:*(W::StencilWeights, x::AbstractVector) = W.ell * x
Base.:*(W::StencilWeights, X::AbstractMatrix) = W.ell * X

# Adjoint apply: the deterministic transpose-map gather (fixed summation order under
# any thread count, no atomics, GPU-capable).
LinearAlgebra.mul!(
    y::AbstractVector, Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector,
    α::Number, β::Number,
) = LinearAlgebra.mul!(y, Wt.parent.ell', x, α, β)
LinearAlgebra.mul!(y::AbstractVector, Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector) =
    LinearAlgebra.mul!(y, Wt.parent.ell', x, true, false)
LinearAlgebra.mul!(
    Y::AbstractMatrix, Wt::Adjoint{<:Any, <:StencilWeights}, X::AbstractMatrix,
    α::Number, β::Number,
) = LinearAlgebra.mul!(Y, Wt.parent.ell', X, α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, Wt::Adjoint{<:Any, <:StencilWeights}, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, Wt.parent.ell', X, true, false)

Base.:*(Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector) = Wt.parent.ell' * x
Base.:*(Wt::Adjoint{<:Any, <:StencilWeights}, X::AbstractMatrix) = Wt.parent.ell' * X

# Weights are real, so transpose delegates to the adjoint gather
LinearAlgebra.mul!(
    y::AbstractVector, Wt::Transpose{<:Any, <:StencilWeights}, x::AbstractVector,
    α::Number, β::Number,
) = LinearAlgebra.mul!(y, Wt.parent.ell', x, α, β)
LinearAlgebra.mul!(y::AbstractVector, Wt::Transpose{<:Any, <:StencilWeights}, x::AbstractVector) =
    LinearAlgebra.mul!(y, Wt.parent.ell', x, true, false)
Base.:*(Wt::Transpose{<:Any, <:StencilWeights}, x::AbstractVector) = Wt.parent.ell' * x
Base.:*(Wt::Transpose{<:Any, <:StencilWeights}, X::AbstractMatrix) = Wt.parent.ell' * X
