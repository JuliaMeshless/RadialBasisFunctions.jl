using LinearAlgebra: Adjoint, Diagonal, Transpose
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC, sparse
using KernelAbstractions: @kernel, @index, @Const, CPU

"""
    EllTransposeMap{VI <: AbstractVector{Int32}}

Transpose adjacency of an ELL stencil structure, CSR-style over data points: for data
point `m`, `positions[offsets[m]:offsets[m+1]-1]` are the linear indices into the
`k × N_eval` weight matrices whose slots reference `m`, ordered by (eval point, slot).
Fixed ordering makes the adjoint apply a deterministic gather — one work item per data
point, no atomics — on CPU threads and GPU alike. Built once at construction; `idx` is
frozen, so the map never invalidates.
"""
struct EllTransposeMap{VI <: AbstractVector{Int32}}
    offsets::VI    # length n_data + 1
    positions::VI  # length k * N_eval
end

Adapt.adapt_structure(to, m::EllTransposeMap) =
    EllTransposeMap(Adapt.adapt(to, m.offsets), Adapt.adapt(to, m.positions))

# Counting sort over the (slot, eval) pairs by data-point index. Linear iteration of the
# column-major idx matrix yields (eval column, slot) ascending order within each data
# point — deterministic regardless of thread count. The sort itself is a serial host
# pass: a device idx is copied to the host once, and the finished map is copied back so
# the resulting struct lives uniformly on idx's backend.
function _build_transpose_map(idx::AbstractMatrix{Int32}, n_data::Int)
    backend = KernelAbstractions.get_backend(idx)
    idx_host = backend isa CPU ? idx : Array(idx)
    nnz = length(idx_host)
    counts = zeros(Int32, n_data + 1)
    @inbounds for q in 1:nnz
        counts[idx_host[q] + 1] += Int32(1)
    end
    offsets = Vector{Int32}(undef, n_data + 1)
    offsets[1] = Int32(1)
    @inbounds for m in 1:n_data
        offsets[m + 1] = offsets[m] + counts[m + 1]
    end
    positions = Vector{Int32}(undef, nnz)
    cursor = copy(offsets)
    @inbounds for q in 1:nnz
        m = idx_host[q]
        positions[cursor[m]] = Int32(q)
        cursor[m] += Int32(1)
    end
    backend isa CPU && return EllTransposeMap(offsets, positions)
    return EllTransposeMap(_to_backend(backend, offsets), _to_backend(backend, positions))
end

# Copy a host vector to `backend` via bulk copyto! (no scalar indexing).
function _to_backend(backend, v::Vector)
    dv = KernelAbstractions.allocate(backend, eltype(v), length(v))
    copyto!(dv, v)
    return dv
end

"""
    StencilWeights{T, M, MI, TM} <: AbstractMatrix{T}

ELL-format stencil weight storage for `RadialBasisOperator`s.

`vals[l, i]` is the l-th stencil weight of eval point `i` (column order matches the
adjacency list `adjl`) and `idx[l, i]` the index of the data point it multiplies. The
logical size is `(N_eval, N_data)`: `W * x` computes `y[i] = Σₗ vals[l, i] * x[idx[l, i]]`
with one dense length-k dot product per eval point — row-parallel on CPU threads or GPU.

`idx` is frozen after construction (`update_weights!` rewrites only `vals`), and all
components of a gradient-family operator share one `idx` matrix and one precomputed
[`EllTransposeMap`](@ref) (`tmap`), which makes the adjoint apply `W' * x` a
deterministic row-parallel gather. Dirichlet boundary rows are padded to k slots:
weight 1 at slot 1 with index `eval_idx`, zeros (with the same in-bounds index)
elsewhere.

Use [`sparse`](@ref) / `SparseMatrixCSC` to convert to a sparse matrix for global system
assembly; duplicate padded indices combine, so Dirichlet rows convert to 1-entry identity
rows exactly as the previous sparse storage produced.
"""
struct StencilWeights{
        T, M <: AbstractMatrix{T}, MI <: AbstractMatrix{Int32}, TM <: EllTransposeMap,
    } <: AbstractMatrix{T}
    vals::M
    idx::MI
    n_data::Int
    tmap::TM
    # Trusted constructor: no validation, no map rebuild. Used where the structure is
    # known-consistent — Adapt, Enzyme shadows, algebra results, and shared-idx tuples.
    function StencilWeights(
            vals::M, idx::MI, n_data::Integer, tmap::TM
        ) where {T, M <: AbstractMatrix{T}, MI <: AbstractMatrix{Int32}, TM <: EllTransposeMap}
        return new{T, M, MI, TM}(vals, idx, Int(n_data), tmap)
    end
end

function StencilWeights(vals::AbstractMatrix, idx::AbstractMatrix{Int32}, n_data::Integer)
    if size(vals) != size(idx)
        throw(DimensionMismatch("vals is $(size(vals)) but idx is $(size(idx))"))
    end
    # The apply kernels launch on get_backend(vals) and read idx/tmap inside the kernel,
    # so a mixed-backend struct would crash at first use — reject it here.
    if KernelAbstractions.get_backend(vals) != KernelAbstractions.get_backend(idx)
        throw(
            ArgumentError(
                "vals and idx must live on the same backend; got " *
                    "$(KernelAbstractions.get_backend(vals)) and " *
                    "$(KernelAbstractions.get_backend(idx))"
            )
        )
    end
    if n_data > typemax(Int32) || length(idx) > typemax(Int32)
        throw(ArgumentError("stencil structure exceeds the Int32 index range"))
    end
    # The apply kernels index x with @inbounds and Matrix(W) writes with @inbounds,
    # so out-of-range neighbor indices must be rejected here (the old sparse() build
    # threw for exactly this input).
    if !isempty(idx)
        lo, hi = extrema(idx)
        if lo < 1 || hi > n_data
            throw(
                ArgumentError(
                    "neighbor indices must lie in 1:$n_data; got values in $lo:$hi"
                )
            )
        end
    end
    return StencilWeights(vals, idx, n_data, _build_transpose_map(idx, Int(n_data)))
end

# Array interface — logical shape is (N_eval, N_data)
Base.size(W::StencilWeights) = (size(W.vals, 2), W.n_data)
Base.IndexStyle(::Type{<:StencilWeights}) = IndexCartesian()

"""
    parent(W::StencilWeights)

Return the dense `k × N_eval` weight value matrix backing `W` (zero-copy). This is the
supported handle for reading or mutating weight values in place and for AD losses over
built weights (e.g. `sum(parent(W) .^ 2)`).
"""
Base.parent(W::StencilWeights) = W.vals

function Base.getindex(W::StencilWeights{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(W, i, j)
    acc = zero(T)
    @inbounds for l in axes(W.vals, 1)
        if W.idx[l, i] == j
            acc += W.vals[l, i]
        end
    end
    return acc
end

function Base.setindex!(::StencilWeights, v, i::Int, j::Int)
    throw(
        ArgumentError(
            "StencilWeights has a fixed stencil structure; mutate `parent(W)` to change " *
                "weight values, or convert with `sparse(W)` for general sparse editing."
        )
    )
end

Base.copy(W::StencilWeights) = StencilWeights(copy(W.vals), copy(W.idx), W.n_data, W.tmap)

# The index matrix is frozen after construction, so copyto! transfers values only —
# and refuses sources with a different stencil structure rather than rewriting idx
# (which may be aliased by operators derived through the algebra methods).
function Base.copyto!(dest::StencilWeights, src::StencilWeights)
    if size(dest.vals) != size(src.vals) || dest.n_data != src.n_data
        throw(DimensionMismatch("destination and source StencilWeights differ in shape"))
    end
    _check_same_stencils(dest, src)
    copyto!(dest.vals, src.vals)
    return dest
end

_same_stencils(A::StencilWeights, B::StencilWeights) =
    A.n_data == B.n_data && (A.idx === B.idx || A.idx == B.idx)

function Base.:(==)(A::StencilWeights, B::StencilWeights)
    size(A) == size(B) || return false
    _same_stencils(A, B) && return A.vals == B.vals
    return sparse(A) == sparse(B)
end

function Base.isapprox(A::StencilWeights, B::StencilWeights; kwargs...)
    size(A) == size(B) || return false
    _same_stencils(A, B) && return isapprox(A.vals, B.vals; kwargs...)
    return isapprox(sparse(A), sparse(B); kwargs...)
end

# Algebra — in-type whenever both operands share the stencil structure. Operator algebra
# guarantees this via `_check_compatible` (equal adjl), so `+`/`-`/scaling never densify.
function _check_same_stencils(A::StencilWeights, B::StencilWeights)
    return _same_stencils(A, B) || throw(
        ArgumentError(
            "StencilWeights operands have different stencil index structures; " *
                "convert with `sparse` for general sparse arithmetic."
        )
    )
end

function Base.:+(A::StencilWeights, B::StencilWeights)
    _check_same_stencils(A, B)
    return StencilWeights(A.vals + B.vals, A.idx, A.n_data, A.tmap)
end

function Base.:-(A::StencilWeights, B::StencilWeights)
    _check_same_stencils(A, B)
    return StencilWeights(A.vals - B.vals, A.idx, A.n_data, A.tmap)
end

Base.:-(W::StencilWeights) = StencilWeights(-W.vals, W.idx, W.n_data, W.tmap)

# Mixing with sparse weights (e.g. combining with a VirtualPartial operator) stays
# sparse — the generic AbstractMatrix fallback would silently densify to N_eval × N_data.
Base.:+(A::StencilWeights, B::AbstractSparseMatrixCSC) = sparse(A) + B
Base.:+(A::AbstractSparseMatrixCSC, B::StencilWeights) = A + sparse(B)
Base.:-(A::StencilWeights, B::AbstractSparseMatrixCSC) = sparse(A) - B
Base.:-(A::AbstractSparseMatrixCSC, B::StencilWeights) = A - sparse(B)
Base.:*(α::Number, W::StencilWeights) = StencilWeights(α * W.vals, W.idx, W.n_data, W.tmap)
Base.:*(W::StencilWeights, α::Number) = α * W
Base.:/(W::StencilWeights, α::Number) = StencilWeights(W.vals / α, W.idx, W.n_data, W.tmap)

# Diagonal(v) * W scales logical row i, i.e. column i of vals
function Base.:*(D::Diagonal, W::StencilWeights)
    if length(D.diag) != size(W, 1)
        throw(DimensionMismatch("Diagonal has $(length(D.diag)) rows, W has $(size(W, 1))"))
    end
    return StencilWeights(W.vals .* reshape(D.diag, 1, :), W.idx, W.n_data, W.tmap)
end

# Generic AbstractMatrix `\` would densify and LU-factorize; route through sparse instead
Base.:\(W::StencilWeights, b::AbstractVecOrMat) = sparse(W) \ b

# Conversions
function Base.Matrix(W::StencilWeights{T}) where {T}
    A = zeros(T, size(W))
    k, n = size(W.vals)
    @inbounds for i in 1:n, l in 1:k
        A[i, W.idx[l, i]] += W.vals[l, i]
    end
    return A
end

function SparseArrays.sparse(W::StencilWeights{T}) where {T}
    k, n = size(W.vals)
    nnz = k * n
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{T}(undef, nnz)
    pos = 1
    @inbounds for i in 1:n, l in 1:k
        I[pos] = i
        J[pos] = Int(W.idx[l, i])
        V[pos] = W.vals[l, i]
        pos += 1
    end
    # sparse() sums duplicates, so Dirichlet-padded columns (all slots share one index)
    # collapse to single identity entries; explicit zeros elsewhere are retained, matching
    # the previous COO-built storage bit-for-bit.
    return sparse(I, J, V, n, W.n_data)
end

SparseArrays.SparseMatrixCSC(W::StencilWeights) = sparse(W)

Adapt.adapt_structure(to, W::StencilWeights) = StencilWeights(
    Adapt.adapt(to, W.vals), Adapt.adapt(to, W.idx), W.n_data, Adapt.adapt(to, W.tmap)
)

KernelAbstractions.get_backend(W::StencilWeights) = KernelAbstractions.get_backend(W.vals)

# ------------------------------------------------------------------------------------------
# Apply kernels: y[i] = β*y[i] + α * Σₗ vals[l, i] * x[idx[l, i]], one row per work item.
# Operator evaluation must reach these only through `_eval_op`/`mul!` — the AD extensions
# register rules on those seams so Enzyme never traces the threaded kernel (see #150).
# ------------------------------------------------------------------------------------------

@kernel function _ell_matvec_kernel!(y, @Const(vals), @Const(idx), @Const(x), α, β)
    i = @index(Global)
    acc = zero(eltype(y))
    @inbounds for l in axes(vals, 1)
        acc += vals[l, i] * x[idx[l, i]]
    end
    # β == 0 must overwrite y even when it holds NaN (LinearAlgebra mul! contract)
    @inbounds y[i] = iszero(β) ? α * acc : muladd(β, y[i], α * acc)
end

function _ell_mul!(y::AbstractVector, W::StencilWeights, x::AbstractVector, α, β, backend)
    kernel! = _ell_matvec_kernel!(backend)
    kernel!(y, W.vals, W.idx, x, α, β; ndrange = length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

# Below this row count the tens-of-µs task-spawn floor of @threads outweighs the row
# work, so single-point probes and small operators take the serial SIMD loop.
const _ELL_SERIAL_CUTOFF = 4096

# CPU fast path: plain threads + SIMD, bypassing per-launch kernel overhead. Default
# :dynamic schedule composes with user-level threading.
function _ell_mul!(y::AbstractVector, W::StencilWeights, x::AbstractVector, α, β, ::CPU)
    vals = W.vals
    idx = W.idx
    k = size(vals, 1)
    if length(y) < _ELL_SERIAL_CUTOFF || Threads.nthreads() == 1
        @inbounds for i in eachindex(y)
            acc = zero(eltype(y))
            @simd for l in 1:k
                acc += vals[l, i] * x[idx[l, i]]
            end
            y[i] = iszero(β) ? α * acc : muladd(β, y[i], α * acc)
        end
        return y
    end
    Threads.@threads for i in eachindex(y)
        acc = zero(eltype(y))
        @inbounds @simd for l in 1:k
            acc += vals[l, i] * x[idx[l, i]]
        end
        @inbounds y[i] = iszero(β) ? α * acc : muladd(β, y[i], α * acc)
    end
    return y
end

function LinearAlgebra.mul!(
        y::AbstractVector, W::StencilWeights, x::AbstractVector, α::Number, β::Number
    )
    m, n = size(W)
    length(y) == m || throw(DimensionMismatch("y has length $(length(y)), needs $m"))
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), needs $n"))
    return _ell_mul!(y, W, x, α, β, KernelAbstractions.get_backend(W))
end

LinearAlgebra.mul!(y::AbstractVector, W::StencilWeights, x::AbstractVector) =
    LinearAlgebra.mul!(y, W, x, true, false)

function LinearAlgebra.mul!(
        Y::AbstractMatrix, W::StencilWeights, X::AbstractMatrix, α::Number, β::Number
    )
    if size(Y) != (size(W, 1), size(X, 2)) || size(X, 1) != size(W, 2)
        throw(DimensionMismatch("mul! sizes: Y $(size(Y)), W $(size(W)), X $(size(X))"))
    end
    return _ell_mul_cols!(Y, W, X, α, β, KernelAbstractions.get_backend(W))
end

function _ell_mul_cols!(Y, W::StencilWeights, X, α, β, ::CPU)
    for j in axes(X, 2)
        _ell_mul!(view(Y, :, j), W, view(X, :, j), α, β, CPU())
    end
    return Y
end

# Device path: launch all columns, synchronize once (per-column syncs stall the queue)
function _ell_mul_cols!(Y, W::StencilWeights, X, α, β, backend)
    kernel! = _ell_matvec_kernel!(backend)
    for j in axes(X, 2)
        kernel!(view(Y, :, j), W.vals, W.idx, view(X, :, j), α, β; ndrange = size(Y, 1))
    end
    KernelAbstractions.synchronize(backend)
    return Y
end

LinearAlgebra.mul!(Y::AbstractMatrix, W::StencilWeights, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, W, X, true, false)

# Destinations are keyed off the weights' backing array, not the input: this keeps them
# dense (a sparse input must not produce a sparse destination for the threaded kernel to
# race on structural setindex!) and on the weights' device.
function Base.:*(W::StencilWeights, x::AbstractVector)
    T = promote_type(eltype(W), eltype(x))
    return LinearAlgebra.mul!(similar(W.vals, T, size(W, 1)), W, x)
end

function Base.:*(W::StencilWeights, X::AbstractMatrix)
    T = promote_type(eltype(W), eltype(X))
    return LinearAlgebra.mul!(similar(W.vals, T, size(W, 1), size(X, 2)), W, X)
end

# Adjoint apply via the transpose map: y[m] = β y[m] + α Σ_{(l,i): idx[l,i]=m} vals[l,i] x[i].
# A pure gather — one work item per data point, fixed summation order (deterministic under
# any thread count), no atomics, GPU-capable.
@kernel function _ell_adjoint_kernel!(
        y, @Const(vals), @Const(offsets), @Const(positions), @Const(x), α, β, k
    )
    m = @index(Global)
    acc = zero(eltype(y))
    @inbounds for p in offsets[m]:(offsets[m + 1] - 1)
        q = Int(positions[p])
        acc += vals[q] * x[(q - 1) ÷ k + 1]
    end
    @inbounds y[m] = iszero(β) ? α * acc : muladd(β, y[m], α * acc)
end

function _ell_adjoint_mul!(y, W::StencilWeights, x, α, β, backend)
    kernel! = _ell_adjoint_kernel!(backend)
    kernel!(
        y, W.vals, W.tmap.offsets, W.tmap.positions, x, α, β, size(W.vals, 1);
        ndrange = length(y),
    )
    KernelAbstractions.synchronize(backend)
    return y
end

function _ell_adjoint_mul!(y, W::StencilWeights, x, α, β, ::CPU)
    vals = W.vals
    offsets = W.tmap.offsets
    positions = W.tmap.positions
    k = size(vals, 1)
    if length(y) < _ELL_SERIAL_CUTOFF || Threads.nthreads() == 1
        @inbounds for m in eachindex(y)
            acc = zero(eltype(y))
            for p in offsets[m]:(offsets[m + 1] - 1)
                q = Int(positions[p])
                acc += vals[q] * x[(q - 1) ÷ k + 1]
            end
            y[m] = iszero(β) ? α * acc : muladd(β, y[m], α * acc)
        end
        return y
    end
    Threads.@threads for m in eachindex(y)
        acc = zero(eltype(y))
        @inbounds for p in offsets[m]:(offsets[m + 1] - 1)
            q = Int(positions[p])
            acc += vals[q] * x[(q - 1) ÷ k + 1]
        end
        @inbounds y[m] = iszero(β) ? α * acc : muladd(β, y[m], α * acc)
    end
    return y
end

function LinearAlgebra.mul!(
        y::AbstractVector, Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector,
        α::Number, β::Number,
    )
    W = Wt.parent
    length(y) == W.n_data ||
        throw(DimensionMismatch("y has length $(length(y)), needs $(W.n_data)"))
    length(x) == size(W.vals, 2) ||
        throw(DimensionMismatch("x has length $(length(x)), needs $(size(W.vals, 2))"))
    return _ell_adjoint_mul!(y, W, x, α, β, KernelAbstractions.get_backend(W))
end

LinearAlgebra.mul!(y::AbstractVector, Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector) =
    LinearAlgebra.mul!(y, Wt, x, true, false)

function LinearAlgebra.mul!(
        Y::AbstractMatrix, Wt::Adjoint{<:Any, <:StencilWeights}, X::AbstractMatrix,
        α::Number, β::Number,
    )
    W = Wt.parent
    return _ell_adjoint_cols!(Y, W, X, α, β, KernelAbstractions.get_backend(W))
end

function _ell_adjoint_cols!(Y, W::StencilWeights, X, α, β, ::CPU)
    for j in axes(X, 2)
        _ell_adjoint_mul!(view(Y, :, j), W, view(X, :, j), α, β, CPU())
    end
    return Y
end

function _ell_adjoint_cols!(Y, W::StencilWeights, X, α, β, backend)
    kernel! = _ell_adjoint_kernel!(backend)
    for j in axes(X, 2)
        kernel!(
            view(Y, :, j), W.vals, W.tmap.offsets, W.tmap.positions, view(X, :, j),
            α, β, size(W.vals, 1); ndrange = size(Y, 1),
        )
    end
    KernelAbstractions.synchronize(backend)
    return Y
end

LinearAlgebra.mul!(Y::AbstractMatrix, Wt::Adjoint{<:Any, <:StencilWeights}, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, Wt, X, true, false)

function Base.:*(Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector)
    W = Wt.parent
    T = promote_type(eltype(W), eltype(x))
    return LinearAlgebra.mul!(similar(W.vals, T, W.n_data), Wt, x)
end

function Base.:*(Wt::Adjoint{<:Any, <:StencilWeights}, X::AbstractMatrix)
    W = Wt.parent
    T = promote_type(eltype(W), eltype(X))
    return LinearAlgebra.mul!(similar(W.vals, T, W.n_data, size(X, 2)), Wt, X)
end

# Weights are real, so transpose delegates to the adjoint scatter
LinearAlgebra.mul!(
    y::AbstractVector, Wt::Transpose{<:Any, <:StencilWeights}, x::AbstractVector,
    α::Number, β::Number,
) = LinearAlgebra.mul!(y, Wt.parent', x, α, β)
LinearAlgebra.mul!(y::AbstractVector, Wt::Transpose{<:Any, <:StencilWeights}, x::AbstractVector) =
    LinearAlgebra.mul!(y, Wt.parent', x, true, false)
Base.:*(Wt::Transpose{<:Any, <:StencilWeights}, x::AbstractVector) = Wt.parent' * x
Base.:*(Wt::Transpose{<:Any, <:StencilWeights}, X::AbstractMatrix) = Wt.parent' * X
