using LinearAlgebra: Adjoint, Diagonal
using SparseArrays: SparseMatrixCSC, sparse
using KernelAbstractions: @kernel, @index, @Const, CPU

"""
    StencilWeights{T, M <: AbstractMatrix{T}, MI <: AbstractMatrix{Int32}} <: AbstractMatrix{T}

ELL-format stencil weight storage for `RadialBasisOperator`s.

`vals[l, i]` is the l-th stencil weight of eval point `i` (column order matches the
adjacency list `adjl`) and `idx[l, i]` the index of the data point it multiplies. The
logical size is `(N_eval, N_data)`: `W * x` computes `y[i] = Σₗ vals[l, i] * x[idx[l, i]]`
with one dense length-k dot product per eval point — row-parallel on CPU threads or GPU.

`idx` is frozen after construction (`update_weights!` rewrites only `vals`), and all
components of a gradient-family operator share one `idx` matrix. Dirichlet boundary rows
are padded to k slots: weight 1 at slot 1 with index `eval_idx`, zeros (with the same
in-bounds index) elsewhere.

Use [`sparse`](@ref) / `SparseMatrixCSC` to convert to a sparse matrix for global system
assembly; duplicate padded indices combine, so Dirichlet rows convert to 1-entry identity
rows exactly as the previous sparse storage produced.
"""
struct StencilWeights{T, M <: AbstractMatrix{T}, MI <: AbstractMatrix{Int32}} <: AbstractMatrix{T}
    vals::M
    idx::MI
    n_data::Int
    function StencilWeights(
            vals::M, idx::MI, n_data::Integer
        ) where {T, M <: AbstractMatrix{T}, MI <: AbstractMatrix{Int32}}
        if size(vals) != size(idx)
            throw(DimensionMismatch("vals is $(size(vals)) but idx is $(size(idx))"))
        end
        if n_data > typemax(Int32)
            throw(ArgumentError("N_data = $n_data exceeds the Int32 index range"))
        end
        return new{T, M, MI}(vals, idx, Int(n_data))
    end
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

Base.copy(W::StencilWeights) = StencilWeights(copy(W.vals), copy(W.idx), W.n_data)

function Base.copyto!(dest::StencilWeights, src::StencilWeights)
    if size(dest.vals) != size(src.vals) || dest.n_data != src.n_data
        throw(DimensionMismatch("destination and source StencilWeights differ in shape"))
    end
    copyto!(dest.vals, src.vals)
    copyto!(dest.idx, src.idx)
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
    return StencilWeights(A.vals + B.vals, A.idx, A.n_data)
end

function Base.:-(A::StencilWeights, B::StencilWeights)
    _check_same_stencils(A, B)
    return StencilWeights(A.vals - B.vals, A.idx, A.n_data)
end

Base.:-(W::StencilWeights) = StencilWeights(-W.vals, W.idx, W.n_data)
Base.:*(α::Number, W::StencilWeights) = StencilWeights(α * W.vals, W.idx, W.n_data)
Base.:*(W::StencilWeights, α::Number) = α * W
Base.:/(W::StencilWeights, α::Number) = StencilWeights(W.vals / α, W.idx, W.n_data)

# Diagonal(v) * W scales logical row i, i.e. column i of vals
function Base.:*(D::Diagonal, W::StencilWeights)
    if length(D.diag) != size(W, 1)
        throw(DimensionMismatch("Diagonal has $(length(D.diag)) rows, W has $(size(W, 1))"))
    end
    return StencilWeights(W.vals .* reshape(D.diag, 1, :), W.idx, W.n_data)
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

Adapt.adapt_structure(to, W::StencilWeights) =
    StencilWeights(Adapt.adapt(to, W.vals), Adapt.adapt(to, W.idx), W.n_data)

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

# CPU fast path: plain threads + SIMD, bypassing per-launch kernel overhead. Default
# :dynamic schedule composes with user-level threading.
function _ell_mul!(y::AbstractVector, W::StencilWeights, x::AbstractVector, α, β, ::CPU)
    vals = W.vals
    idx = W.idx
    k = size(vals, 1)
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
    for j in axes(X, 2)
        LinearAlgebra.mul!(view(Y, :, j), W, view(X, :, j), α, β)
    end
    return Y
end

LinearAlgebra.mul!(Y::AbstractMatrix, W::StencilWeights, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, W, X, true, false)

function Base.:*(W::StencilWeights, x::AbstractVector)
    T = promote_type(eltype(W), eltype(x))
    return LinearAlgebra.mul!(similar(x, T, size(W, 1)), W, x)
end

function Base.:*(W::StencilWeights, X::AbstractMatrix)
    T = promote_type(eltype(W), eltype(X))
    return LinearAlgebra.mul!(similar(X, T, size(W, 1), size(X, 2)), W, X)
end

# Adjoint apply: y[idx[l, i]] += vals[l, i] * x[i] — a scatter over the stencil graph.
# Serial by design (deterministic; write collisions across eval points sharing a data
# point rule out naive threading). The transpose-map gather kernel arrives with the
# ∂W-through-eval work (issue #156, stage 3).
function LinearAlgebra.mul!(
        y::AbstractVector, Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector,
        α::Number, β::Number,
    )
    W = Wt.parent
    length(y) == W.n_data ||
        throw(DimensionMismatch("y has length $(length(y)), needs $(W.n_data)"))
    length(x) == size(W.vals, 2) ||
        throw(DimensionMismatch("x has length $(length(x)), needs $(size(W.vals, 2))"))
    if iszero(β)
        fill!(y, zero(eltype(y)))
    elseif !isone(β)
        y .*= β
    end
    vals = W.vals
    idx = W.idx
    k, n = size(vals)
    @inbounds for i in 1:n
        xi = α * x[i]
        iszero(xi) && continue
        for l in 1:k
            y[idx[l, i]] = muladd(vals[l, i], xi, y[idx[l, i]])
        end
    end
    return y
end

LinearAlgebra.mul!(y::AbstractVector, Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector) =
    LinearAlgebra.mul!(y, Wt, x, true, false)

function Base.:*(Wt::Adjoint{<:Any, <:StencilWeights}, x::AbstractVector)
    W = Wt.parent
    T = promote_type(eltype(W), eltype(x))
    return LinearAlgebra.mul!(similar(x, T, W.n_data), Wt, x)
end
