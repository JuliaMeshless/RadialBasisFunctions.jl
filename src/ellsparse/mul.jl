# ------------------------------------------------------------------------------------------
# Forward apply: y[i] = β*y[i] + α * Σₗ vals[q(i,l)] * x[colind[q(i,l)]], one row per
# work item. The sentinel-guarded kernel handles padding; the branchless twin runs when
# the structure has no sentinels (`!padded`).
# ------------------------------------------------------------------------------------------

@kernel function _sell_matvec_kernel!(
        y, @Const(vals), @Const(colind), @Const(sliceptr), @Const(x), α, β, C,
    )
    i = @index(Global)
    base, w = _row_base(sliceptr, i, C)
    acc = zero(eltype(y))
    @inbounds for l in 1:w
        q = base + (l - 1) * C
        j = Int(colind[q])
        if j != 0
            acc += vals[q] * x[j]
        end
    end
    # β == 0 must overwrite y even when it holds NaN (LinearAlgebra mul! contract)
    @inbounds y[i] = iszero(β) ? α * acc : muladd(β, y[i], α * acc)
end

@kernel function _sell_matvec_nopad_kernel!(
        y, @Const(vals), @Const(colind), @Const(sliceptr), @Const(x), α, β, C,
    )
    i = @index(Global)
    base, w = _row_base(sliceptr, i, C)
    acc = zero(eltype(y))
    @inbounds for l in 1:w
        q = base + (l - 1) * C
        acc += vals[q] * x[colind[q]]
    end
    @inbounds y[i] = iszero(β) ? α * acc : muladd(β, y[i], α * acc)
end

function _sell_mul!(
        y::AbstractVector, A::SellMatrix{T, C}, x::AbstractVector, α, β, backend,
    ) where {T, C}
    S = A.structure
    kernel! = S.padded ? _sell_matvec_kernel!(backend) : _sell_matvec_nopad_kernel!(backend)
    kernel!(y, A.vals, S.colind, S.sliceptr, x, α, β, C; ndrange = length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

# Below this row count the tens-of-µs task-spawn floor of @threads outweighs the row
# work, so single-row probes and small matrices take the serial SIMD loop. The value is
# an order-of-magnitude estimate, not a tuned optimum: the two cost curves are nearly
# equal near the crossover, so its exact (machine-dependent) location barely matters.
const _SELL_SERIAL_CUTOFF = 4096

# CPU fast path for uniform unpadded C == 1: plain threads + SIMD over the contiguous
# per-row storage, bypassing per-launch kernel overhead.
function _sell_mul!(
        y::AbstractVector, A::SellMatrix{T, 1}, x::AbstractVector, α, β, ::CPU,
    ) where {T}
    S = A.structure
    if S.width > 0 && !S.padded
        return _uniform_ell_mul!(y, A.vals, S.colind, S.width, x, α, β)
    end
    return _sell_cpu_mul!(y, A, x, α, β)
end

_sell_mul!(y::AbstractVector, A::SellMatrix, x::AbstractVector, α, β, ::CPU) =
    _sell_cpu_mul!(y, A, x, α, β)

function _uniform_ell_mul!(y, vals, colind, k, x, α, β)
    if length(y) < _SELL_SERIAL_CUTOFF || Threads.nthreads() == 1
        @inbounds for i in eachindex(y)
            qb = (i - 1) * k
            acc = zero(eltype(y))
            @simd for l in 1:k
                acc += vals[qb + l] * x[colind[qb + l]]
            end
            y[i] = iszero(β) ? α * acc : muladd(β, y[i], α * acc)
        end
        return y
    end
    # Default :dynamic schedule: unlike :static it may be nested inside a caller's own
    # threaded region and cooperates with the scheduler. Row i is written by exactly
    # one iteration, so thread assignment never affects the values.
    Threads.@threads for i in eachindex(y)
        qb = (i - 1) * k
        acc = zero(eltype(y))
        @inbounds @simd for l in 1:k
            acc += vals[qb + l] * x[colind[qb + l]]
        end
        @inbounds y[i] = iszero(β) ? α * acc : muladd(β, y[i], α * acc)
    end
    return y
end

# Generic-C CPU path: same slice math as the kernels, sentinel-guarded.
function _sell_cpu_mul!(
        y::AbstractVector, A::SellMatrix{T, C}, x::AbstractVector, α, β,
    ) where {T, C}
    S = A.structure
    vals = A.vals
    colind = S.colind
    sliceptr = S.sliceptr
    if length(y) < _SELL_SERIAL_CUTOFF || Threads.nthreads() == 1
        for i in eachindex(y)
            @inbounds y[i] = _sell_row_apply(vals, colind, sliceptr, x, i, C, y[i], α, β)
        end
        return y
    end
    Threads.@threads for i in eachindex(y)
        @inbounds y[i] = _sell_row_apply(vals, colind, sliceptr, x, i, C, y[i], α, β)
    end
    return y
end

@inline function _sell_row_apply(vals, colind, sliceptr, x, i, C, yi, α, β)
    base, w = _row_base(sliceptr, i, C)
    acc = zero(typeof(yi))
    @inbounds for l in 1:w
        q = base + (l - 1) * C
        j = Int(colind[q])
        if j != 0
            acc += vals[q] * x[j]
        end
    end
    return iszero(β) ? α * acc : muladd(β, yi, α * acc)
end

function LinearAlgebra.mul!(
        y::AbstractVector, A::SellMatrix, x::AbstractVector, α::Number, β::Number,
    )
    m, n = size(A)
    length(y) == m || throw(DimensionMismatch("y has length $(length(y)), needs $m"))
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), needs $n"))
    return _sell_mul!(y, A, x, α, β, KernelAbstractions.get_backend(A))
end

LinearAlgebra.mul!(y::AbstractVector, A::SellMatrix, x::AbstractVector) =
    LinearAlgebra.mul!(y, A, x, true, false)

function LinearAlgebra.mul!(
        Y::AbstractMatrix, A::SellMatrix, X::AbstractMatrix, α::Number, β::Number,
    )
    if size(Y) != (size(A, 1), size(X, 2)) || size(X, 1) != size(A, 2)
        throw(DimensionMismatch("mul! sizes: Y $(size(Y)), A $(size(A)), X $(size(X))"))
    end
    return _sell_mul_cols!(Y, A, X, α, β, KernelAbstractions.get_backend(A))
end

LinearAlgebra.mul!(Y::AbstractMatrix, A::SellMatrix, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, A, X, true, false)

function _sell_mul_cols!(Y, A::SellMatrix, X, α, β, ::CPU)
    for j in axes(X, 2)
        _sell_mul!(view(Y, :, j), A, view(X, :, j), α, β, CPU())
    end
    return Y
end

# Device path: launch all columns, synchronize once (per-column syncs stall the queue)
function _sell_mul_cols!(Y, A::SellMatrix{T, C}, X, α, β, backend) where {T, C}
    S = A.structure
    kernel! = S.padded ? _sell_matvec_kernel!(backend) : _sell_matvec_nopad_kernel!(backend)
    for j in axes(X, 2)
        kernel!(
            view(Y, :, j), A.vals, S.colind, S.sliceptr, view(X, :, j), α, β, C;
            ndrange = size(Y, 1),
        )
    end
    KernelAbstractions.synchronize(backend)
    return Y
end

# Destinations are keyed off the values' backing array, not the input: this keeps them
# dense (a sparse input must not produce a sparse destination for the threaded kernel to
# race on structural setindex!) and on the matrix's device.
function Base.:*(A::SellMatrix, x::AbstractVector)
    T = promote_type(eltype(A), eltype(x))
    return LinearAlgebra.mul!(similar(A.vals, T, size(A, 1)), A, x)
end

function Base.:*(A::SellMatrix, X::AbstractMatrix)
    T = promote_type(eltype(A), eltype(X))
    return LinearAlgebra.mul!(similar(A.vals, T, size(A, 1), size(X, 2)), A, X)
end
