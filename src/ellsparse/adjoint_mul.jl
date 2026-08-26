# ------------------------------------------------------------------------------------------
# Adjoint apply via the transpose map: y[j] = β y[j] + α Σ conj(vals[q]) x[row(q)] over
# the stored entries with column j. A pure gather — one work item per column, fixed
# summation order (ascending storage position — deterministic under any thread count),
# no atomics, GPU-capable. `reslice` preserves the order via its sequence-preserving
# position remap, so device adjoints keep the original summation order.
#
# Two twins per backend: the uniform C == 1 form recovers the row as (q-1) ÷ k + 1
# (`rows === nothing`); the general form reads the map's `rows` alongside `positions`.
# ------------------------------------------------------------------------------------------

@kernel function _sell_adjoint_uniform_kernel!(
        y, @Const(vals), @Const(offsets), @Const(positions), @Const(x), α, β, k,
    )
    j = @index(Global)
    acc = zero(eltype(y))
    @inbounds for p in offsets[j]:(offsets[j + 1] - 1)
        q = Int(positions[p])
        acc += conj(vals[q]) * x[(q - 1) ÷ k + 1]
    end
    @inbounds y[j] = iszero(β) ? α * acc : muladd(β, y[j], α * acc)
end

@kernel function _sell_adjoint_rows_kernel!(
        y, @Const(vals), @Const(offsets), @Const(positions), @Const(rows), @Const(x), α, β,
    )
    j = @index(Global)
    acc = zero(eltype(y))
    @inbounds for p in offsets[j]:(offsets[j + 1] - 1)
        acc += conj(vals[Int(positions[p])]) * x[Int(rows[p])]
    end
    @inbounds y[j] = iszero(β) ? α * acc : muladd(β, y[j], α * acc)
end

function _sell_adjoint_mul!(y, A::SellMatrix, x, α, β, backend)
    tm = _require_tmap(A)
    return _sell_adjoint_mul!(y, A, tm, tm.rows, x, α, β, backend)
end

function _sell_adjoint_mul!(y, A::SellMatrix, tm::TransposeMap, ::Nothing, x, α, β, backend)
    kernel! = _sell_adjoint_uniform_kernel!(backend)
    kernel!(y, A.vals, tm.offsets, tm.positions, x, α, β, uniform_width(A); ndrange = length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

function _sell_adjoint_mul!(
        y, A::SellMatrix, tm::TransposeMap, rows::AbstractVector, x, α, β, backend,
    )
    kernel! = _sell_adjoint_rows_kernel!(backend)
    kernel!(y, A.vals, tm.offsets, tm.positions, rows, x, α, β; ndrange = length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

function _sell_adjoint_mul!(y, A::SellMatrix, tm::TransposeMap, ::Nothing, x, α, β, ::CPU)
    vals = A.vals
    offsets = tm.offsets
    positions = tm.positions
    k = uniform_width(A)
    if length(y) < _SELL_SERIAL_CUTOFF || Threads.nthreads() == 1
        @inbounds for j in eachindex(y)
            acc = zero(eltype(y))
            for p in offsets[j]:(offsets[j + 1] - 1)
                q = Int(positions[p])
                acc += conj(vals[q]) * x[(q - 1) ÷ k + 1]
            end
            y[j] = iszero(β) ? α * acc : muladd(β, y[j], α * acc)
        end
        return y
    end
    Threads.@threads for j in eachindex(y)
        acc = zero(eltype(y))
        @inbounds for p in offsets[j]:(offsets[j + 1] - 1)
            q = Int(positions[p])
            acc += conj(vals[q]) * x[(q - 1) ÷ k + 1]
        end
        @inbounds y[j] = iszero(β) ? α * acc : muladd(β, y[j], α * acc)
    end
    return y
end

function _sell_adjoint_mul!(
        y, A::SellMatrix, tm::TransposeMap, rows::AbstractVector, x, α, β, ::CPU,
    )
    vals = A.vals
    offsets = tm.offsets
    positions = tm.positions
    if length(y) < _SELL_SERIAL_CUTOFF || Threads.nthreads() == 1
        @inbounds for j in eachindex(y)
            acc = zero(eltype(y))
            for p in offsets[j]:(offsets[j + 1] - 1)
                acc += conj(vals[Int(positions[p])]) * x[Int(rows[p])]
            end
            y[j] = iszero(β) ? α * acc : muladd(β, y[j], α * acc)
        end
        return y
    end
    Threads.@threads for j in eachindex(y)
        acc = zero(eltype(y))
        @inbounds for p in offsets[j]:(offsets[j + 1] - 1)
            acc += conj(vals[Int(positions[p])]) * x[Int(rows[p])]
        end
        @inbounds y[j] = iszero(β) ? α * acc : muladd(β, y[j], α * acc)
    end
    return y
end

function LinearAlgebra.mul!(
        y::AbstractVector, At::Adjoint{<:Any, <:SellMatrix}, x::AbstractVector,
        α::Number, β::Number,
    )
    A = At.parent
    length(y) == size(A, 2) ||
        throw(DimensionMismatch("y has length $(length(y)), needs $(size(A, 2))"))
    length(x) == size(A, 1) ||
        throw(DimensionMismatch("x has length $(length(x)), needs $(size(A, 1))"))
    return _sell_adjoint_mul!(y, A, x, α, β, KernelAbstractions.get_backend(A))
end

LinearAlgebra.mul!(y::AbstractVector, At::Adjoint{<:Any, <:SellMatrix}, x::AbstractVector) =
    LinearAlgebra.mul!(y, At, x, true, false)

function LinearAlgebra.mul!(
        Y::AbstractMatrix, At::Adjoint{<:Any, <:SellMatrix}, X::AbstractMatrix,
        α::Number, β::Number,
    )
    A = At.parent
    if size(Y) != (size(A, 2), size(X, 2)) || size(X, 1) != size(A, 1)
        throw(DimensionMismatch("mul! sizes: Y $(size(Y)), A' $(size(At)), X $(size(X))"))
    end
    return _sell_adjoint_cols!(Y, A, X, α, β, KernelAbstractions.get_backend(A))
end

LinearAlgebra.mul!(Y::AbstractMatrix, At::Adjoint{<:Any, <:SellMatrix}, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, At, X, true, false)

function _sell_adjoint_cols!(Y, A::SellMatrix, X, α, β, ::CPU)
    for j in axes(X, 2)
        _sell_adjoint_mul!(view(Y, :, j), A, view(X, :, j), α, β, CPU())
    end
    return Y
end

# Device path: launch all columns, synchronize once
function _sell_adjoint_cols!(Y, A::SellMatrix, X, α, β, backend)
    tm = _require_tmap(A)
    if tm.rows === nothing
        kernel! = _sell_adjoint_uniform_kernel!(backend)
        for j in axes(X, 2)
            kernel!(
                view(Y, :, j), A.vals, tm.offsets, tm.positions, view(X, :, j),
                α, β, uniform_width(A); ndrange = size(Y, 1),
            )
        end
    else
        kernel! = _sell_adjoint_rows_kernel!(backend)
        for j in axes(X, 2)
            kernel!(
                view(Y, :, j), A.vals, tm.offsets, tm.positions, tm.rows, view(X, :, j),
                α, β; ndrange = size(Y, 1),
            )
        end
    end
    KernelAbstractions.synchronize(backend)
    return Y
end

function Base.:*(At::Adjoint{<:Any, <:SellMatrix}, x::AbstractVector)
    A = At.parent
    T = promote_type(eltype(A), eltype(x))
    return LinearAlgebra.mul!(similar(A.vals, T, size(A, 2)), At, x)
end

function Base.:*(At::Adjoint{<:Any, <:SellMatrix}, X::AbstractMatrix)
    A = At.parent
    T = promote_type(eltype(A), eltype(X))
    return LinearAlgebra.mul!(similar(A.vals, T, size(A, 2), size(X, 2)), At, X)
end

# For real element types transpose ≡ adjoint (conj is the identity in the gather).
LinearAlgebra.mul!(
    y::AbstractVector, At::Transpose{<:Real, <:SellMatrix}, x::AbstractVector,
    α::Number, β::Number,
) = LinearAlgebra.mul!(y, At.parent', x, α, β)
LinearAlgebra.mul!(y::AbstractVector, At::Transpose{<:Real, <:SellMatrix}, x::AbstractVector) =
    LinearAlgebra.mul!(y, At.parent', x, true, false)
LinearAlgebra.mul!(
    Y::AbstractMatrix, At::Transpose{<:Real, <:SellMatrix}, X::AbstractMatrix,
    α::Number, β::Number,
) = LinearAlgebra.mul!(Y, At.parent', X, α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, At::Transpose{<:Real, <:SellMatrix}, X::AbstractMatrix) =
    LinearAlgebra.mul!(Y, At.parent', X, true, false)
Base.:*(At::Transpose{<:Real, <:SellMatrix}, x::AbstractVector) = At.parent' * x
Base.:*(At::Transpose{<:Real, <:SellMatrix}, X::AbstractMatrix) = At.parent' * X

# Complex transpose routes through conj of the adjoint gather: transpose(A) * x =
# conj(A' * conj(x)). Allocates one temporary; the map stays the single source of the
# gather order.
function LinearAlgebra.mul!(
        y::AbstractVector, At::Transpose{<:Complex, <:SellMatrix}, x::AbstractVector,
        α::Number, β::Number,
    )
    t = At.parent' * conj(x)
    if iszero(β)
        y .= α .* conj.(t)
    else
        y .= β .* y .+ α .* conj.(t)
    end
    return y
end
LinearAlgebra.mul!(y::AbstractVector, At::Transpose{<:Complex, <:SellMatrix}, x::AbstractVector) =
    LinearAlgebra.mul!(y, At, x, true, false)
Base.:*(At::Transpose{<:Complex, <:SellMatrix}, x::AbstractVector) =
    conj.(At.parent' * conj(x))
