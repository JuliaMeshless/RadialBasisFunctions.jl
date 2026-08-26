#=
Forward-apply tests: kernel correctness against the sparse oracle over slice heights,
the mul! α/β contract (β = 0 overwrites NaN), padded-slot isolation (NaN flaw-hunter),
threading determinism across the serial/threaded cutoff, inference, and dispatch
integrity of the full mul! surface (a missed forwarding signature silently falls back
to the generic AbstractMatrix mul! — near-identical numerics, catastrophic perf).
=#

using RadialBasisFunctions
using RadialBasisFunctions.EllSparse
using Adapt
using LinearAlgebra
using SparseArrays
using Random: MersenneTwister
using Test
using JLArrays
import KernelAbstractions
using KernelAbstractions: CPU
# JLArrays runs kernels synchronously but omits synchronize(::JLBackend); a no-op
# shim lets the generic device path run under test. hasmethod-guarded: @safetestset
# modules share the global method table, so an unguarded shim in every file would
# overwrite the method once per testset.
if !hasmethod(KernelAbstractions.synchronize, Tuple{JLArrays.JLBackend})
    KernelAbstractions.synchronize(::JLArrays.JLBackend) = nothing
end

rng = MersenneTwister(31)

# ragged fixture with empty rows/columns and an explicit stored zero
S_ragged = sparse(
    [1, 1, 1, 2, 3, 3, 5, 5, 5, 5, 7, 7],
    [2, 4, 5, 1, 3, 5, 1, 2, 3, 4, 2, 5],
    [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    7, 5,
)

@testset "matvec vs sparse oracle: C = $C" for C in (1, 2, 4, 32)
    A = SellMatrix(S_ragged, Val(C))
    m, n = size(A)
    x = randn(rng, n)

    @test A * x ≈ S_ragged * x
    y = fill(NaN, m)
    @test mul!(y, A, x) ≈ S_ragged * x               # β = 0 must overwrite NaN-filled y
    y2 = randn(rng, m)
    expected = 2.5 .* (S_ragged * x) .+ 0.5 .* y2
    @test mul!(copy(y2), A, x, 2.5, 0.5) ≈ expected

    X = randn(rng, n, 3)
    @test A * X ≈ S_ragged * X
    Y = fill(NaN, m, 3)
    @test mul!(Y, A, X) ≈ S_ragged * X
    Y2 = randn(rng, m, 3)
    @test mul!(copy(Y2), A, X, -1.5, 2.0) ≈ -1.5 .* (S_ragged * X) .+ 2.0 .* Y2

    @test_throws DimensionMismatch mul!(zeros(m), A, randn(rng, n + 1))
    @test_throws DimensionMismatch mul!(zeros(m + 1), A, randn(rng, n))
    @test_throws DimensionMismatch mul!(zeros(m, 3), A, randn(rng, n, 2))
end

@testset "padding NaN flaw-hunter: C = $C" for C in (2, 32)
    # Poison every padding slot: a kernel that touches any sentinel-marked slot (or
    # indexes x through a sentinel column) now produces NaN instead of silently adding
    # zero. Deliberately violates the padded-slots-are-zero invariant as a probe.
    A = SellMatrix(S_ragged, Val(C))
    St = structure(A)
    @test St.padded
    v = parent(A)
    for q in eachindex(St.colind)
        iszero(St.colind[q]) && (v[q] = NaN)
    end
    x = randn(rng, size(A, 2))
    @test A * x ≈ S_ragged * x
    w = randn(rng, size(A, 1))
    @test A' * w ≈ S_ragged' * w   # the transpose map must skip sentinels too
end

@testset "serial/threaded parity across the cutoff" begin
    # Rows are independent, so the threaded path must agree bitwise with the serial
    # path. A big matrix runs threaded; its leading-row block rebuilt as its own matrix
    # runs serial with identical per-row storage, so shared rows must match exactly.
    n = 600
    m_big = EllSparse._SELL_SERIAL_CUTOFF + 100
    m_small = 500
    k = 7
    idx = Int32.(reduce(hcat, [rand(rng, 1:n, k) for _ in 1:m_big]))
    vals = randn(rng, k, m_big)
    x = randn(rng, n)
    A_big = SellMatrix(vals, idx, n)
    A_small = SellMatrix(vals[:, 1:m_small], idx[:, 1:m_small], n)
    y_big = A_big * x
    y_small = A_small * x
    @test length(y_big) >= EllSparse._SELL_SERIAL_CUTOFF   # threaded when nthreads > 1
    @test length(y_small) < EllSparse._SELL_SERIAL_CUTOFF  # serial
    @test y_big[1:m_small] == y_small
    # repeated applies are deterministic
    @test A_big * x == y_big
end

@testset "inference and eltype" begin
    A1 = SellMatrix(S_ragged, Val(1))
    A32 = SellMatrix(S_ragged, Val(32))
    x = randn(rng, size(A1, 2))
    @test (@inferred A1 * x) isa Vector{Float64}
    @test (@inferred A32 * x) isa Vector{Float64}
    y = zeros(size(A1, 1))
    @inferred mul!(y, A1, x)
    @inferred mul!(y, A32, x, 2.0, 0.5)

    # uniform matrix-backed C = 1 (the fast path)
    k, m, n = 3, 6, 5
    valsm = randn(rng, k, m)
    idxm = Int32.(reduce(hcat, [rand(rng, 1:n, k) for _ in 1:m]))
    Au = SellMatrix(valsm, idxm, n)
    @test (@inferred Au * randn(rng, n)) isa Vector{Float64}

    # Float32 values are preserved through the kernel
    S32 = SparseMatrixCSC{Float32, Int}(S_ragged)
    for C in (1, 4)
        A = SellMatrix(S32, Val(C))
        x32 = randn(rng, Float32, size(A, 2))
        @test A * x32 isa Vector{Float32}
        @test A * x32 ≈ Float32.(Matrix(A) * x32)
    end
end

@testset "device apply (JLArrays): C = $C" for C in (1, 2, 32)
    A = SellMatrix(S_ragged, Val(C))
    A_d = Adapt.adapt(JLArray, A)
    x = randn(rng, size(A, 2))
    x_d = JLArray(x)
    y_d = A_d * x_d
    @test y_d isa JLArray
    @test Array(y_d) == A * x   # same order, same arithmetic — bitwise
    X = randn(rng, size(A, 2), 3)
    @test Array(A_d * JLArray(X)) == A * X
end

@testset "mul! dispatch integrity (no silent AbstractMatrix fallback)" begin
    A = SellMatrix(S_ragged, Val(2))
    V, M = Vector{Float64}, Matrix{Float64}
    T_A = typeof(A)
    T_At = typeof(A')
    T_Atr = typeof(transpose(A))
    for sig in (
            Tuple{V, T_A, V}, Tuple{V, T_A, V, Float64, Float64},
            Tuple{M, T_A, M}, Tuple{M, T_A, M, Float64, Float64},
            Tuple{V, T_At, V}, Tuple{V, T_At, V, Float64, Float64},
            Tuple{M, T_At, M}, Tuple{M, T_At, M, Float64, Float64},
            Tuple{V, T_Atr, V}, Tuple{V, T_Atr, V, Float64, Float64},
            Tuple{M, T_Atr, M}, Tuple{M, T_Atr, M, Float64, Float64},
        )
        @test parentmodule(which(mul!, sig)) === EllSparse
    end
    for sig in (
            Tuple{T_A, V}, Tuple{T_A, M}, Tuple{T_At, V}, Tuple{T_At, M},
            Tuple{T_Atr, V}, Tuple{T_Atr, M},
        )
        @test parentmodule(which(*, sig)) === EllSparse
    end
end
