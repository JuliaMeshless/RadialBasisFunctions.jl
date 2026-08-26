#=
Adjoint-apply tests. The contract under test is stronger than correctness: the gather
order is fixed by the transpose map's `positions` sequence per column, so adjoint
results must be BITWISE identical across repeated calls, thread counts, and the
serial/threaded cutoff — the determinism the AD seams rely on. A test-side serial
gather that walks the map in sequence is the order oracle.
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

rng = MersenneTwister(43)

S_ragged = sparse(
    [1, 1, 1, 2, 3, 3, 5, 5, 5, 5, 7, 7],
    [2, 4, 5, 1, 3, 5, 1, 2, 3, 4, 2, 5],
    [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    7, 5,
)

# The order oracle: gather column j's entries in the map's sequence, serially.
function reference_adjoint(A, x)
    St = structure(A)
    tm = St.tmap
    vals = vec(parent(A))
    y = zeros(promote_type(eltype(A), eltype(x)), size(A, 2))
    for j in eachindex(y)
        acc = zero(eltype(y))
        for p in Int(tm.offsets[j]):(Int(tm.offsets[j + 1]) - 1)
            q = Int(tm.positions[p])
            i = tm.rows === nothing ? (q - 1) ÷ St.width + 1 : Int(tm.rows[p])
            acc += vals[q] * x[i]
        end
        y[j] = acc
    end
    return y
end

@testset "adjoint vs sparse oracle: C = $C" for C in (1, 2, 4, 32)
    A = SellMatrix(S_ragged, Val(C))
    m, n = size(A)
    v = randn(rng, m)

    @test A' * v ≈ S_ragged' * v
    @test transpose(A) * v ≈ transpose(S_ragged) * v
    y = fill(NaN, n)
    @test mul!(y, A', v) ≈ S_ragged' * v              # β = 0 overwrites NaN
    y2 = randn(rng, n)
    @test mul!(copy(y2), A', v, 2.5, 0.5) ≈ 2.5 .* (S_ragged' * v) .+ 0.5 .* y2

    V = randn(rng, m, 3)
    @test A' * V ≈ S_ragged' * V
    Y = fill(NaN, n, 3)
    @test mul!(Y, A', V) ≈ S_ragged' * V
    Y2 = randn(rng, n, 3)
    @test mul!(copy(Y2), A', V, -1.0, 3.0) ≈ -1.0 .* (S_ragged' * V) .+ 3.0 .* Y2

    # the map sequence is the exact summation order
    @test A' * v == reference_adjoint(A, v)

    @test_throws DimensionMismatch mul!(zeros(n + 1), A', v)
    @test_throws DimensionMismatch mul!(zeros(n), A', randn(rng, m + 1))
end

@testset "bitwise determinism, repeated and across the cutoff" begin
    # Big enough that the threaded path runs (columns > cutoff); the reference gather
    # is serial, so agreement here is threaded-vs-serial bitwise parity.
    n = EllSparse._SELL_SERIAL_CUTOFF + 200
    m = 3000
    k = 9
    idx = Int32.(reduce(hcat, [rand(rng, 1:n, k) for _ in 1:m]))
    vals = randn(rng, k, m)
    A = SellMatrix(vals, idx, n)
    v = randn(rng, m)
    y1 = A' * v
    @test y1 == reference_adjoint(A, v)
    for _ in 1:10
        @test A' * v == y1
    end
    # small (serial) matrices agree with the same reference
    A_small = SellMatrix(vals[:, 1:100], idx[:, 1:100], n)
    v_small = v[1:100]
    @test A_small' * v_small == reference_adjoint(A_small, v_small)
end

@testset "rows-path vs uniform-path agreement" begin
    # The same logical matrix at C = 1 (rows === nothing) and C > 1 (rows stored) must
    # agree to rounding; the summation order differs between layouts, so this is ≈, not
    # == — bitwise equality across layouts is reslice's contract, tested there.
    A1 = SellMatrix(S_ragged, Val(1))
    v = randn(rng, size(S_ragged, 1))
    for C in (2, 4, 32)
        AC = SellMatrix(S_ragged, Val(C))
        @test structure(AC).tmap.rows !== nothing
        @test AC' * v ≈ A1' * v
    end
end

@testset "transpose_map = false throws informatively" begin
    k, m, n = 3, 5, 6
    vals = randn(rng, k, m)
    idx = Int32.(reduce(hcat, [rand(rng, 1:n, k) for _ in 1:m]))
    A = SellMatrix(vals, idx, n; transpose_map = false)
    x = randn(rng, n)
    @test A * x isa Vector{Float64}   # forward path unaffected
    v = randn(rng, m)
    @test_throws ArgumentError A' * v
    @test_throws ArgumentError mul!(zeros(n), A', v)
    @test_throws ArgumentError transpose(A) * v
end

@testset "device adjoint (JLArrays): C = $C" for C in (1, 4)
    A = SellMatrix(S_ragged, Val(C))
    A_d = Adapt.adapt(JLArray, A)
    v = randn(rng, size(A, 1))
    y_d = A_d' * JLArray(v)
    @test y_d isa JLArray
    # adapt preserves the map sequence, so the device gather is bitwise-equal
    @test Array(y_d) == A' * v
    V = randn(rng, size(A, 1), 2)
    @test Array(A_d' * JLArray(V)) == A' * V
end

@testset "complex transpose routes through conj" begin
    Sc = sparse([1, 2, 2, 3], [2, 1, 3, 2], ComplexF64[1 + 2im, 3 - im, -2 + im, 4im], 3, 3)
    Ac = SellMatrix(Sc, Val(1))
    v = randn(rng, ComplexF64, 3)
    @test Ac' * v ≈ Sc' * v
    @test transpose(Ac) * v ≈ transpose(Sc) * v
    y = fill(NaN + NaN * im, 3)
    @test mul!(y, transpose(Ac), v) ≈ transpose(Sc) * v
end
