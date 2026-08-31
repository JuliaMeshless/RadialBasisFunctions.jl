#=
Conversion cross-agreement (coo/sparse/Matrix/getindex say the same thing, explicit
zeros retained, duplicate indices collapse exactly) and the algebra surface: in-type
same-structure arithmetic, anti-densification when mixing with CSC, Diagonal scaling
on both sides, and the `\` guardrail through sparse.
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
# JLArrays runs kernels synchronously but omits synchronize(::JLBackend); a no-op
# shim lets the generic device path run under test. hasmethod-guarded: @safetestset
# modules share the global method table, so an unguarded shim in every file would
# overwrite the method once per testset.
if !hasmethod(KernelAbstractions.synchronize, Tuple{JLArrays.JLBackend})
    KernelAbstractions.synchronize(::JLArrays.JLBackend) = nothing
end

rng = MersenneTwister(67)

S_ragged = sparse(
    [1, 1, 1, 2, 3, 3, 5, 5, 5, 5, 7, 7],
    [2, 4, 5, 1, 3, 5, 1, 2, 3, 4, 2, 5],
    [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    7, 5,
)

@testset "conversion cross-agreement: C = $C" for C in (1, 2, 32)
    A = SellMatrix(S_ragged, Val(C))
    m, n = size(A)
    Sc = sparse(A)
    @test Sc isa SparseMatrixCSC{Float64, Int}
    @test Sc == S_ragged
    @test SparseMatrixCSC(A) == Sc
    @test Matrix(A) == Matrix(S_ragged)
    @test all(A[i, j] == Sc[i, j] for i in 1:m, j in 1:n)
    # explicit stored zero is retained through the round trip
    @test nnz(Sc) == nnz(S_ragged)
    I, J, V = coo(A)
    @test length(V) == nnz(S_ragged)
    @test sparse(I, J, V, m, n) == S_ragged
end

@testset "duplicate-index columns collapse exactly" begin
    # Fixture shape from the StencilWeights characterization suite: identity row padded
    # with repeated indices — duplicates must sum to exactly {1, 0, 0, 0} → one entry.
    k, n_eval, n_data = 4, 6, 8
    vals = randn(rng, k, n_eval)
    idx = Int32.(reduce(hcat, [rand(rng, 1:n_data, k) for _ in 1:n_eval]))
    vals[:, 5] .= [1.0, 0.0, 0.0, 0.0]
    idx[:, 5] .= Int32(5)
    A = SellMatrix(vals, idx, n_data)
    S = sparse(A)
    @test nnz(S[5, :]) == 1
    @test S[5, 5] == 1.0
    @test A[5, 5] == 1.0
    @test Matrix(A)[5, :] == [i == 5 ? 1.0 : 0.0 for i in 1:n_data]
end

@testset "same-structure algebra: C = $C" for C in (1, 4)
    A = SellMatrix(S_ragged, Val(C))
    B = with_values(A, 2.0 .* parent(A))
    @test sparse(A + B) == 3.0 .* S_ragged
    @test sparse(B - A) == S_ragged
    @test sparse(-A) == -S_ragged
    @test sparse(2.5 * A) == 2.5 .* S_ragged
    @test sparse(A * 2.5) == 2.5 .* S_ragged
    @test sparse(A / 2.0) == S_ragged ./ 2.0
    @test A + B isa SellMatrix
    @test A == A
    @test A != B
    @test isapprox(A, with_values(A, parent(A) .+ 1.0e-12); atol = 1.0e-10)
    # value-equal matrices with distinct structures compare equal via sparse
    B_other = SellMatrix(S_ragged, Val(C == 1 ? 2 : 1))
    @test A == B_other
    @test isapprox(A, with_values(B_other, parent(B_other) .+ 1.0e-12); atol = 1.0e-10)
    @test !isapprox(A, 2.0 * B_other)
end

@testset "Diagonal scaling" begin
    for C in (1, 2, 32)
        A = SellMatrix(S_ragged, Val(C))
        d_rows = randn(rng, size(A, 1))
        d_cols = randn(rng, size(A, 2))
        @test sparse(Diagonal(d_rows) * A) == Diagonal(d_rows) * S_ragged
        @test sparse(A * Diagonal(d_cols)) == S_ragged * Diagonal(d_cols)
        @test Diagonal(d_rows) * A isa SellMatrix
        @test A * Diagonal(d_cols) isa SellMatrix
        @test_throws DimensionMismatch Diagonal(randn(rng, size(A, 1) + 1)) * A
        @test_throws DimensionMismatch A * Diagonal(randn(rng, size(A, 2) + 1))
    end
    # uniform matrix-backed C = 1 broadcast path (distinct indices per column: with
    # duplicates, scale-then-sum vs sum-then-scale differ in the last bit)
    k, m, n = 3, 5, 6
    vals = randn(rng, k, m)
    idx = Int32.(reduce(hcat, [sortperm(randn(rng, n))[1:k] for _ in 1:m]))
    Au = SellMatrix(vals, idx, n)
    d = randn(rng, m)
    @test sparse(Diagonal(d) * Au) == Diagonal(d) * sparse(Au)
    @test structure(Diagonal(d) * Au) === structure(Au)
end

# The two Diagonal paths dispatch on backend: CPU takes the explicit loops above,
# every other backend goes through _scale_rows_kernel! / _scale_cols_kernel!. Only a
# device run exercises those kernels at all, so a scaling bug there would otherwise
# reach GPU users untested.
@testset "device Diagonal scaling (JLArrays): C = $C" for C in (1, 2, 32)
    A = SellMatrix(S_ragged, Val(C))
    A_d = Adapt.adapt(JLArray, A)
    d_rows = randn(rng, size(A, 1))
    d_cols = randn(rng, size(A, 2))

    # Diagonal * A has a broadcast fast path, but only for C == 1 with a uniform width.
    # S_ragged is ragged, so width == 0 there and even C == 1 routes through _scale_rows
    # — i.e. the device kernel. Assert it, so this stays a kernel test if the fixture
    # ever changes. For C > 1 no fast path exists and dispatch reaches _scale_rows
    # unconditionally.
    C == 1 && @test structure(A).width == 0

    row_d = Diagonal(d_rows) * A_d
    col_d = A_d * Diagonal(d_cols)
    @test row_d isa SellMatrix
    @test col_d isa SellMatrix
    # values only: the device structure is a separate object holding device arrays
    @test Array(vec(parent(row_d))) == vec(parent(Diagonal(d_rows) * A))
    @test Array(vec(parent(col_d))) == vec(parent(A * Diagonal(d_cols)))
    # sentinel slots must survive scaling as exact zeros on device too
    @test sparse(Adapt.adapt(Array, row_d)) == Diagonal(d_rows) * S_ragged
    @test sparse(Adapt.adapt(Array, col_d)) == S_ragged * Diagonal(d_cols)
end

@testset "sparse mixing stays sparse (anti-densification)" begin
    A = SellMatrix(S_ragged, Val(2))
    Sc = sprandn(rng, size(A)..., 0.3)
    @test A + Sc isa SparseArrays.AbstractSparseMatrixCSC
    @test A + Sc == sparse(A) + Sc
    @test Sc + A == Sc + sparse(A)
    @test A - Sc == sparse(A) - Sc
    @test Sc - A == Sc - sparse(A)
end

@testset "backslash routes through sparse" begin
    # square nonsingular fixture
    Ssq = sparse([1, 1, 2, 3, 3], [1, 3, 2, 1, 3], [4.0, 1.0, 2.0, 1.0, 3.0], 3, 3)
    A = SellMatrix(Ssq, Val(1))
    b = randn(rng, 3)
    @test A \ b ≈ Ssq \ b
end
