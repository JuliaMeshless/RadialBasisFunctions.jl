#=
Construction tests for the EllSparse module: validating-constructor guards, CSC→SELL
over slice heights × shape fixtures, padding policies, and structure-flag invariants.
The kernels index with @inbounds, so every rejection tested here is what stands between
bad input and undefined behavior.
=#

using RadialBasisFunctions
using RadialBasisFunctions.EllSparse
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

rng = MersenneTwister(2026)

# distinct in-range columns for one stencil column
function randperm_cols(rng, n, k)
    cols = collect(1:n)
    for i in 1:k
        j = rand(rng, i:n)
        cols[i], cols[j] = cols[j], cols[i]
    end
    return cols[1:k]
end

# Named CSC fixtures covering the shapes the slicing math has to get right.
fixtures = [
    "ragged" => sparse(
        [1, 1, 1, 2, 3, 3, 5, 5, 5, 5, 7],
        [2, 4, 5, 1, 3, 5, 1, 2, 3, 4, 2],
        [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # incl. explicit zero
        7, 5,
    ),
    "empty rows and columns" => sparse([2, 5], [1, 3], [1.5, -2.5], 6, 4),
    "ghost rows (m % C != 0)" => sparse(
        [1, 2, 3, 4, 5], [1, 2, 3, 1, 2], [1.0, 2.0, 3.0, 4.0, 5.0], 5, 3
    ),
    "empty (m = 0)" => spzeros(0, 3),
    "single row" => sparse([1, 1, 1], [1, 3, 4], [1.0, 2.0, 3.0], 1, 4),
    "uniform rows" => sparse(
        [1, 1, 2, 2, 3, 3, 4, 4], [1, 2, 2, 3, 1, 3, 2, 3],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4, 3,
    ),
]

@testset "CSC -> SELL over C and pad policies: $name" for (name, S) in fixtures
    m, n = size(S)
    nnz_stored = length(rowvals(S))
    @testset "C = $C, pad = $pad" for C in (1, 2, 4, 32), pad in (:slice, :global)
        A = SellMatrix(S, Val(C); pad)
        St = structure(A)
        @test size(A) == (m, n)
        @test slice_height(A) == C
        # logical round trips
        @test sparse(A) == S
        @test Matrix(A) == Matrix(S)
        # structure invariants
        nsl = cld(m, C)
        @test length(St.sliceptr) == nsl + 1
        @test issorted(St.sliceptr)
        @test Int(St.sliceptr[end]) - 1 == length(St.colind) == length(parent(A))
        @test count(!iszero, St.colind) == nnz_stored
        @test St.padded == (length(St.colind) != nnz_stored)
        # padding slots always store zero values
        @test all(parent(A)[q] == 0 for q in eachindex(St.colind) if iszero(St.colind[q]))
        # width > 0 means every slice has that width
        widths = [Int(St.sliceptr[s + 1] - St.sliceptr[s]) ÷ C for s in 1:nsl]
        if St.width > 0
            @test all(==(St.width), widths)
        else
            @test isempty(widths) || !all(==(widths[1]), widths)
        end
        if pad == :global && m > 0
            @test all(==(maximum(widths; init = 0)), widths)
        end
        # transpose map: positions ascending per column, covering all stored entries
        tm = St.tmap
        @test length(tm.positions) == nnz_stored
        @test Int(tm.offsets[end]) - 1 == nnz_stored
        for j in 1:n
            colpos = tm.positions[Int(tm.offsets[j]):(Int(tm.offsets[j + 1]) - 1)]
            @test issorted(colpos)
            @test all(St.colind[Int(p)] == j for p in colpos)
        end
        @test sort(Int.(tm.positions)) == findall(!iszero, St.colind)
        # rows carried exactly when the (q-1) ÷ width + 1 recovery is unavailable
        if C == 1 && St.width > 0
            @test tm.rows === nothing
        else
            @test tm.rows isa AbstractVector
            @test length(tm.rows) == nnz_stored
        end
    end
end

@testset "sell() runtime slice-height entry" begin
    S = last(fixtures[1])
    for C in (1, 4)
        A = sell(S; slice_height = C)
        @test slice_height(A) == C
        @test sparse(A) == S
    end
    A64 = sell(S; slice_height = 2, index_type = Int64)
    @test eltype(structure(A64).colind) == Int64
    @test sparse(A64) == S
    # structure-level accessor variants agree with the matrix-level ones
    @test slice_height(structure(A64)) == slice_height(A64) == 2
    @test uniform_width(structure(A64)) == uniform_width(A64)
    @test Base.IndexStyle(typeof(A64)) === IndexCartesian()
end

@testset "uniform C = 1 matrix constructor (guards ported from StencilWeights)" begin
    k, m, n = 4, 6, 8
    vals = randn(rng, k, m)
    colind = Int32.(reduce(hcat, [randperm_cols(rng, n, k) for _ in 1:m]))
    A = SellMatrix(vals, colind, n)
    @test size(A) == (m, n)
    @test parent(A) === vals
    @test values_matrix(A) === vals
    @test slice_height(A) == 1
    @test uniform_width(A) == k
    @test !structure(A).padded
    @test structure(A).colind == vec(colind)
    @test Base.mightalias(structure(A).colind, colind)  # zero-copy flat view
    @test A isa EllMatrix

    # getindex sums duplicate-index slots (identity row)
    vals2 = copy(vals)
    colind2 = copy(colind)
    vals2[:, 5] .= [1.0, 0.0, 0.0, 0.0]
    colind2[:, 5] .= Int32(5)
    A2 = SellMatrix(vals2, colind2, n)
    @test A2[5, 5] == 1.0
    S2 = sparse(A2)
    @test nnz(S2[5, :]) == 1

    @test_throws ArgumentError A[1, 1] = 2.0
    @test_throws DimensionMismatch SellMatrix(vals, colind[1:(k - 1), :], n)
    @test_throws ArgumentError SellMatrix(vals, colind, Int64(typemax(Int32)) + 1)
    # Out-of-range column indices are rejected (the apply kernels index with @inbounds)
    @test_throws ArgumentError SellMatrix(vals, fill(Int32(n + 1), k, m), n)
    @test_throws ArgumentError SellMatrix(vals, fill(Int32(0), k, m), n)
    # Mixed-backend struct would crash at first kernel launch — rejected at construction
    @test_throws ArgumentError SellMatrix(JLArray(vals), colind, n)

    # transpose_map = false stores nothing and defers the cost
    A3 = SellMatrix(vals, colind, n; transpose_map = false)
    @test structure(A3).tmap === nothing

    # empty matrix
    A0 = SellMatrix(zeros(3, 0), zeros(Int32, 3, 0), 5)
    @test size(A0) == (0, 5)
    @test sparse(A0) == spzeros(0, 5)
end

@testset "constructor argument validation" begin
    S = last(fixtures[1])
    @test_throws ArgumentError SellMatrix(S, Val(2); pad = :nope)
    @test_throws ArgumentError SellMatrix(S, Val(0))
    @test_throws ArgumentError sell(S; slice_height = -1)
    @test_throws ArgumentError values_matrix(SellMatrix(S, Val(2)))
    @test_throws ArgumentError values_matrix(SellMatrix(S, Val(1)))  # ragged widths
    # CSC-path index-range guard: 144 stored entries overflow Int8's typemax of 127
    @test_throws ArgumentError SellMatrix(sparse(ones(12, 12)), Val(1); index_type = Int8)
end
