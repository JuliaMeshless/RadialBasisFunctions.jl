#=
Structure aliasing tests: the `===`-shared SellStructure is what makes same-pattern
families (shared column indices, shared transpose map) first-class. Every operation
that preserves the pattern must preserve the aliasing — with_values, algebra results,
copy, and adapt_family — and copyto! must refuse foreign structures rather than
rewrite aliased indices.
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

rng = MersenneTwister(59)

k, m, n = 4, 6, 8
vals = randn(rng, k, m)
idx = Int32.(reduce(hcat, [rand(rng, 1:n, k) for _ in 1:m]))
A = SellMatrix(vals, idx, n)

@testset "with_values aliases the structure" begin
    B = with_values(A, randn(rng, k, m))
    @test structure(B) === structure(A)
    @test same_structure(A, B)
    # flat vector values are accepted for the same structure
    Bv = with_values(A, randn(rng, k * m))
    @test structure(Bv) === structure(A)
    @test_throws DimensionMismatch with_values(A, randn(rng, k, m + 1))
    @test_throws ArgumentError with_values(A, JLArray(randn(rng, k, m)))
end

@testset "algebra results alias the structure" begin
    B = with_values(A, randn(rng, k, m))
    for R in (A + B, A - B, -A, 2.5 * A, A * 2.5, A / 2.0)
        @test structure(R) === structure(A)
    end
    D = Diagonal(randn(rng, m))
    @test structure(D * A) === structure(A)
    Dn = Diagonal(randn(rng, n))
    @test structure(A * Dn) === structure(A)
    @test structure(copy(A)) === structure(A)
end

@testset "copyto! refuses foreign structures" begin
    B = with_values(A, randn(rng, k, m))
    dest = copy(A)
    copyto!(dest, B)
    @test dest == B
    # equal-valued but distinct structure objects are still accepted (== fallback)
    A_rebuilt = SellMatrix(copy(vals), copy(idx), n)
    @test structure(A_rebuilt) !== structure(A)
    copyto!(dest, A_rebuilt)
    @test dest == A

    # genuinely different pattern → rejected, values untouched
    idx2 = copy(idx)
    idx2[1, 1] = idx2[1, 1] == Int32(1) ? Int32(2) : Int32(1)
    C = SellMatrix(copy(vals), idx2, n)
    before = copy(parent(dest))
    @test_throws ArgumentError copyto!(dest, C)
    @test parent(dest) == before
    @test_throws ArgumentError A + C
    @test_throws DimensionMismatch copyto!(copy(A), SellMatrix(vals[:, 1:3], idx[:, 1:3], n))
end

@testset "adapt_family shares one adapted structure" begin
    B = with_values(A, randn(rng, k, m))
    C2 = with_values(A, randn(rng, k, m))
    fam_d = adapt_family(JLArray, (A, B, C2))
    @test length(fam_d) == 3
    S_d = structure(fam_d[1])
    @test all(F -> structure(F) === S_d, fam_d)
    @test S_d.colind isa JLArray
    @test all(F -> parent(F) isa JLArray, fam_d)
    # values move faithfully
    @test Array(parent(fam_d[2])) == parent(B)
    # empty and singleton families
    @test adapt_family(JLArray, ()) === ()
    fam1 = adapt_family(Array, (A,))
    @test structure(fam1[1]).colind isa Vector{Int32}

    # members that do not alias one structure are refused
    A_rebuilt = SellMatrix(copy(vals), copy(idx), n)
    @test_throws ArgumentError adapt_family(JLArray, (A, A_rebuilt))
end

@testset "plain adapt keeps layout and map sequence" begin
    A32 = SellMatrix(sparse(Matrix(A)), Val(32))
    A32_d = Adapt.adapt(JLArray, A32)
    @test slice_height(A32_d) == 32
    St_h, St_d = structure(A32), structure(A32_d)
    @test Array(St_d.colind) == St_h.colind
    @test Array(St_d.sliceptr) == St_h.sliceptr
    @test St_d.width == St_h.width && St_d.padded == St_h.padded
    @test Array(St_d.tmap.positions) == St_h.tmap.positions
    @test Array(St_d.tmap.rows) == St_h.tmap.rows
    # round trip back to host
    A32_h = Adapt.adapt(Array, A32_d)
    @test A32_h == A32
end
