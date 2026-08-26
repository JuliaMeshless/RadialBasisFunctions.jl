#=
Reslice tests: layout changes are explicit, logically lossless, and — the hard
guarantee — adjoint-order-preserving. `reslice` remaps the transpose map's positions
through the (row, slot) bijection in sequence order, so the resliced adjoint is
bitwise-equal to the original, which is what lets a device-preferred layout keep CPU
summation order. Also covers preferred_slice_height and adapt/reslice composition.
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

rng = MersenneTwister(73)

S_ragged = sparse(
    [1, 1, 1, 2, 3, 3, 5, 5, 5, 5, 7, 7],
    [2, 4, 5, 1, 3, 5, 1, 2, 3, 4, 2, 5],
    [1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    7, 5,
)

@testset "reslice is logically lossless: C = $C1 -> $C2" for C1 in (1, 2, 32),
        C2 in (1, 2, 4, 32)

    A = SellMatrix(S_ragged, Val(C1))
    R = reslice(A, Val(C2))
    @test slice_height(R) == C2
    @test sparse(R) == sparse(A)
    @test sparse(R) == S_ragged
    x = randn(rng, size(A, 2))
    @test R * x ≈ A * x
end

@testset "adjoint order survives reslice: $C1 -> $C2" for C1 in (1, 2), C2 in (1, 4, 32)
    # A big ill-conditioned-ish random fixture so any order change would actually show
    # up in the low bits.
    n = 400
    m = 1000
    k = 11
    idx = Int32.(reduce(hcat, [rand(rng, 1:n, k) for _ in 1:m]))
    vals = randn(rng, k, m) .* exp.(5.0 .* randn(rng, k, m))
    A1 = SellMatrix(vals, idx, n)
    A = C1 == 1 ? A1 : reslice(A1, Val(C1))
    R = reslice(A, Val(C2))
    v = randn(rng, m)
    @test R' * v == A1' * v
    @test transpose(R) * v == transpose(A1) * v
end

@testset "uniform RBF-shaped round trip 1 -> 32 -> 1" begin
    k, m, n = 5, 200, 150
    idx = Int32.(reduce(hcat, [rand(rng, 1:n, k) for _ in 1:m]))
    vals = randn(rng, k, m)
    A = SellMatrix(vals, idx, n)
    R32 = reslice(A, Val(32))
    @test slice_height(R32) == 32
    @test uniform_width(R32) == k
    @test structure(R32).padded == (m % 32 != 0)   # only ghost rows pad a uniform matrix
    @test structure(R32).tmap.rows !== nothing
    back = reslice(R32, Val(1))
    @test uniform_width(back) == k
    @test structure(back).tmap.rows === nothing
    @test vec(parent(back)) == vec(vals)
    @test structure(back).colind == vec(idx)
    @test structure(back).tmap.positions == structure(A).tmap.positions
    v = randn(rng, m)
    @test back' * v == A' * v
end

@testset "reslice re-pads: pad = :global and same-C tightening" begin
    A = SellMatrix(S_ragged, Val(2); pad = :global)
    @test uniform_width(A) == 4
    tight = reslice(A, Val(2))               # :slice default trims global padding
    @test uniform_width(tight) == 0           # ragged again
    @test sparse(tight) == S_ragged
    wide = reslice(tight, Val(2); pad = :global)
    @test uniform_width(wide) == 4
    @test sparse(wide) == S_ragged
    v = randn(rng, size(A, 1))
    @test wide' * v == A' * v
end

@testset "tmap-less matrices reslice to tmap-less" begin
    A = SellMatrix(S_ragged, Val(1); transpose_map = false)
    R = reslice(A, Val(4))
    @test structure(R).tmap === nothing
    @test sparse(R) == S_ragged
end

@testset "device reslice round-trips through the host" begin
    A = SellMatrix(S_ragged, Val(1))
    A_d = Adapt.adapt(JLArray, A)
    R_d = reslice(A_d, Val(32))
    @test slice_height(R_d) == 32
    @test parent(R_d) isa JLArray
    @test structure(R_d).colind isa JLArray
    @test Array(structure(R_d).tmap.positions) ==
        structure(reslice(A, Val(32))).tmap.positions
    x = randn(rng, size(A, 2))
    @test Array(R_d * JLArray(x)) == reslice(A, Val(32)) * x
    v = randn(rng, size(A, 1))
    @test Array(R_d' * JLArray(v)) == A' * v   # order preserved through upload
end

@testset "preferred_slice_height policy" begin
    @test preferred_slice_height(CPU()) === Val(1)
    @test preferred_slice_height(JLArrays.JLBackend()) === Val(32)
end

@testset "adapt never changes layout" begin
    for C in (1, 32)
        A = SellMatrix(S_ragged, Val(C))
        @test slice_height(Adapt.adapt(JLArray, A)) == C
        @test slice_height(Adapt.adapt(Array, Adapt.adapt(JLArray, A))) == C
    end
end

@testset "reslice argument validation" begin
    A = SellMatrix(S_ragged, Val(2))
    @test_throws ArgumentError reslice(A, Val(0))
    @test_throws ArgumentError reslice(A, Val(2); pad = :nope)
end
