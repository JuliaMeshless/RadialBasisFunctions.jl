#=
Pins the EllSparse transpose map bit-for-bit to StencilWeights' _build_transpose_map
while both implementations exist. The map's `positions` sequence IS the adjoint
summation order, so exact equality here — not ≈ — is what lets StencilWeights migrate
onto EllSparse without changing a single adjoint bit. Highest-risk seam of the whole
migration; everything downstream (Enzyme rules, device adjoints) inherits this order.
=#

using RadialBasisFunctions
using RadialBasisFunctions.EllSparse
using Random: MersenneTwister
using SparseArrays
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

rng = MersenneTwister(11)

# StencilWeights is a thin wrapper since the cut-over; its map lives in the shared
# structure. (On the pre-cut-over revision this file pinned EllSparse's map against the
# independent _build_transpose_map implementation; it now pins the wrapper wiring.)
wtmap(W::StencilWeights) = structure(W.ell).tmap

random_idx(rng, k, n_eval, n_data) =
    Int32.(reduce(hcat, [rand(rng, 1:n_data, k) for _ in 1:n_eval]; init = zeros(Int32, k, 0)))

@testset "host parity: k=$k, N_eval=$n_eval, N_data=$n_data" for (k, n_eval, n_data) in
    ((4, 6, 8), (1, 1, 1), (5, 40, 12), (3, 200, 50), (4, 0, 8))
    idx = random_idx(rng, k, n_eval, n_data)
    vals = randn(rng, k, n_eval)
    W = StencilWeights(vals, idx, n_data)
    A = SellMatrix(vals, idx, n_data)
    tm = structure(A).tmap
    @test tm.offsets == wtmap(W).offsets
    @test tm.positions == wtmap(W).positions
    @test tm.offsets isa Vector{Int32}
    @test tm.positions isa Vector{Int32}
    @test tm.rows === nothing
end

@testset "host parity: duplicate-index columns" begin
    k, n_eval, n_data = 4, 6, 8
    idx = random_idx(rng, k, n_eval, n_data)
    vals = randn(rng, k, n_eval)
    vals[:, 5] .= [1.0, 0.0, 0.0, 0.0]
    idx[:, 5] .= Int32(5)
    W = StencilWeights(vals, idx, n_data)
    A = SellMatrix(vals, idx, n_data)
    @test structure(A).tmap.offsets == wtmap(W).offsets
    @test structure(A).tmap.positions == wtmap(W).positions
end

@testset "device parity (JLArrays)" begin
    k, n_eval, n_data = 4, 12, 9
    idx = random_idx(rng, k, n_eval, n_data)
    vals = randn(rng, k, n_eval)
    W_d = StencilWeights(JLArray(vals), JLArray(idx), n_data)
    A_d = SellMatrix(JLArray(vals), JLArray(idx), n_data)
    tm = structure(A_d).tmap
    @test tm.offsets isa JLArray{Int32, 1}
    @test tm.positions isa JLArray{Int32, 1}
    @test Array(tm.offsets) == Array(wtmap(W_d).offsets)
    @test Array(tm.positions) == Array(wtmap(W_d).positions)
    # and the device map matches the host map entry-for-entry
    A_h = SellMatrix(vals, idx, n_data)
    @test Array(tm.offsets) == structure(A_h).tmap.offsets
    @test Array(tm.positions) == structure(A_h).tmap.positions
end

@testset "build_transpose_map contract on general C" begin
    # For C > 1 the map must list each column's entries in ascending storage position
    # (the determinism anchor) with rows aligned to positions.
    S = sparse(
        [1, 1, 2, 3, 3, 4, 5, 5], [2, 3, 1, 2, 4, 4, 1, 3],
        Float64.(1:8), 5, 4,
    )
    for C in (2, 4)
        A = SellMatrix(S, Val(C))
        St = structure(A)
        tm = St.tmap
        for j in 1:size(A, 2)
            rng_j = Int(tm.offsets[j]):(Int(tm.offsets[j + 1]) - 1)
            @test issorted(tm.positions[rng_j])
            # rows agree with a dense reconstruction
            for p in rng_j
                i = Int(tm.rows[p])
                @test St.colind[Int(tm.positions[p])] == j
                @test S[i, j] == parent(A)[Int(tm.positions[p])]
            end
        end
    end
end
