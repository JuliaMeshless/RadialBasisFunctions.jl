using RadialBasisFunctions
import RadialBasisFunctions as RBF
using Adapt
using KernelAbstractions
using KernelAbstractions: CPU
using LinearAlgebra
using SparseArrays
using Random: MersenneTwister, randperm
using StaticArraysCore
using Test

rng = MersenneTwister(123)

# Hand-built ELL fixture: k = 4, N_eval = 6, N_data = 8. Column 5 mimics a Dirichlet
# boundary row — weight 1 at slot 1, zero pads all sharing the same index — to exercise
# duplicate-index combining in getindex/Matrix/sparse.
k, n_eval, n_data = 4, 6, 8
vals = randn(rng, k, n_eval)
idx = Int32.(reduce(hcat, [randperm(rng, n_data)[1:k] for _ in 1:n_eval]))
vals[:, 5] .= [1.0, 0.0, 0.0, 0.0]
idx[:, 5] .= Int32(5)
W = StencilWeights(vals, idx, n_data)
A = Matrix(W)

@testset "Construction and Array Interface" begin
    @test size(W) == (n_eval, n_data)
    @test eltype(W) == Float64
    @test parent(W) === vals

    # getindex sums duplicate-index slots, matching sparse() combine semantics
    S = sparse(W)
    @test all(W[i, j] == S[i, j] for i in 1:n_eval, j in 1:n_data)
    @test W[5, 5] == 1.0

    @test_throws ArgumentError W[1, 1] = 2.0
    @test_throws DimensionMismatch StencilWeights(vals, idx[1:(k - 1), :], n_data)
    @test_throws ArgumentError StencilWeights(vals, idx, Int64(typemax(Int32)) + 1)
    # Out-of-range neighbor indices are rejected (the apply kernels index with @inbounds)
    @test_throws ArgumentError StencilWeights(vals, fill(Int32(n_data + 1), k, n_eval), n_data)
    @test_throws ArgumentError StencilWeights(vals, fill(Int32(0), k, n_eval), n_data)
end

@testset "Conversions" begin
    S = sparse(W)
    @test S isa SparseMatrixCSC{Float64, Int}
    @test size(S) == (n_eval, n_data)
    @test Matrix(S) == A
    @test SparseMatrixCSC(W) == S
    # Dirichlet-style column collapses to a single identity entry
    @test nnz(S[5, :]) == 1
    @test S[5, 5] == 1.0
end

@testset "Matvec vs Sparse Reference" begin
    S = sparse(W)
    x = randn(rng, n_data)

    @test W * x ≈ S * x
    y = fill(NaN, n_eval)
    @test mul!(y, W, x) ≈ S * x                    # β = 0 must overwrite NaN-filled y
    y2 = randn(rng, n_eval)
    expected = 2.5 .* (S * x) .+ 0.5 .* y2
    @test mul!(copy(y2), W, x, 2.5, 0.5) ≈ expected

    X = randn(rng, n_data, 3)
    @test W * X ≈ S * X
    Y = fill(NaN, n_eval, 3)
    @test mul!(Y, W, X) ≈ S * X
    @test_throws DimensionMismatch mul!(zeros(n_eval, 3), W, randn(rng, n_data, 2))

    # Float32 values are preserved through the kernel
    W32 = StencilWeights(Float32.(vals), idx, n_data)
    x32 = randn(rng, Float32, n_data)
    @test W32 * x32 isa Vector{Float32}
    @test W32 * x32 ≈ Float32.(Matrix(W32) * x32)
end

@testset "Adjoint Apply" begin
    S = sparse(W)
    v = randn(rng, n_eval)

    @test W' * v ≈ S' * v
    y = fill(NaN, n_data)
    @test mul!(y, W', v) ≈ S' * v
    y2 = randn(rng, n_data)
    @test mul!(copy(y2), W', v, 2.0, 3.0) ≈ 2.0 .* (S' * v) .+ 3.0 .* y2
    # SubArray inputs (the AD pullbacks feed matrix column views)
    V = randn(rng, n_eval, 2)
    @test W' * view(V, :, 1) ≈ S' * V[:, 1]

    # Matrix right-hand sides and the transpose spelling (real weights: same scatter)
    @test W' * V ≈ S' * V
    @test transpose(W) * v ≈ transpose(S) * v
    @test transpose(W) * V ≈ transpose(S) * V

    # In-place transpose spellings delegate to the adjoint scatter
    yt = randn(rng, n_data)
    @test mul!(copy(yt), transpose(W), v, 2.0, 3.0) ≈ 2.0 .* (S' * v) .+ 3.0 .* yt
    @test mul!(fill(NaN, n_data), transpose(W), v) ≈ S' * v

    # Matrix cotangent accumulation used by the AD eval rules on multi-column fields
    @test RBF.accumulate_eval_pullback!(zeros(n_data, 2), W, V) ≈ S' * V

    # Generic fallback for sparse weights (VirtualPartial) accumulates, never overwrites
    Δx = randn(rng, n_data)
    @test RBF.accumulate_eval_pullback!(copy(Δx), S, v) ≈ Δx .+ S' * v

    # A sparse input must still produce a DENSE product (destination keyed off the
    # weights' backing array, not the input)
    Ssq = sparse(randn(rng, n_data, 3))
    @test W * Ssq isa Matrix
    @test W * Ssq ≈ Matrix(W) * Matrix(Ssq)
end

@testset "Algebra" begin
    B = StencilWeights(randn(rng, k, n_eval), idx, n_data)   # shared idx object
    C = StencilWeights(copy(B.vals), copy(idx), n_data)      # equal idx content

    @test Matrix(W + B) ≈ A .+ Matrix(B)
    @test Matrix(W - C) ≈ A .- Matrix(C)
    @test (W + B) isa StencilWeights
    @test Matrix(-W) ≈ -A
    @test Matrix(2.5 * W) ≈ 2.5 .* A
    @test Matrix(W * 2.5) ≈ 2.5 .* A
    @test Matrix(W / 4) ≈ A ./ 4

    d = randn(rng, n_eval)
    @test Matrix(Diagonal(d) * W) ≈ Diagonal(d) * A
    @test (Diagonal(d) * W).idx === W.idx
    @test_throws DimensionMismatch Diagonal(randn(rng, n_eval + 1)) * W

    other_idx = Int32.(reduce(hcat, [randperm(rng, n_data)[1:k] for _ in 1:n_eval]))
    D = StencilWeights(randn(rng, k, n_eval), other_idx, n_data)
    @test_throws ArgumentError W + D
    @test_throws ArgumentError W - D

    # Mixing with sparse matrices (the VirtualPartial storage) stays sparse, never dense
    Ssp = sparse(D)
    @test W + Ssp isa SparseMatrixCSC
    @test Matrix(W + Ssp) ≈ A .+ Matrix(Ssp)
    @test Matrix(Ssp - W) ≈ Matrix(Ssp) .- A
    @test Ssp + W isa SparseMatrixCSC
    @test Matrix(Ssp + W) ≈ Matrix(Ssp) .+ A
    @test W - Ssp isa SparseMatrixCSC
    @test Matrix(W - Ssp) ≈ A .- Matrix(Ssp)
end

@testset "Equality, Copy, and copyto!" begin
    W2 = copy(W)
    @test W2 isa StencilWeights
    @test W2 == W
    @test W2 ≈ W
    @test W2 !== W && W2.vals !== W.vals

    parent(W2) .= 0.0
    @test W2 != W
    copyto!(W2, W)
    @test W2 == W

    small = StencilWeights(randn(rng, k, n_eval - 1), idx[:, 1:(n_eval - 1)], n_data)
    @test_throws DimensionMismatch copyto!(small, W)

    # copyto! transfers values only — a source with different stencil structure is refused
    # rather than rewriting the frozen (possibly aliased) idx matrix
    reordered_idx = reverse(idx; dims = 1)
    other = StencilWeights(randn(rng, k, n_eval), reordered_idx, n_data)
    @test_throws ArgumentError copyto!(W2, other)

    # Different stencil structures holding the same logical matrix compare via sparse
    Wr = StencilWeights(reverse(vals; dims = 1), reordered_idx, n_data)
    @test Wr == W
    @test Wr ≈ W
    Wr2 = StencilWeights(2 .* reverse(vals; dims = 1), reordered_idx, n_data)
    @test Wr2 != W
    @test !(Wr2 ≈ W)
end

@testset "Backslash Guardrail" begin
    # Square, diagonally-dominant system: W \ b must match the sparse solve
    ks, ns = 3, 5
    sq_idx = Int32.(reduce(hcat, [circshift(1:ns, -(i - 1))[1:ks] for i in 1:ns]))
    sq_vals = randn(rng, ks, ns)
    sq_vals[1, :] .+= 10.0                          # slot 1 is the diagonal entry
    Wsq = StencilWeights(sq_vals, sq_idx, ns)
    b = randn(rng, ns)
    @test Wsq \ b ≈ sparse(Wsq) \ b
    @test Wsq \ b ≈ Matrix(Wsq) \ b
end

@testset "Threaded CPU apply paths" begin
    # The @threads branches gate on length(y) ≥ _ELL_SERIAL_CUTOFF and nthreads() > 1;
    # below that (and on single-thread runs) the serial SIMD loop handles the row. A
    # square cutoff-sized operator exercises both forward and adjoint threaded loops on
    # the multi-thread CI jobs while staying cheap enough for the single-thread ones.
    kL, nL = 3, RBF._ELL_SERIAL_CUTOFF
    idxL = Int32[mod1(i + l, nL) for l in 1:kL, i in 1:nL]
    WL = StencilWeights(randn(rng, kL, nL), idxL, nL)
    SL = sparse(WL)
    xL = randn(rng, nL)
    yL = randn(rng, nL)

    @test WL * xL ≈ SL * xL
    @test mul!(copy(yL), WL, xL, 2.0, 3.0) ≈ 2.0 .* (SL * xL) .+ 3.0 .* yL
    @test WL' * xL ≈ SL' * xL
    @test mul!(copy(yL), WL', xL, 2.0, 3.0) ≈ 2.0 .* (SL' * xL) .+ 3.0 .* yL
end

@testset "Adapt and Backend" begin
    @test KernelAbstractions.get_backend(W) isa CPU
    W_adapted = Adapt.adapt(CPU(), W)
    @test W_adapted isa StencilWeights
    @test W_adapted.vals === W.vals && W_adapted.idx === W.idx
end

# ============================================================================
# Operator integration: weights produced by the build pipeline
# ============================================================================

pts = map(_ -> SVector{2}(rand(rng, 2)), 1:50)

# Square-grid Hermite fixture: boundary = points on the unit-square edge. Normals are
# dummies — Dirichlet conditions ignore them.
n_grid = 6
h = n_grid - 1
grid = vec([SVector{2}(i / h, j / h) for i in 0:h, j in 0:h])
is_b = map(p -> p[1] == 0 || p[1] == 1 || p[2] == 0 || p[2] == 1, grid)
bcs = [RBF.Dirichlet() for _ in 1:count(is_b)]
bnormals = [SVector{2}(1.0, 0.0) for _ in 1:count(is_b)]

@testset "Operator weights are StencilWeights" begin
    op = partial(pts, 1, 1)
    @test op.weights isa StencilWeights
    @test all(Int32.(op.adjl[i]) == op.weights.idx[:, i] for i in eachindex(op.adjl))
    @test SparseMatrixCSC(op) == sparse(op)
    @test SparseMatrixCSC(op) isa SparseMatrixCSC

    grad = gradient(pts, PHS(3; poly_deg = 2))
    @test grad.weights isa NTuple{2, <:StencilWeights}
    @test grad.weights[1].idx === grad.weights[2].idx
    @test grad.weights[1].tmap === grad.weights[2].tmap
    # Tuple-weight operators can't collapse to one matrix; only sparse(op) works
    @test_throws ArgumentError SparseMatrixCSC(grad)
    @test sparse(grad) isa NTuple{2, <:SparseMatrixCSC}

    # Algebra results share the stencil structure without rebuilding the transpose map
    scaled = 2.5 * op.weights
    @test scaled.idx === op.weights.idx && scaled.tmap === op.weights.tmap
end

@testset "Dirichlet columns act as identity" begin
    op = laplacian(grid; hermite = (is_boundary = is_b, bc = bcs, normals = bnormals))
    z = randn(rng, length(grid))
    u = op(z)
    b_idx = findall(is_b)
    @test u[b_idx] == z[b_idx]

    S = sparse(op)
    @test all(nnz(S[i, :]) == 1 for i in b_idx)
    @test all(S[i, i] == 1.0 for i in b_idx)
    # ELL columns for Dirichlet points carry the identity in slot 1, zero pads below
    @test all(iszero, parent(op.weights)[2:end, b_idx])
end

@testset "Rectangular regrid weights" begin
    xe = map(_ -> SVector{2}(rand(rng, 2)), 1:20)
    r = regrid(pts, xe)
    @test size(r.weights) == (length(xe), length(pts))
    z = randn(rng, length(pts))
    @test r(z) ≈ sparse(r) * z
end

@testset "Ragged adjacency list is rejected" begin
    ragged = [collect(1:5) for _ in eachindex(pts)]
    ragged[1] = collect(1:4)
    @test_throws ArgumentError laplacian(pts; adjl = ragged)
end

@testset "Hermite requires matching eval points" begin
    @test_throws ArgumentError laplacian(
        grid;
        hermite = (is_boundary = is_b, bc = bcs, normals = bnormals),
        eval_points = grid[1:10],
    )
end
