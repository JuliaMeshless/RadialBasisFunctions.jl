#=
Tests for the backend-aware ELL allocation seam and the un-pinned AD cache
constructors. These guard the "no hardcoded Matrix" invariant: weight buffers
are allocated through the KernelAbstractions backend (a CPU backend must remain
behavior-neutral, returning plain Arrays), and the AD cache/workspace structs
infer their array type parameters from the actual buffers instead of pinning
Matrix{T}/Vector{T} at the call site.
=#

using Test
using StaticArraysCore
using LinearAlgebra: mul!
using KernelAbstractions: CPU
using JLArrays  # JuliaGPU reference backend: device semantics (no scalar indexing) on CPU
import KernelAbstractions
# JLArrays runs kernels synchronously but omits synchronize(::JLBackend);
# a no-op shim lets the generic device path run under test.
KernelAbstractions.synchronize(::JLArrays.JLBackend) = nothing
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "allocate_ell backend seam" begin
    for TD in (Float64, Float32)
        k, N_eval, num_ops = 5, 12, 3
        vals_list, idx = RBF.allocate_ell(CPU(), TD, k, N_eval, num_ops)
        @test vals_list isa Vector{Matrix{TD}}
        @test length(vals_list) == num_ops
        @test all(v -> size(v) == (k, N_eval), vals_list)
        @test idx isa Matrix{Int32}
        @test size(idx) == (k, N_eval)
    end
end

@testset "AD cache constructors infer array parameters" begin
    # Small deterministic 2D grid problem
    side = range(0.0, 1.0; length = 5)
    data = [SVector(x, y) for x in side for y in side]
    basis = PHS(3; poly_deg = 2)
    k = 12
    adjl = RBF.find_neighbors(data, data, k)
    ℒ = RBF.Laplacian()
    ℒrbf = ℒ(basis)
    mon = RBF.MonomialBasis(2, basis.poly_deg)
    ℒmon = ℒ(mon)

    W, cache = RBF._forward_with_cache(data, data, adjl, basis, ℒrbf, ℒmon, mon, typeof(ℒ))

    @test W isa RBF.StencilWeights{Float64, <:RBF.EllSparse.SellMatrix{Float64}}
    @test parent(W) isa Matrix{Float64}
    @test RBF._neighbor_matrix(W) isa AbstractMatrix{Int32}
    # eltype of the stencil-cache vector is derived from the actual weight buffer
    @test cache.stencil_caches isa Vector{<:RBF.StencilForwardCache{Float64, Matrix{Float64}}}

    ws = RBF.BackwardWorkspace(cache, data)
    @test ws isa RBF.BackwardWorkspace{
        Float64, SVector{2, Float64}, Matrix{Float64}, Vector{Float64},
    }
    n = cache.k + cache.nmon
    @test size(ws.ΔA) == (n, n)
    @test size(ws.Δb) == (n, cache.num_ops)
    @test length(ws.Δlocal_data) == cache.k
    @test length(ws.Δeval_pt) == 2

    # Ragged stencils are rejected on the AD forward path, mirroring the primal build
    ragged = [copy(neighbors) for neighbors in adjl]
    ragged[1] = ragged[1][1:(end - 1)]
    @test_throws ArgumentError RBF._forward_with_cache(
        data, data, ragged, basis, ℒrbf, ℒmon, mon, typeof(ℒ)
    )
end

@testset "StencilWeights 3-arg constructor is device-safe" begin
    k, N_eval, N_data = 3, 6, 6
    idx = Int32[mod1(i + l, N_data) for l in 1:k, i in 1:N_eval]
    vals = reshape(collect(1.0:(k * N_eval)), k, N_eval)
    W_cpu = RBF.StencilWeights(vals, idx, N_data)

    # A device idx must not be scalar-indexed: the transpose map is built from a host
    # copy and moved back, so the whole struct lives on the device backend.
    W_dev = RBF.StencilWeights(JLArray(vals), JLArray(idx), N_data)
    @test parent(W_dev) isa JLArray{Float64, 2}
    @test RBF.EllSparse.structure(W_dev.ell).colind isa JLArray{Int32, 1}
    tm_dev = RBF.EllSparse.structure(W_dev.ell).tmap
    tm_cpu = RBF.EllSparse.structure(W_cpu.ell).tmap
    @test tm_dev.offsets isa JLArray{Int32, 1}
    @test tm_dev.positions isa JLArray{Int32, 1}
    @test Array(tm_dev.offsets) == tm_cpu.offsets
    @test Array(tm_dev.positions) == tm_cpu.positions

    # Mixed-backend vals/idx would crash in the apply kernels — rejected at construction
    @test_throws ArgumentError RBF.StencilWeights(JLArray(vals), idx, N_data)
    @test_throws ArgumentError RBF.StencilWeights(vals, JLArray(idx), N_data)

    # Out-of-range neighbor indices are still rejected on the device path
    bad_idx = JLArray(Int32[N_data + 1;; 1;; 1])
    @test_throws ArgumentError RBF.StencilWeights(JLArray(ones(1, 3)), bad_idx, N_data)
end

@testset "Apply kernels on a device backend" begin
    # The CPU suite only reaches the ::CPU fast-path methods; JLArrays' KA backend
    # dispatches to the generic @kernel launches — forward matvec, adjoint gather, and
    # their multi-column variants — without needing a GPU.
    k, N_eval, N_data = 3, 8, 7
    idx = Int32[mod1(i + l, N_data) for l in 1:k, i in 1:N_eval]
    vals = reshape(collect(1.0:(k * N_eval)), k, N_eval)
    W_cpu = RBF.StencilWeights(vals, idx, N_data)
    W_dev = RBF.StencilWeights(JLArray(vals), JLArray(idx), N_data)

    x = collect(1.0:N_data)
    X = [i + 10.0 * j for i in 1:N_data, j in 1:2]
    v = collect(1.0:N_eval)
    V = [i + 10.0 * j for i in 1:N_eval, j in 1:2]

    # Forward matvec: destination allocated on the weights' backend
    y_dev = W_dev * JLArray(x)
    @test y_dev isa JLArray{Float64, 1}
    @test Array(y_dev) ≈ W_cpu * x

    # β = 0 must overwrite a NaN destination; β ≠ 0 must read through (mul! contract)
    @test Array(mul!(JLArray(fill(NaN, N_eval)), W_dev, JLArray(x))) ≈ W_cpu * x
    fwd_scaled = 2.0 .* (W_cpu * x) .+ 3.0
    @test Array(mul!(JLArray(ones(N_eval)), W_dev, JLArray(x), 2.0, 3.0)) ≈ fwd_scaled

    # Multi-column forward: all columns launch, one synchronize
    @test Array(W_dev * JLArray(X)) ≈ W_cpu * X

    # Adjoint gather through the device-resident transpose map
    @test Array(W_dev' * JLArray(v)) ≈ W_cpu' * v
    @test Array(W_dev' * JLArray(V)) ≈ W_cpu' * V
    adj_scaled = 2.0 .* (W_cpu' * v) .+ 3.0
    @test Array(mul!(JLArray(ones(N_data)), W_dev', JLArray(v), 2.0, 3.0)) ≈ adj_scaled
end
