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
using KernelAbstractions: CPU
using JLArrays  # JuliaGPU reference backend: device semantics (no scalar indexing) on CPU
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

    @test W isa RBF.StencilWeights{Float64, Matrix{Float64}, Matrix{Int32}}
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
end

@testset "StencilWeights 3-arg constructor is device-safe" begin
    k, N_eval, N_data = 3, 6, 6
    idx = Int32[mod1(i + l, N_data) for l in 1:k, i in 1:N_eval]
    vals = reshape(collect(1.0:(k * N_eval)), k, N_eval)
    W_cpu = RBF.StencilWeights(vals, idx, N_data)

    # A device idx must not be scalar-indexed: the transpose map is built from a host
    # copy and moved back, so the whole struct lives on the device backend.
    W_dev = RBF.StencilWeights(JLArray(vals), JLArray(idx), N_data)
    @test W_dev.vals isa JLArray{Float64, 2}
    @test W_dev.idx isa JLArray{Int32, 2}
    @test W_dev.tmap.offsets isa JLArray{Int32, 1}
    @test W_dev.tmap.positions isa JLArray{Int32, 1}
    @test Array(W_dev.tmap.offsets) == W_cpu.tmap.offsets
    @test Array(W_dev.tmap.positions) == W_cpu.tmap.positions

    # Mixed-backend vals/idx would crash in the apply kernels — rejected at construction
    @test_throws ArgumentError RBF.StencilWeights(JLArray(vals), idx, N_data)
    @test_throws ArgumentError RBF.StencilWeights(vals, JLArray(idx), N_data)

    # Out-of-range neighbor indices are still rejected on the device path
    bad_idx = JLArray(Int32[N_data + 1;; 1;; 1])
    @test_throws ArgumentError RBF.StencilWeights(JLArray(ones(1, 3)), bad_idx, N_data)
end
