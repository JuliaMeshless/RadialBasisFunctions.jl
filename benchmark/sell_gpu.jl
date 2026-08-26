#=
Manual GPU orientation benchmark: SELL-1 (stencil-major, the CPU layout) vs SELL-32
(slot-major coalesced, the cuSPARSE layout) device matvec and adjoint. NOT part of
SUITE — AirspeedVelocity CI is CPU-only; this needs real CUDA hardware.

Run from the repo root on a CUDA machine with CUDA.jl available next to this package:

    julia --project=. --startup-file=no -e 'using Pkg; Pkg.add("CUDA")'   # once
    julia --project=. --startup-file=no benchmark/sell_gpu.jl

Expected: SELL-32 wins the forward matvec on GPU (warp-coalesced reads; SELL-1 strides
by k), while CPU results (benchmark/benchmarks.jl "eval (SELL-1/32)") show the
opposite — that disagreement is the orientation axis doing its job.
=#

using RadialBasisFunctions
using RadialBasisFunctions.EllSparse
using Adapt
using BenchmarkTools
using CUDA
using Random: MersenneTwister

CUDA.functional() || error("CUDA is not functional on this machine")

rng = MersenneTwister(2026)
N = 100_000
k = 50

# Uniform random stencil structure shaped like an RBF operator's weights
idx = Int32.(reduce(hcat, [rand(rng, 1:N, k) for _ in 1:N]))
vals = randn(rng, k, N)
A1 = SellMatrix(vals, idx, N)
A32 = reslice(A1, Val(32))

A1_d = adapt(CuArray, A1)
A32_d = adapt(CuArray, reslice(A1, Val(32)))
x_d = CuArray(randn(rng, N))
y_d = CUDA.zeros(Float64, N)
v_d = CuArray(randn(rng, N))
z_d = CUDA.zeros(Float64, N)

using LinearAlgebra: mul!

println("device: ", CUDA.name(CUDA.device()))
println("N = $N, k = $k, slice heights 1 vs 32\n")

for (label, bench) in (
        ("matvec  SELL-1 ", @benchmarkable CUDA.@sync mul!($y_d, $A1_d, $x_d)),
        ("matvec  SELL-32", @benchmarkable CUDA.@sync mul!($y_d, $A32_d, $x_d)),
        ("adjoint SELL-1 ", @benchmarkable CUDA.@sync mul!($z_d, $A1_d', $v_d)),
        ("adjoint SELL-32", @benchmarkable CUDA.@sync mul!($z_d, $A32_d', $v_d)),
    )
    t = run(bench; seconds = 5)
    println(label, ": ", BenchmarkTools.prettytime(minimum(t).time))
end

# Correctness cross-check while we're here: the resliced adjoint must be bitwise-equal
# (order-preserving remap), the forward ≈ (different slot traversal order).
y1 = Array(A1_d * x_d)
y32 = Array(A32_d * x_d)
z1 = Array(A1_d' * v_d)
z32 = Array(A32_d' * v_d)
@assert y1 ≈ y32
@assert z1 == z32
println("\ncross-checks passed (forward ≈, adjoint bitwise ==)")
