using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using LinearAlgebra
using SparseArrays: SparseVector

include("../test_utils.jl")

@testset "2D Strain Rate (constant)" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])

    # u = [x₂, x₁] → ε₁₁=0, ε₂₂=0, ε₁₂=ε₂₁=1
    u = hcat(getindex.(x, 2), getindex.(x, 1))
    ε_op = strain_rate(x)
    ε = ε_op(u)

    @test size(ε) == (N, 2, 2)
    @test mean_percent_error(ε[:, 1, 2], ones(N)) < 10
    @test mean_percent_error(ε[:, 2, 1], ones(N)) < 10
    # Diagonal should be near zero — use absolute tolerance
    @test all(abs.(ε[:, 1, 1]) .< 0.1)
    @test all(abs.(ε[:, 2, 2]) .< 0.1)
    # Symmetry
    @test ε[:, 1, 2] ≈ ε[:, 2, 1]
end

@testset "2D Strain Rate (non-trivial)" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])

    # u = [x₁², x₁·x₂] → ε₁₁=2x₁, ε₂₂=x₁, ε₁₂=½(2x₁·x₂ + x₂)=½x₂(2x₁+1)
    # Actually: ∂u₁/∂x₁=2x₁, ∂u₁/∂x₂=0, ∂u₂/∂x₁=x₂, ∂u₂/∂x₂=x₁
    # ε₁₁=2x₁, ε₂₂=x₁, ε₁₂=½(0+x₂)=x₂/2
    x1 = getindex.(x, 1)
    x2 = getindex.(x, 2)
    u = hcat(x1 .^ 2, x1 .* x2)
    exact_11 = 2 .* x1
    exact_22 = x1
    exact_12 = x2 ./ 2

    ε_op = strain_rate(x)
    ε = ε_op(u)

    @test mean_percent_error(ε[:, 1, 1], exact_11) < 10
    @test mean_percent_error(ε[:, 2, 2], exact_22) < 10
    @test mean_percent_error(ε[:, 1, 2], exact_12) < 10
    @test ε[:, 1, 2] ≈ ε[:, 2, 1]
end

@testset "3D Strain Rate" begin
    N = 10_000
    x = SVector{3}.(HaltonPoint(3)[1:N])

    # u = [x₂, x₃, x₁] → ε₁₂=½, ε₁₃=½, ε₂₃=½, diag=0
    u = hcat(getindex.(x, 2), getindex.(x, 3), getindex.(x, 1))
    ε_op = strain_rate(x)
    ε = ε_op(u)

    @test size(ε) == (N, 3, 3)
    half = fill(0.5, N)
    @test mean_percent_error(ε[:, 1, 2], half) < 10
    @test mean_percent_error(ε[:, 1, 3], half) < 10
    @test mean_percent_error(ε[:, 2, 3], half) < 10
    # Symmetry
    @test ε[:, 1, 2] ≈ ε[:, 2, 1]
    @test ε[:, 1, 3] ≈ ε[:, 3, 1]
    @test ε[:, 2, 3] ≈ ε[:, 3, 2]
end

@testset "Different Eval Points" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    x2 = SVector{2}.(HaltonPoint(2)[(N + 1):(2N)])

    u = hcat(getindex.(x, 2), getindex.(x, 1))
    ε_op = strain_rate(x; eval_points = x2)
    ε = ε_op(u)

    @test size(ε) == (N, 2, 2)
    @test mean_percent_error(ε[:, 1, 2], ones(N)) < 10
end

@testset "One-Shot" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(getindex.(x, 2), getindex.(x, 1))

    ε = strain_rate(x, u)
    @test mean_percent_error(ε[:, 1, 2], ones(N)) < 10
end

@testset "Scalar Input Error" begin
    x = SVector{2}.(HaltonPoint(2)[1:100])
    ε_op = strain_rate(x)
    @test_throws ArgumentError ε_op(ones(100))
end

@testset "In-Place" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(getindex.(x, 2), getindex.(x, 1))

    ε_op = strain_rate(x)
    y = similar(u, N, 2, 2)
    ε_op(y, u)
    @test mean_percent_error(y[:, 1, 2], ones(N)) < 10
    @test y[:, 1, 2] ≈ y[:, 2, 1]
end

@testset "Single Eval Point" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    eval_pt = [SVector{2}(0.5, 0.5)]

    u = hcat(getindex.(x, 2), getindex.(x, 1))

    ε_op = strain_rate(x; eval_points = eval_pt)
    @test ε_op.weights[1] isa SparseVector
    result = ε_op(u)
    @test result isa Matrix
    @test size(result) == (2, 2)
    @test abs(result[1, 2] - 1.0) < 0.1
    @test result[1, 2] ≈ result[2, 1]
end

@testset "Printing" begin
    @test RadialBasisFunctions.print_op(StrainRate{2}()) == "Strain Rate (ε = ½(∇u + (∇u)ᵀ))"
end
