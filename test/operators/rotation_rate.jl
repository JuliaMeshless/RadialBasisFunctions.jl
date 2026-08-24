using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using LinearAlgebra

include("../test_utils.jl")

@testset "2D Rotation Rate (solid body rotation)" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])

    # u = [-x₂, x₁] (solid body rotation) → ω₁₂ = ½(∂u₁/∂x₂ − ∂u₂/∂x₁) = ½(-1-1) = -1
    u = hcat(-getindex.(x, 2), getindex.(x, 1))
    ω_op = rotation_rate(x)
    ω = ω_op(u)

    @test size(ω) == (N, 2, 2)
    @test mean_percent_error(ω[:, 1, 2], fill(-1.0, N)) < 10
    # Anti-symmetry
    @test ω[:, 1, 2] ≈ -ω[:, 2, 1]
    # Diagonal is zero
    @test all(abs.(ω[:, 1, 1]) .< 1.0e-10)
    @test all(abs.(ω[:, 2, 2]) .< 1.0e-10)
end

@testset "2D Rotation Rate (opposite sign)" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])

    # u = [x₂, -x₁] → ω₁₂ = ½(1 - (-1)) = 1
    u = hcat(getindex.(x, 2), -getindex.(x, 1))
    ω_op = rotation_rate(x)
    ω = ω_op(u)

    @test mean_percent_error(ω[:, 1, 2], ones(N)) < 10
    @test ω[:, 1, 2] ≈ -ω[:, 2, 1]
end

@testset "2D Irrotational Field" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])

    # u = [x₁, x₂] (pure expansion, no rotation) → ω₁₂ = ½(0-0) = 0
    u = hcat(getindex.(x, 1), getindex.(x, 2))
    ω_op = rotation_rate(x)
    ω = ω_op(u)

    @test all(abs.(ω[:, 1, 2]) .< 0.1)
end

@testset "3D Rotation Rate" begin
    N = 10_000
    x = SVector{3}.(HaltonPoint(3)[1:N])

    # u = [x₂-x₃, x₃-x₁, x₁-x₂]
    # ∂u₁/∂x₂=1, ∂u₁/∂x₃=-1, ∂u₂/∂x₁=-1, ∂u₂/∂x₃=1, ∂u₃/∂x₁=1, ∂u₃/∂x₂=-1
    # ω₁₂ = ½(1-(-1)) = 1, ω₁₃ = ½(-1-1) = -1, ω₂₃ = ½(1-(-1)) = 1
    x1, x2, x3 = getindex.(x, 1), getindex.(x, 2), getindex.(x, 3)
    u = hcat(x2 .- x3, x3 .- x1, x1 .- x2)
    ω_op = rotation_rate(x)
    ω = ω_op(u)

    @test size(ω) == (N, 3, 3)
    @test mean_percent_error(ω[:, 1, 2], ones(N)) < 10
    @test mean_percent_error(ω[:, 1, 3], fill(-1.0, N)) < 10
    @test mean_percent_error(ω[:, 2, 3], ones(N)) < 10
    # Anti-symmetry
    @test ω[:, 1, 2] ≈ -ω[:, 2, 1]
    @test ω[:, 1, 3] ≈ -ω[:, 3, 1]
    @test ω[:, 2, 3] ≈ -ω[:, 3, 2]
    # Diagonal zero
    @test all(abs.(ω[:, 1, 1]) .< 1.0e-10)
    @test all(abs.(ω[:, 2, 2]) .< 1.0e-10)
    @test all(abs.(ω[:, 3, 3]) .< 1.0e-10)
end

@testset "Different Eval Points" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    x2 = SVector{2}.(HaltonPoint(2)[(N + 1):(2N)])

    u = hcat(-getindex.(x, 2), getindex.(x, 1))
    ω_op = rotation_rate(x; eval_points = x2)
    ω = ω_op(u)

    @test size(ω) == (N, 2, 2)
    @test mean_percent_error(ω[:, 1, 2], fill(-1.0, N)) < 10
end

@testset "One-Shot" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(-getindex.(x, 2), getindex.(x, 1))

    ω = rotation_rate(x, u)
    @test mean_percent_error(ω[:, 1, 2], fill(-1.0, N)) < 10
end

@testset "Scalar Input Error" begin
    x = SVector{2}.(HaltonPoint(2)[1:100])
    ω_op = rotation_rate(x)
    @test_throws ArgumentError ω_op(ones(100))
end

@testset "In-Place" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(-getindex.(x, 2), getindex.(x, 1))

    ω_op = rotation_rate(x)
    y = similar(u, N, 2, 2)
    ω_op(y, u)
    @test mean_percent_error(y[:, 1, 2], fill(-1.0, N)) < 10
    @test y[:, 1, 2] ≈ -y[:, 2, 1]
end

@testset "Single Eval Point" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    eval_pt = [SVector{2}(0.5, 0.5)]

    u = hcat(-getindex.(x, 2), getindex.(x, 1))

    ω_op = rotation_rate(x; eval_points = eval_pt)
    @test ω_op.weights[1] isa StencilWeights
    result = ω_op(u)
    @test result isa Array{Float64, 3}
    @test size(result) == (1, 2, 2)
    @test abs(result[1, 1, 2] - (-1.0)) < 0.1
    @test result[1, 1, 2] ≈ -result[1, 2, 1]
end

@testset "Printing" begin
    @test RadialBasisFunctions.print_op(RotationRate{2}()) == "Rotation Rate (ω = ½(∇u − (∇u)ᵀ))"
end
