using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using LinearAlgebra
using SparseArrays: SparseVector

include("../test_utils.jl")

@testset "2D Curl" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])

    # u = (-x₂, x₁) → curl(u) = ∂u₂/∂x₁ - ∂u₁/∂x₂ = 1 - (-1) = 2
    u = hcat(-getindex.(x, 2), getindex.(x, 1))
    exact = fill(2.0, N)

    curl_op = curl(x)
    @test mean_percent_error(curl_op(u), exact) < 10
end

@testset "3D Curl" begin
    N = 10_000
    x = SVector{3}.(HaltonPoint(3)[1:N])

    # u = (-x₂, x₁, 0) → curl(u) = (0, 0, 2)
    u = hcat(-getindex.(x, 2), getindex.(x, 1), zeros(N))
    exact = hcat(zeros(N), zeros(N), fill(2.0, N))

    curl_op = curl(x)
    result = curl_op(u)
    @test mean_percent_error(view(result, :, 3), view(exact, :, 3)) < 10
    @test all(abs.(view(result, :, 1)) .< 0.1)
    @test all(abs.(view(result, :, 2)) .< 0.1)
end

@testset "Different Eval Points 2D" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    x2 = SVector{2}.(HaltonPoint(2)[(N + 1):(2N)])

    u = hcat(-getindex.(x, 2), getindex.(x, 1))
    exact = fill(2.0, N)

    curl_op = curl(x; eval_points = x2)
    @test mean_percent_error(curl_op(u), exact) < 10
end

@testset "One-Shot 2D" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(-getindex.(x, 2), getindex.(x, 1))
    exact = fill(2.0, N)

    result = curl(x, u)
    @test mean_percent_error(result, exact) < 10
end

@testset "Invalid Dimension" begin
    @test_throws ArgumentError Curl{1}()
    @test_throws ArgumentError Curl{4}()
end

@testset "Scalar Input Error" begin
    x = SVector{2}.(HaltonPoint(2)[1:100])
    curl_op = curl(x)
    @test_throws ArgumentError curl_op(ones(100))
end

@testset "In-Place 2D" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(-getindex.(x, 2), getindex.(x, 1))
    exact = fill(2.0, N)

    curl_op = curl(x)
    y = similar(exact)
    curl_op(y, u)
    @test mean_percent_error(y, exact) < 10
end

@testset "In-Place 3D" begin
    N = 10_000
    x = SVector{3}.(HaltonPoint(3)[1:N])
    u = hcat(-getindex.(x, 2), getindex.(x, 1), zeros(N))

    curl_op = curl(x)
    y = similar(u, length(x), 3)
    curl_op(y, u)
    @test all(abs.(view(y, :, 1)) .< 0.1)
    @test all(abs.(view(y, :, 2)) .< 0.1)
    @test mean_percent_error(view(y, :, 3), fill(2.0, N)) < 10
end

@testset "Single Eval Point 2D" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    eval_pt = [SVector{2}(0.5, 0.5)]

    # u = (-x₂, x₁) → curl(u) = 2
    u = hcat(-getindex.(x, 2), getindex.(x, 1))

    curl_op = curl(x; eval_points=eval_pt)
    @test curl_op.weights[1] isa SparseVector
    result = curl_op(u)
    @test result isa Number
    @test abs(result - 2.0) < 0.1
end

@testset "Single Eval Point 3D" begin
    N = 10_000
    x = SVector{3}.(HaltonPoint(3)[1:N])
    eval_pt = [SVector{3}(0.5, 0.5, 0.5)]

    # u = (-x₂, x₁, 0) → curl(u) = (0, 0, 2)
    u = hcat(-getindex.(x, 2), getindex.(x, 1), zeros(N))

    curl_op = curl(x; eval_points=eval_pt)
    @test curl_op.weights[1] isa SparseVector
    result = curl_op(u)
    @test result isa SVector{3}
    @test abs(result[1]) < 0.1
    @test abs(result[2]) < 0.1
    @test abs(result[3] - 2.0) < 0.1
end

@testset "Printing" begin
    @test RadialBasisFunctions.print_op(Curl{2}()) == "Curl 2D (∇×)"
    @test RadialBasisFunctions.print_op(Curl{3}()) == "Curl 3D (∇×)"
end
