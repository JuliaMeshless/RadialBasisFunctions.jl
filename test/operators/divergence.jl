using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using LinearAlgebra

include("../test_utils.jl")

@testset "2D Divergence" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])

    # u = [sin(x₁), cos(x₂)] → div(u) = cos(x₁) - sin(x₂)
    u = hcat(sin.(getindex.(x, 1)), cos.(getindex.(x, 2)))
    exact = cos.(getindex.(x, 1)) .- sin.(getindex.(x, 2))

    div_op = divergence(x)
    @test mean_percent_error(div_op(u), exact) < 10
end

@testset "3D Divergence" begin
    N = 10_000
    x = SVector{3}.(HaltonPoint(3)[1:N])

    # u = [x₁², x₂², x₃²] → div(u) = 2x₁ + 2x₂ + 2x₃
    u = hcat(getindex.(x, 1) .^ 2, getindex.(x, 2) .^ 2, getindex.(x, 3) .^ 2)
    exact = 2 .* (getindex.(x, 1) .+ getindex.(x, 2) .+ getindex.(x, 3))

    div_op = divergence(x)
    @test mean_percent_error(div_op(u), exact) < 10
end

@testset "Different Eval Points" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    x2 = SVector{2}.(HaltonPoint(2)[(N + 1):(2N)])

    u = hcat(sin.(getindex.(x, 1)), cos.(getindex.(x, 2)))
    exact = cos.(getindex.(x2, 1)) .- sin.(getindex.(x2, 2))

    div_op = divergence(x; eval_points = x2)
    @test mean_percent_error(div_op(u), exact) < 10
end

@testset "One-Shot" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(sin.(getindex.(x, 1)), cos.(getindex.(x, 2)))
    exact = cos.(getindex.(x, 1)) .- sin.(getindex.(x, 2))

    result = divergence(x, u)
    @test mean_percent_error(result, exact) < 10
end

@testset "Scalar Input Error" begin
    x = SVector{2}.(HaltonPoint(2)[1:100])
    div_op = divergence(x)
    @test_throws ArgumentError div_op(ones(100))
end

@testset "In-Place" begin
    N = 10_000
    x = SVector{2}.(HaltonPoint(2)[1:N])
    u = hcat(sin.(getindex.(x, 1)), cos.(getindex.(x, 2)))
    exact = cos.(getindex.(x, 1)) .- sin.(getindex.(x, 2))

    div_op = divergence(x)
    y = similar(exact)
    div_op(y, u)
    @test mean_percent_error(y, exact) < 10
end

@testset "Printing" begin
    @test RadialBasisFunctions.print_op(Divergence{2}()) == "Divergence (∇⋅)"
end
