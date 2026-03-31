using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using LinearAlgebra

include("../test_utils.jl")

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
df_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
df_dy(x) = 2 * cos(2 * x[2])

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "Unit Normals" begin
    normals = [normalize(SVector(1.0, 1.0)) for _ in 1:N]
    ∂ₙ = normal_derivative(x, normals)
    n̂ = normalize(SVector(1.0, 1.0))
    exact = map(p -> SVector(df_dx(p), df_dy(p)) ⋅ n̂, x)
    @test mean_percent_error(∂ₙ(y), exact) < 10
end

@testset "Non-Unit Normals Are Normalized" begin
    normals_unnorm = [SVector(3.0, 4.0) for _ in 1:N]
    normals_unit = [normalize(SVector(3.0, 4.0)) for _ in 1:N]
    ∂ₙ_unnorm = normal_derivative(x, normals_unnorm)
    ∂ₙ_unit = normal_derivative(x, normals_unit)
    @test ∂ₙ_unnorm(y) ≈ ∂ₙ_unit(y)
end

@testset "Spatially Varying Normals" begin
    normals = [normalize(p) for p in x]
    ∂ₙ = normal_derivative(x, normals)
    exact = map((p, n) -> SVector(df_dx(p), df_dy(p)) ⋅ n, x, normalize.(x))
    @test mean_percent_error(∂ₙ(y), exact) < 10
end

@testset "Different Eval Points" begin
    x2 = SVector{2}.(HaltonPoint(2)[(N + 1):(2N)])
    normals = [normalize(p) for p in x2]
    ∂ₙ = normal_derivative(x, normals; eval_points = x2)
    exact = map((p, n) -> SVector(df_dx(p), df_dy(p)) ⋅ n, x2, normals)
    @test mean_percent_error(∂ₙ(y), exact) < 10
end

@testset "One-Shot" begin
    normals = [normalize(SVector(1.0, 0.0)) for _ in 1:N]
    result = normal_derivative(x, normals, y)
    exact = df_dx.(x)
    @test mean_percent_error(result, exact) < 10
end
