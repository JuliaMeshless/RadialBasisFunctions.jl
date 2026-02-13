using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using Random: MersenneTwister

rng = MersenneTwister(678)

include("../test_utils.jl")

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
df_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
df_dy(x) = 2 * cos(2 * x[2])
d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
d2f_dyy(x) = -4 * sin(2 * x[2])

Δ = 1.0e-4
N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "First Derivative Partials" begin
    @testset "Polyharmonic Splines" begin
        ∂x = ∂virtual(x, 1, Δ, PHS(3; poly_deg = 2))
        ∂y = ∂virtual(x, 2, Δ, PHS(3; poly_deg = 2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 10
        @test mean_percent_error(∂y(y), df_dy.(x)) < 10
    end

    @testset "Inverse Multiquadrics" begin
        ∂x = ∂virtual(x, 1, Δ, IMQ(1; poly_deg = 2))
        ∂y = ∂virtual(x, 2, Δ, IMQ(1; poly_deg = 2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 10
        @test mean_percent_error(∂y(y), df_dy.(x)) < 10
    end

    @testset "Gaussian" begin
        ∂x = ∂virtual(x, 1, Δ, Gaussian(1; poly_deg = 2))
        ∂y = ∂virtual(x, 2, Δ, Gaussian(1; poly_deg = 2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 10
        @test mean_percent_error(∂y(y), df_dy.(x)) < 10
    end
end

@testset "Different Evaluation Points" begin
    x2 = map(x -> SVector{2}(rand(rng, 2)), 1:100)
    ∂x = ∂virtual(x, x2, 1, Δ, PHS(3; poly_deg = 2))
    ∂y = ∂virtual(x, x2, 2, Δ, PHS(3; poly_deg = 2))
    @test mean_percent_error(∂x(y), df_dx.(x2)) < 10
    @test mean_percent_error(∂y(y), df_dy.(x2)) < 10
end
