using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using Random: MersenneTwister

rng = MersenneTwister(234)

include("../test_utils.jl")

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
df_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
df_dy(x) = 2 * cos(2 * x[2])
d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
d2f_dyy(x) = -4 * sin(2 * x[2])

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "First Derivative Partials" begin
    @testset "Polyharmonic Splines" begin
        ∂x = partial(x, 1, 1, PHS(3; poly_deg = 2))
        ∂y = partial(x, 1, 2, PHS(3; poly_deg = 2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 10
        @test mean_percent_error(∂y(y), df_dy.(x)) < 10
    end

    @testset "Inverse Multiquadrics" begin
        ∂x = partial(x, 1, 1, IMQ(1; poly_deg = 2))
        ∂y = partial(x, 1, 2, IMQ(1; poly_deg = 2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 10
        @test mean_percent_error(∂y(y), df_dy.(x)) < 10
    end

    @testset "Gaussian" begin
        ∂x = partial(x, 1, 1, Gaussian(1; poly_deg = 2))
        ∂y = partial(x, 1, 2, Gaussian(1; poly_deg = 2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 10
        @test mean_percent_error(∂y(y), df_dy.(x)) < 10
    end
end

@testset "Second Derivative Partials" begin
    @testset "Polyharmonic Splines" begin
        ∂2x = partial(x, 2, 1, PHS(3; poly_deg = 4))
        ∂2y = partial(x, 2, 2, PHS(3; poly_deg = 4))
        @test mean_percent_error(∂2x(y), d2f_dxx.(x)) < 10
        @test mean_percent_error(∂2y(y), d2f_dyy.(x)) < 10
    end

    @testset "Inverse Multiquadrics" begin
        ∂2x = partial(x, 2, 1, IMQ(1; poly_deg = 4))
        ∂2y = partial(x, 2, 2, IMQ(1; poly_deg = 4))
        @test mean_percent_error(∂2x(y), d2f_dxx.(x)) < 10
        @test mean_percent_error(∂2y(y), d2f_dyy.(x)) < 10
    end

    @testset "Gaussian" begin
        ∂2x = partial(x, 2, 1, Gaussian(1; poly_deg = 4))
        ∂2y = partial(x, 2, 2, Gaussian(1; poly_deg = 4))
        @test mean_percent_error(∂2x(y), d2f_dxx.(x)) < 10
        @test mean_percent_error(∂2y(y), d2f_dyy.(x)) < 10
    end
end

@testset "Different Evaluation Points" begin
    x2 = map(x -> SVector{2}(rand(rng, 2)), 1:100)
    ∂x = partial(x, x2, 1, 1, PHS(3; poly_deg = 2))
    ∂y = partial(x, x2, 1, 2, PHS(3; poly_deg = 2))
    @test mean_percent_error(∂x(y), df_dx.(x2)) < 10
    @test mean_percent_error(∂y(y), df_dy.(x2)) < 10
end

@testset "Higher Order Derivatives" begin
    @test_throws ArgumentError partial(x, 3, 1)
end

@testset "Printing" begin
    ∂ = Partial(1, 2)
    @test RadialBasisFunctions.print_op(∂) == "∂ⁿf/∂xᵢ (n = 1, i = 2)"
end
