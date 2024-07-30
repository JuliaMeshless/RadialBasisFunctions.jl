using RadialBasisFunctions
using StaticArrays
using Statistics
using Random

rsme(test, correct) = sqrt(sum((test - correct) .^ 2) / sum(correct .^ 2))
mean_percent_error(test, correct) = mean(abs.((test .- correct) ./ correct)) * 100

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
df_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
df_dy(x) = 2 * cos(2 * x[2])
d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
d2f_dyy(x) = -4 * sin(2 * x[2])

N = 100
Δ = 1 / (N - 1)
points = 0:Δ:1
structured_points = ((x, y) for x in points for y in points)
x = map(x -> SVector{2}(x .+ (Δ / 10 .* rand(2))), structured_points)
y = f.(x)

@testset "First Derivative Partials" begin
    @testset "Polyharmonic Splines" begin
        ∂x = ∂virtual(x, 1, Δ / 5, PHS(3; poly_deg=2))
        ∂y = ∂virtual(x, 2, Δ / 5, PHS(3; poly_deg=2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 5
        @test mean_percent_error(∂y(y), df_dy.(x)) < 5
    end

    @testset "Inverse Multiquadrics" begin
        ∂x = ∂virtual(x, 1, Δ / 5, IMQ(1; poly_deg=2))
        ∂y = ∂virtual(x, 2, Δ / 5, IMQ(1; poly_deg=2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 5
        @test mean_percent_error(∂y(y), df_dy.(x)) < 5
    end

    @testset "Gaussian" begin
        ∂x = ∂virtual(x, 1, Δ / 5, Gaussian(1; poly_deg=2))
        ∂y = ∂virtual(x, 2, Δ / 5, Gaussian(1; poly_deg=2))
        @test mean_percent_error(∂x(y), df_dx.(x)) < 5
        @test mean_percent_error(∂y(y), df_dy.(x)) < 5
    end
end

@testset "Different Evaluation Points" begin
    x2 = map(x -> SVector{2}(rand(2)), 1:100)
    ∂x = ∂virtual(x, x2, 1, Δ / 5, PHS(3; poly_deg=2))
    ∂y = ∂virtual(x, x2, 2, Δ / 5, PHS(3; poly_deg=2))
    @test mean_percent_error(∂x(y), df_dx.(x2)) < 5
    @test mean_percent_error(∂y(y), df_dy.(x2)) < 5
end
