using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences

include("../test_utils.jl")

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
df_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
df_dy(x) = 2 * cos(2 * x[2])
d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
d2f_dyy(x) = -4 * sin(2 * x[2])

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "First Derivative gradients" begin
    ∇ = gradient(x, PHS(3; poly_deg = 2))
    ∇y = ∇(y)
    @test ∇y isa Matrix
    @test size(∇y) == (N, 2)
    @test mean_percent_error(∇y[:, 1], df_dx.(x)) < 10
    @test mean_percent_error(∇y[:, 2], df_dy.(x)) < 10
end

@testset "Different Evaluation Points" begin
    x2 = map(x -> SVector{2}(rand(2)), 1:100)
    ∇ = gradient(x, x2, PHS(3; poly_deg = 2))
    ∇y = ∇(y)
    @test ∇y isa Matrix
    @test size(∇y) == (100, 2)
    @test mean_percent_error(∇y[:, 1], df_dx.(x2)) < 10
    @test mean_percent_error(∇y[:, 2], df_dy.(x2)) < 10
end

@testset "In-place gradient evaluation" begin
    ∇ = gradient(x, PHS(3; poly_deg = 2))
    out = Matrix{Float64}(undef, N, 2)
    ∇(out, y)
    @test mean_percent_error(out[:, 1], df_dx.(x)) < 10
    @test mean_percent_error(out[:, 2], df_dy.(x)) < 10
end

@testset "Printing" begin
    J = Jacobian{2}()
    @test RadialBasisFunctions.print_op(J) == "Jacobian (J)"
end
