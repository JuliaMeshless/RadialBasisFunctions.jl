using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences

rsme(test, correct) = sqrt(sum((test - correct) .^ 2) / sum(correct .^ 2))
mean_percent_error(test, correct) = mean(abs.((test .- correct) ./ correct)) * 100

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
d2f_dyy(x) = -4 * sin(2 * x[2])
∇²f(x) = d2f_dxx(x) + d2f_dyy(x)

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "Laplacian" begin
    ∇² = laplacian(x, PHS(3; poly_deg=4))
    @test mean_percent_error(∇²(y), ∇²f.(x)) < 10

    ∇² = laplacian(x, IMQ(1; poly_deg=4))
    @test mean_percent_error(∇²(y), ∇²f.(x)) < 10

    ∇² = laplacian(x, Gaussian(1; poly_deg=4))
    @test mean_percent_error(∇²(y), ∇²f.(x)) < 10
end

@testset "Different Evaluation Points" begin
    x2 = SVector{2}.(HaltonPoint(2)[1:N])
    ∇² = laplacian(x, x2, PHS(3; poly_deg=4))
    @test mean_percent_error(∇²(y), ∇²f.(x2)) < 10
end

@testset "Printing" begin
    ∇ = Laplacian()
    @test RadialBasisFunctions.print_op(∇) == "Laplacian (∇²f)"
end
