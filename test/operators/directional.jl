using RadialBasisFunctions
using StaticArrays
using Statistics
using Random
using LinearAlgebra

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
x = map(x -> SVector{2}(x .+ (Δ / 5 .* rand(2))), structured_points)
y = f.(x)

@testset "Single Direction" begin
    v = SVector(2.0, 1.0)
    v /= norm(v)
    ∇v = directional(x, v, PHS3(2))
    exact = map(x -> SVector(df_dx(x), df_dy(x)) ⋅ v, x)
    @test mean_percent_error(∇v(y), exact) < 5
end

@testset "Direction Vector for Each Data Center" begin
    v = map(1:length(x)) do i
        v = SVector{2}(rand(2))
        return v /= norm(v)
    end
    ∇v = directional(x, v, PHS3(2))
    exact = map((x, vv) -> SVector(df_dx(x), df_dy(x)) ⋅ vv, x, v)
    @test mean_percent_error(∇v(y), exact) < 5
end

@testset "Different Evaluation Points" begin
    x2 = map(x -> SVector{2}(rand(MersenneTwister(x), 2)), 1:N)
    v = map(1:length(x2)) do i
        v = SVector{2}(rand(2))
        return v /= norm(v)
    end
    ∇v = directional(x, x2, v, PHS3(2))
    exact = map((x, vv) -> SVector(df_dx(x), df_dy(x)) ⋅ vv, x2, v)
    @test mean_percent_error(∇v(y), exact) < 5
end
