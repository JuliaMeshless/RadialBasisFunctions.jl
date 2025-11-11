using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences
using LinearAlgebra

include("../test_utils.jl")

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
df_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
df_dy(x) = 2 * cos(2 * x[2])
d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
d2f_dyy(x) = -4 * sin(2 * x[2])

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "Single Direction" begin
    v = SVector(2.0, 1.0)
    v /= norm(v)
    ∇v = directional(x, v, PHS3(2))
    exact = map(x -> SVector(df_dx(x), df_dy(x)) ⋅ v, x)
    @test mean_percent_error(∇v(y), exact) < 10
end

@testset "Direction Vector for Each Data Center" begin
    v = map(1:length(x)) do i
        v = SVector{2}(rand(2))
        return v /= norm(v)
    end
    ∇v = directional(x, v, PHS3(2))
    exact = map((x, vv) -> SVector(df_dx(x), df_dy(x)) ⋅ vv, x, v)
    @test mean_percent_error(∇v(y), exact) < 10
end

@testset "Different Evaluation Points" begin
    x2 = SVector{2}.(HaltonPoint(2)[1:N])
    v = map(1:length(x2)) do i
        v = SVector{2}(rand(2))
        return v /= norm(v)
    end
    ∇v = directional(x, x2, v, PHS3(2))
    exact = map((x, vv) -> SVector(df_dx(x), df_dy(x)) ⋅ vv, x2, v)
    @test mean_percent_error(∇v(y), exact) < 10
end

@testset "Incompatible Geometrical  Vector" begin
    # wrong geometrical vector dimension
    v = SVector(2.0, 1.0, 0.5)
    v /= norm(v)
    @test_throws DomainError directional(x, v)

    # number of geometrical vectors don't match number of points when using a different
    # geometrical vector for each point
    v = map(1:(length(x) - 1)) do i
        v = SVector{2}(rand(2))
        return v /= norm(v)
    end
    @test_throws DomainError directional(x, v)
end

@testset "Printing" begin
    ∇v = Directional{2}(SVector(1.0, 1.0))
    @test RadialBasisFunctions.print_op(∇v) == "Directional Derivative (∇f⋅v)"
end
