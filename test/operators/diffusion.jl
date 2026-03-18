using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences

include("../test_utils.jl")

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
d2f_dyy(x) = -4 * sin(2 * x[2])

basis = PHS(5; poly_deg=3)

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "Scalar Diffusivity" begin
    κ = 2.5
    op = diffusion(x, κ; basis=basis)
    exact = κ .* (d2f_dxx.(x) .+ d2f_dyy.(x))
    @test mean_percent_error(op(y), exact) < 5
end

@testset "Vector Diffusivity" begin
    κ = [3.0, 0.5]
    op = diffusion(x, κ; basis=basis)
    exact = κ[1] .* d2f_dxx.(x) .+ κ[2] .* d2f_dyy.(x)
    @test mean_percent_error(op(y), exact) < 5
end

@testset "Macro Syntax" begin
    κ = [3.0, 0.5]
    op = custom(x, @operator(∇ ⋅ (κ * ∇)); rank=0, basis=basis)
    exact = κ[1] .* d2f_dxx.(x) .+ κ[2] .* d2f_dyy.(x)
    @test mean_percent_error(op(y), exact) < 5
end

@testset "Macro Composition" begin
    κ = 2.0
    k² = 1.0
    op = custom(x, @operator(∇ ⋅ (κ * ∇) + k² * f); rank=0, basis=basis)
    exact = κ .* (d2f_dxx.(x) .+ d2f_dyy.(x)) .+ k² .* f.(x)
    @test mean_percent_error(op(y), exact) < 5
end

@testset "Dimension Mismatch" begin
    κ_bad = [1.0, 2.0, 3.0]
    @test_throws DimensionMismatch diffusion(x, κ_bad)
end
