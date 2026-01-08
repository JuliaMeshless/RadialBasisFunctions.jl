using RadialBasisFunctions
using StaticArraysCore
using Statistics
using HaltonSequences

include("../test_utils.jl")

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

x2 = map(x -> SVector{2}(rand(2)), 1:100)

@testset "Positional Basis Constructor" begin
    r = regrid(x, x2, PHS(3; poly_deg=2))
    @test mean_percent_error(r(y), f.(x2)) < 0.1
end

@testset "Keyword Constructor" begin
    # Test regridding.jl lines 39-41: primary keyword constructor
    r_kw = regrid(x, x2)  # Uses default basis
    @test r_kw isa RadialBasisOperator
    @test mean_percent_error(r_kw(y), f.(x2)) < 0.1

    # Test with explicit keyword basis
    r_kw2 = regrid(x, x2; basis=PHS(5; poly_deg=3))
    @test r_kw2 isa RadialBasisOperator
end

@testset "Printing" begin
    r = regrid(x, x2, PHS(3; poly_deg=2))
    @test RadialBasisFunctions.print_op(r.ℒ) == "regrid"
end
