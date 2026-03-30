using RadialBasisFunctions
using SparseArrays: SparseVector
using StaticArraysCore
using Statistics
using HaltonSequences
using Test

include("../test_utils.jl")

# f(x,y) = sin(x) * cos(y)
# ∂²f/∂x∂y = -cos(x) * sin(y)
# ∂²f/∂x² = -sin(x) * cos(y)
# ∂²f/∂y² = -sin(x) * cos(y)
f(x) = sin(x[1]) * cos(x[2])
d2f_dxdy(x) = -cos(x[1]) * sin(x[2])
d2f_dxx(x) = -sin(x[1]) * cos(x[2])
d2f_dyy(x) = -sin(x[1]) * cos(x[2])

N = 10_000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

@testset "Mixed Partial ∂²f/∂x∂y" begin
    @testset "Polyharmonic Splines" begin
        ∂²xy = mixed_partial(x, 1, 2, PHS(3; poly_deg = 4))
        @test mean_percent_error(∂²xy(y), d2f_dxdy.(x)) < 10
    end

    @testset "Inverse Multiquadrics" begin
        ∂²xy = mixed_partial(x, 1, 2, IMQ(1; poly_deg = 4))
        @test mean_percent_error(∂²xy(y), d2f_dxdy.(x)) < 10
    end

    @testset "Gaussian" begin
        ∂²xy = mixed_partial(x, 1, 2, Gaussian(1; poly_deg = 4))
        @test mean_percent_error(∂²xy(y), d2f_dxdy.(x)) < 10
    end
end

@testset "Symmetry: ∂²f/∂x∂y == ∂²f/∂y∂x" begin
    ∂²xy = mixed_partial(x, 1, 2, PHS(3; poly_deg = 4))
    ∂²yx = mixed_partial(x, 2, 1, PHS(3; poly_deg = 4))
    @test ∂²xy(y) ≈ ∂²yx(y) atol = 1.0e-10
end

@testset "Degenerates to second partial when dim1 == dim2" begin
    ∂²xx_mixed = mixed_partial(x, 1, 1, PHS(3; poly_deg = 4))
    ∂²xx_partial = partial(x, 2, 1, PHS(3; poly_deg = 4))
    @test ∂²xx_mixed(y) ≈ ∂²xx_partial(y) atol = 1.0e-10
end

@testset "Different evaluation points" begin
    eval_pts = SVector{2}.(HaltonPoint(2)[(N + 1):(N + 100)])
    ∂²xy = mixed_partial(x, eval_pts, 1, 2, PHS(3; poly_deg = 4))
    @test mean_percent_error(∂²xy(y), d2f_dxdy.(eval_pts)) < 10
end

@testset "@operator macro ∂(i,j)" begin
    op = custom(x, @operator(∂(1, 2)); basis = PHS(3; poly_deg = 4))
    @test mean_percent_error(op(y), d2f_dxdy.(x)) < 10
end

@testset "Printing" begin
    op = MixedPartial(1, 2)
    @test RadialBasisFunctions.print_op(op) == "∂²f/∂x1∂x2"
end
