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

@testset "Default poly_deg=2 differentiates its own polynomial space exactly" begin
    # Regression: the hand-coded monomial evaluators (2D/3D, degree 2) order terms
    # differently from the generic multiexponents pipeline `_∂mixed` used to rely on,
    # which made mixed partials silently wrong at the default degree (f = xy → 0).
    x2 = SVector{2}.(HaltonPoint(2)[1:500])
    fxy(p) = p[1] * p[2]
    @test maximum(abs, mixed_partial(x2, 1, 2)(fxy.(x2)) .- 1) < 1.0e-8
    @test maximum(abs, mixed_partial(x2, 2, 1)(fxy.(x2)) .- 1) < 1.0e-8
    # @operator path shares MixedPartial
    op = custom(x2, @operator(∂(1, 2)))
    @test maximum(abs, op(fxy.(x2)) .- 1) < 1.0e-8
    # 3D: every mixed pair at the default degree
    x3 = SVector{3}.(HaltonPoint(3)[1:800])
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        g(p) = p[i] * p[j]
        @test maximum(abs, mixed_partial(x3, i, j)(g.(x3)) .- 1) < 1.0e-6
        @test maximum(abs, mixed_partial(x3, j, i)(g.(x3)) .- 1) < 1.0e-6
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
    ∂²xy = mixed_partial(x, 1, 2; eval_points = eval_pts, basis = PHS(3; poly_deg = 4))
    @test mean_percent_error(∂²xy(y), d2f_dxdy.(eval_pts)) < 10
end

@testset "@operator macro ∂(i,j)" begin
    op = custom(x, @operator(∂(1, 2)); basis = PHS(3; poly_deg = 4))
    @test mean_percent_error(op(y), d2f_dxdy.(x)) < 10
end

@testset "One-shot convenience" begin
    result = mixed_partial(x, 1, 2, y; basis = PHS(3; poly_deg = 4))
    @test mean_percent_error(result, d2f_dxdy.(x)) < 10
end

@testset "Printing" begin
    op = MixedPartial(1, 2)
    @test RadialBasisFunctions.print_op(op) == "∂²f/∂x1∂x2"
end
