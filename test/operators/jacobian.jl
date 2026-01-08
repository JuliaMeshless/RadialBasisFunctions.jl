using RadialBasisFunctions
using SparseArrays: SparseVector
using StaticArraysCore
using Statistics
using HaltonSequences
using Test

include("../test_utils.jl")

N = 1000
points = SVector{2}.(HaltonPoint(2)[1:N])

@testset "Jacobian - Scalar field (equivalent to gradient)" begin
    # f(x,y) = sin(x) + cos(y)
    # ∂f/∂x = cos(x), ∂f/∂y = -sin(y)
    u = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))
    op = jacobian(points, PHS(3; poly_deg = 2))
    J = op(u)

    @test J isa Matrix
    @test size(J) == (N, 2)
    @test mean_percent_error(J[:, 1], cos.(getindex.(points, 1))) < 10
    @test mean_percent_error(J[:, 2], -sin.(getindex.(points, 2))) < 10
end

@testset "Jacobian - Vector field" begin
    # u = [sin(x), cos(y)]
    # J = [[cos(x), 0], [0, -sin(y)]]
    # J[:, 1, 1] = ∂u₁/∂x = cos(x)
    # J[:, 1, 2] = ∂u₁/∂y = 0
    # J[:, 2, 1] = ∂u₂/∂x = 0
    # J[:, 2, 2] = ∂u₂/∂y = -sin(y)
    u1 = sin.(getindex.(points, 1))
    u2 = cos.(getindex.(points, 2))
    u = hcat(u1, u2)

    op = jacobian(points, PHS(3; poly_deg = 2))
    J = op(u)

    @test J isa Array{<:Any, 3}
    @test size(J) == (N, 2, 2)

    # ∂u₁/∂x = cos(x)
    @test mean_percent_error(J[:, 1, 1], cos.(getindex.(points, 1))) < 10
    # ∂u₁/∂y ≈ 0
    @test maximum(abs.(J[:, 1, 2])) < 0.1
    # ∂u₂/∂x ≈ 0
    @test maximum(abs.(J[:, 2, 1])) < 0.1
    # ∂u₂/∂y = -sin(y)
    @test mean_percent_error(J[:, 2, 2], -sin.(getindex.(points, 2))) < 10
end

@testset "Jacobian - Coupled vector field" begin
    # u = [x*y, x^2 + y^2]
    # J = [[y, x], [2x, 2y]]
    u1 = getindex.(points, 1) .* getindex.(points, 2)
    u2 = getindex.(points, 1) .^ 2 .+ getindex.(points, 2) .^ 2
    u = hcat(u1, u2)

    op = jacobian(points, PHS(3; poly_deg = 2))
    J = op(u)

    @test size(J) == (N, 2, 2)

    # ∂u₁/∂x = y
    @test mean_percent_error(J[:, 1, 1], getindex.(points, 2)) < 10
    # ∂u₁/∂y = x
    @test mean_percent_error(J[:, 1, 2], getindex.(points, 1)) < 10
    # ∂u₂/∂x = 2x
    @test mean_percent_error(J[:, 2, 1], 2 .* getindex.(points, 1)) < 10
    # ∂u₂/∂y = 2y
    @test mean_percent_error(J[:, 2, 2], 2 .* getindex.(points, 2)) < 10
end

@testset "Jacobian in-place - Scalar" begin
    u = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))
    op = jacobian(points, PHS(3; poly_deg = 2))
    out = Matrix{Float64}(undef, N, 2)
    op(out, u)

    @test mean_percent_error(out[:, 1], cos.(getindex.(points, 1))) < 10
    @test mean_percent_error(out[:, 2], -sin.(getindex.(points, 2))) < 10
end

@testset "Jacobian in-place - Vector field" begin
    u1 = sin.(getindex.(points, 1))
    u2 = cos.(getindex.(points, 2))
    u = hcat(u1, u2)

    op = jacobian(points, PHS(3; poly_deg = 2))
    out = Array{Float64, 3}(undef, N, 2, 2)
    op(out, u)

    @test mean_percent_error(out[:, 1, 1], cos.(getindex.(points, 1))) < 10
    @test maximum(abs.(out[:, 1, 2])) < 0.1
    @test maximum(abs.(out[:, 2, 1])) < 0.1
    @test mean_percent_error(out[:, 2, 2], -sin.(getindex.(points, 2))) < 10
end

@testset "Jacobian convenience function" begin
    # One-shot jacobian creation and evaluation
    u = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))
    J = jacobian(points, u)

    @test J isa Matrix
    @test size(J) == (N, 2)
    @test mean_percent_error(J[:, 1], cos.(getindex.(points, 1))) < 10
    @test mean_percent_error(J[:, 2], -sin.(getindex.(points, 2))) < 10
end

@testset "Jacobian with different evaluation points" begin
    eval_points = SVector{2}.(HaltonPoint(2)[(N + 1):(N + 100)])
    u = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))

    op = jacobian(points, eval_points, PHS(3; poly_deg = 2))
    J = op(u)

    @test size(J) == (100, 2)
    @test mean_percent_error(J[:, 1], cos.(getindex.(eval_points, 1))) < 10
    @test mean_percent_error(J[:, 2], -sin.(getindex.(eval_points, 2))) < 10
end

@testset "3D Jacobian" begin
    # 3D points
    points_3d = SVector{3}.(HaltonPoint(3)[1:500])

    # f(x,y,z) = x*y + y*z + z*x
    # ∂f/∂x = y + z, ∂f/∂y = x + z, ∂f/∂z = y + x
    u =
        getindex.(points_3d, 1) .* getindex.(points_3d, 2) .+
        getindex.(points_3d, 2) .* getindex.(points_3d, 3) .+
        getindex.(points_3d, 3) .* getindex.(points_3d, 1)

    op = jacobian(points_3d, PHS(3; poly_deg = 2))
    J = op(u)

    @test J isa Matrix
    @test size(J) == (500, 3)

    expected_dx = getindex.(points_3d, 2) .+ getindex.(points_3d, 3)
    expected_dy = getindex.(points_3d, 1) .+ getindex.(points_3d, 3)
    expected_dz = getindex.(points_3d, 2) .+ getindex.(points_3d, 1)

    @test mean_percent_error(J[:, 1], expected_dx) < 10
    @test mean_percent_error(J[:, 2], expected_dy) < 10
    @test mean_percent_error(J[:, 3], expected_dz) < 10
end

@testset "Single eval point - Scalar field returns Vector" begin
    # Single evaluation point
    eval_pt = [SVector{2}(0.5, 0.5)]
    u = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))

    op = jacobian(points, eval_pt, PHS(3; poly_deg = 2))

    # Weights should be SparseVectors, not matrices
    @test op.weights[1] isa SparseVector
    @test op.weights[2] isa SparseVector

    J = op(u)

    # Result should be a Vector (size D), not a 1×D Matrix
    @test J isa Vector
    @test size(J) == (2,)
    @test length(J) == 2

    # Check accuracy
    @test abs(J[1] - cos(0.5)) < 0.1
    @test abs(J[2] - (-sin(0.5))) < 0.1
end

@testset "Single eval point - Vector field returns Matrix" begin
    # Single evaluation point
    eval_pt = [SVector{2}(0.5, 0.5)]
    # u = [x*y, x² + y²]
    u1 = getindex.(points, 1) .* getindex.(points, 2)
    u2 = getindex.(points, 1) .^ 2 .+ getindex.(points, 2) .^ 2
    u = hcat(u1, u2)

    op = jacobian(points, eval_pt, PHS(3; poly_deg = 2))
    J = op(u)

    # Result should be a D_in × D Matrix, not 1×D_in×D tensor
    @test J isa Matrix
    @test size(J) == (2, 2)

    # Expected Jacobian at (0.5, 0.5):
    # J = [[y, x], [2x, 2y]] = [[0.5, 0.5], [1.0, 1.0]]
    @test abs(J[1, 1] - 0.5) < 0.1  # ∂u₁/∂x = y
    @test abs(J[1, 2] - 0.5) < 0.1  # ∂u₁/∂y = x
    @test abs(J[2, 1] - 1.0) < 0.1  # ∂u₂/∂x = 2x
    @test abs(J[2, 2] - 1.0) < 0.1  # ∂u₂/∂y = 2y
end
