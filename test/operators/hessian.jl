using RadialBasisFunctions
using SparseArrays: SparseVector
using StaticArraysCore
using Statistics
using HaltonSequences
using Test

include("../test_utils.jl")

# f(x,y) = sin(x) * cos(y)
# H[1,1] = -sin(x)*cos(y),  H[1,2] = -cos(x)*sin(y)
# H[2,1] = -cos(x)*sin(y),  H[2,2] = -sin(x)*cos(y)
f(x) = sin(x[1]) * cos(x[2])
H11(x) = -sin(x[1]) * cos(x[2])
H12(x) = -cos(x[1]) * sin(x[2])
H22(x) = -sin(x[1]) * cos(x[2])

N = 1000
points = SVector{2}.(HaltonPoint(2)[1:N])
u = f.(points)

# Vector field data (shared across tests)
u1 = getindex.(points, 1) .* getindex.(points, 2)          # x*y
u2 = getindex.(points, 1) .^ 2 .+ getindex.(points, 2) .^ 2  # x² + y²
u_vec = hcat(u1, u2)

@testset "Hessian - Scalar field" begin
    op = hessian(points, PHS(3; poly_deg = 4))
    Hu = op(u)

    @test Hu isa Array{<:Any, 3}
    @test size(Hu) == (N, 2, 2)
    @test mean_percent_error(Hu[:, 1, 1], H11.(points)) < 10
    @test mean_percent_error(Hu[:, 1, 2], H12.(points)) < 10
    @test mean_percent_error(Hu[:, 2, 1], H12.(points)) < 10
    @test mean_percent_error(Hu[:, 2, 2], H22.(points)) < 10
end

@testset "Symmetry in output" begin
    op = hessian(points, PHS(3; poly_deg = 4))
    Hu = op(u)
    @test Hu[:, 1, 2] ≈ Hu[:, 2, 1] atol = 1.0e-12
end

@testset "Symmetric weight storage" begin
    op = hessian(points, PHS(3; poly_deg = 4))
    # D=2: should store D*(D+1)/2 = 3 weight matrices, not 4
    @test length(op.weights) == 3
end

@testset "Hessian - Vector field" begin
    op = hessian(points, PHS(3; poly_deg = 4))
    Hu = op(u_vec)

    @test Hu isa Array{<:Any, 4}
    @test size(Hu) == (N, 2, 2, 2)

    # H[u1]: all second derivs of x*y are 0 except ∂²/∂x∂y = 1
    @test maximum(abs.(Hu[:, 1, 1, 1])) < 0.1        # ∂²(xy)/∂x² = 0
    @test maximum(abs.(Hu[:, 1, 1, 2] .- 1.0)) < 0.1 # ∂²(xy)/∂x∂y = 1
    @test maximum(abs.(Hu[:, 1, 2, 2])) < 0.1        # ∂²(xy)/∂y² = 0

    # H[u2]: ∂²(x²+y²)/∂x² = 2, ∂²/∂y² = 2, cross = 0
    @test maximum(abs.(Hu[:, 2, 1, 1] .- 2.0)) < 0.1
    @test maximum(abs.(Hu[:, 2, 1, 2])) < 0.1
    @test maximum(abs.(Hu[:, 2, 2, 2] .- 2.0)) < 0.1
end

@testset "Hessian in-place - Scalar" begin
    op = hessian(points, PHS(3; poly_deg = 4))
    out = Array{Float64, 3}(undef, N, 2, 2)
    op(out, u)
    @test mean_percent_error(out[:, 1, 1], H11.(points)) < 10
    @test mean_percent_error(out[:, 1, 2], H12.(points)) < 10
end

@testset "Hessian convenience function" begin
    Hu = hessian(points, u; basis = PHS(3; poly_deg = 4))
    @test Hu isa Array{<:Any, 3}
    @test size(Hu) == (N, 2, 2)
end

@testset "Different evaluation points" begin
    eval_pts = SVector{2}.(HaltonPoint(2)[(N + 1):(N + 100)])
    op = hessian(points, eval_pts, PHS(3; poly_deg = 4))
    Hu = op(u)

    @test size(Hu) == (100, 2, 2)
    @test mean_percent_error(Hu[:, 1, 1], H11.(eval_pts)) < 10
    @test mean_percent_error(Hu[:, 1, 2], H12.(eval_pts)) < 10
end

@testset "Single eval point" begin
    eval_pt = [SVector{2}(0.5, 0.5)]
    op = hessian(points, eval_pt, PHS(3; poly_deg = 4))

    @test op.weights[1] isa SparseVector
    @test length(op.weights) == 3

    Hu = op(u)
    @test Hu isa Matrix
    @test size(Hu) == (2, 2)
    @test Hu[1, 2] ≈ Hu[2, 1] atol = 1.0e-12
    @test abs(Hu[1, 1] - H11(SVector(0.5, 0.5))) < 0.5
    @test abs(Hu[1, 2] - H12(SVector(0.5, 0.5))) < 0.5
end

@testset "3D Hessian" begin
    points_3d = SVector{3}.(HaltonPoint(3)[1:500])
    # f(x,y,z) = x² + y² + z² + x*y
    # H = [[2, 1, 0], [1, 2, 0], [0, 0, 2]]
    u_3d =
        getindex.(points_3d, 1) .^ 2 .+ getindex.(points_3d, 2) .^ 2 .+
        getindex.(points_3d, 3) .^ 2 .+ getindex.(points_3d, 1) .* getindex.(points_3d, 2)

    op = hessian(points_3d, PHS(3; poly_deg = 4))
    Hu = op(u_3d)

    @test size(Hu) == (500, 3, 3)
    # D=3: should store 6 unique weight matrices
    @test length(op.weights) == 6

    @test maximum(abs.(Hu[:, 1, 1] .- 2.0)) < 0.5
    @test maximum(abs.(Hu[:, 1, 2] .- 1.0)) < 0.5
    @test maximum(abs.(Hu[:, 1, 3])) < 0.5
    @test maximum(abs.(Hu[:, 2, 2] .- 2.0)) < 0.5
    @test maximum(abs.(Hu[:, 2, 3])) < 0.5
    @test maximum(abs.(Hu[:, 3, 3] .- 2.0)) < 0.5
    # Symmetry
    @test Hu[:, 1, 2] ≈ Hu[:, 2, 1] atol = 1.0e-12
    @test Hu[:, 1, 3] ≈ Hu[:, 3, 1] atol = 1.0e-12
    @test Hu[:, 2, 3] ≈ Hu[:, 3, 2] atol = 1.0e-12
end

@testset "Single eval point - Vector field" begin
    eval_pt = [SVector{2}(0.5, 0.5)]
    op = hessian(points, eval_pt, PHS(3; poly_deg = 4))
    Hu = op(u_vec)
    @test Hu isa Array{<:Any, 3}
    @test size(Hu) == (2, 2, 2)
    # Symmetry
    @test Hu[1, 1, 2] ≈ Hu[1, 2, 1] atol = 1.0e-12
    @test Hu[2, 1, 2] ≈ Hu[2, 2, 1] atol = 1.0e-12
end

@testset "Hessian in-place - Vector field" begin
    op = hessian(points, PHS(3; poly_deg = 4))
    out = Array{Float64, 4}(undef, N, 2, 2, 2)
    op(out, u_vec)
    @test size(out) == (N, 2, 2, 2)
    # ∂²(xy)/∂x² = 0
    @test maximum(abs.(out[:, 1, 1, 1])) < 0.1
    # ∂²(x²+y²)/∂x² = 2
    @test maximum(abs.(out[:, 2, 1, 1] .- 2.0)) < 0.1
end

@testset "Tensor field (rank-3 input)" begin
    # Build a (N, 2, 2) tensor field from functions with known Hessians
    # Channel (1,1): x*y      → H = [[0,1],[1,0]]
    # Channel (2,1): x²+y²    → H = [[2,0],[0,2]]
    # Channel (1,2): sin(x)cos(y) = f  → H already defined above
    # Channel (2,2): x*y      → H = [[0,1],[1,0]]
    u_tensor = Array{Float64, 3}(undef, N, 2, 2)
    u_tensor[:, 1, 1] .= u1
    u_tensor[:, 2, 1] .= u2
    u_tensor[:, 1, 2] .= u
    u_tensor[:, 2, 2] .= u1

    op = hessian(points, PHS(3; poly_deg = 4))
    Hu = op(u_tensor)

    @test Hu isa Array{<:Any, 5}
    @test size(Hu) == (N, 2, 2, 2, 2)

    # Channel (1,1): H(x*y) — ∂²/∂x∂y = 1, diag = 0
    @test maximum(abs.(Hu[:, 1, 1, 1, 1])) < 0.1
    @test maximum(abs.(Hu[:, 1, 1, 1, 2] .- 1.0)) < 0.1
    @test maximum(abs.(Hu[:, 1, 1, 2, 2])) < 0.1

    # Channel (2,1): H(x²+y²) — diag = 2, off-diag = 0
    @test maximum(abs.(Hu[:, 2, 1, 1, 1] .- 2.0)) < 0.1
    @test maximum(abs.(Hu[:, 2, 1, 1, 2])) < 0.1
    @test maximum(abs.(Hu[:, 2, 1, 2, 2] .- 2.0)) < 0.1

    # Symmetry in Hessian dimensions
    @test Hu[:, 1, 1, 1, 2] ≈ Hu[:, 1, 1, 2, 1] atol = 1.0e-12
    @test Hu[:, 2, 1, 1, 2] ≈ Hu[:, 2, 1, 2, 1] atol = 1.0e-12
end

@testset "Single eval point - Tensor field" begin
    eval_pt = [SVector{2}(0.5, 0.5)]
    op = hessian(points, eval_pt, PHS(3; poly_deg = 4))

    @test op.weights[1] isa SparseVector

    u_tensor = Array{Float64, 3}(undef, N, 2, 2)
    u_tensor[:, 1, 1] .= u1
    u_tensor[:, 2, 1] .= u2
    u_tensor[:, 1, 2] .= u
    u_tensor[:, 2, 2] .= u1

    Hu = op(u_tensor)

    @test Hu isa Array{<:Any, 4}
    @test size(Hu) == (2, 2, 2, 2)

    # Channel (2,1): H(x²+y²) at (0.5,0.5) → diag = 2, off-diag = 0
    @test abs(Hu[2, 1, 1, 1] - 2.0) < 0.5
    @test abs(Hu[2, 1, 1, 2]) < 0.5
    @test abs(Hu[2, 1, 2, 2] - 2.0) < 0.5

    # Symmetry
    @test Hu[1, 1, 1, 2] ≈ Hu[1, 1, 2, 1] atol = 1.0e-12
    @test Hu[2, 1, 1, 2] ≈ Hu[2, 1, 2, 1] atol = 1.0e-12
end

@testset "update_weights! rank-2" begin
    op = hessian(points, PHS(3; poly_deg = 4))
    RadialBasisFunctions.invalidate_cache!(op)
    @test !RadialBasisFunctions.is_cache_valid(op)
    result = op(u)
    @test RadialBasisFunctions.is_cache_valid(op)
    @test size(result) == (N, 2, 2)
end

@testset "Printing" begin
    @test RadialBasisFunctions.print_op(Hessian{2}()) == "Hessian (H)"
end
