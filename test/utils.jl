using Test
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF

# 12-point 2D domain used by the neighbor/reorder tests below.
function create_2d_domain()
    data = [
        SVector(0.0, 0.0),   # boundary
        SVector(0.15, 0.1),
        SVector(0.2, 0.25),
        SVector(0.3, 0.15),
        SVector(0.4, 0.3),
        SVector(0.5, 0.2),
        SVector(0.6, 0.35),
        SVector(0.7, 0.25),
        SVector(0.75, 0.15),
        SVector(0.85, 0.3),
        SVector(0.9, 0.2),
        SVector(1.0, 0.0),   # boundary
    ]
    is_boundary = vcat(true, fill(false, 10), true)
    normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]
    return data, is_boundary, normals
end

@testset "find_neighbors" begin
    points = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(2.0, 2.0)]

    adjl = find_neighbors(points, 2)
    @test length(adjl) == length(points)
    @test all(length.(adjl) .== 2)
    @test adjl[1][1] == 1  # sorted output → each point is its own nearest neighbor
    @test adjl[1][2] in (2, 3)  # (1,0) and (0,1) are equidistant from (0,0)
    @test adjl[4][2] != 1  # (0,0) is the farthest point from (2,2)

    eval_points = [SVector(0.9, 0.1)]
    adjl_eval = find_neighbors(points, eval_points, 3)
    @test length(adjl_eval) == length(eval_points)
    @test length(adjl_eval[1]) == 3
    @test adjl_eval[1][1] == 2  # nearest data point to (0.9, 0.1) is (1,0)
end

@testset "autoselect_k" begin
    points = [SVector(0.01 * i, 0.02 * i^2) for i in 1:50]

    # Bayona: min(N, max(2*binomial(m+d, d), 2d+1)) with d=2
    @test RBF.autoselect_k(points, PHS(3; poly_deg = 2)) == 12  # 2*binomial(4,2)
    @test RBF.autoselect_k(points, PHS(3; poly_deg = 1)) == 6   # 2*binomial(3,2)
    @test RBF.autoselect_k(points, PHS(3; poly_deg = 0)) == 5   # 2d+1 dominates
    @test RBF.autoselect_k(points[1:5], PHS(3; poly_deg = 2)) == 5  # clamped to N
end

@testset "check_poly_deg" begin
    @test RBF.check_poly_deg(-1) === nothing
    @test RBF.check_poly_deg(2) === nothing
    @test_throws ArgumentError RBF.check_poly_deg(-2)
end

@testset "reorder_points!" begin
    original = [SVector(float(i), 0.5 * i) for i in 10:-1:1]
    k = 3

    # 2-arg exported form (was a MethodError before the dispatch fix)
    points = copy(original)
    perm = reorder_points!(points, k)
    @test perm isa Vector{Int}
    @test isperm(perm)
    @test points == original[perm]

    # 3-arg form with precomputed adjacency
    points = copy(original)
    adjl = find_neighbors(points, k)
    perm3 = reorder_points!(points, adjl, k)
    @test isperm(perm3)
    @test points == original[perm3]
end

@testset "Regrid action" begin
    basis = PHS(3; poly_deg = 2)
    @test Regrid()(basis) === basis
    mon = MonomialBasis(2, 2)
    @test Regrid()(mon) === mon

    # calling with data now builds an operator (was a dispatch ambiguity MethodError)
    data, _, _ = create_2d_domain()
    @test Regrid()(data) isa RadialBasisOperator

    # end-to-end regrid: poly_deg=2 reproduces a linear field exactly
    targets = [SVector(0.3, 0.2), SVector(0.6, 0.25), SVector(0.8, 0.2)]
    rg = regrid(data, targets)
    field = [1.0 + 2.0 * x[1] - 0.5 * x[2] for x in data]
    expected = [1.0 + 2.0 * x[1] - 0.5 * x[2] for x in targets]
    @test rg(field) ≈ expected atol = 1.0e-8
end
