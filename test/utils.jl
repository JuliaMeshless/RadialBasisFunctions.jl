using Test
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF

# 12-point 2D domain with 2 boundary points at (0,0) and (1,0), mirrors test/hermite.jl
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

@testset "Hermite basis-support guard" begin
    data, is_boundary, normals = create_2d_domain()
    neumann = (is_boundary = is_boundary, bc = [Neumann(), Neumann()], normals = normals)
    dirichlet = (is_boundary = is_boundary, bc = [Dirichlet(), Dirichlet()], normals = normals)

    # Neumann/Robin BCs need the 3-arg normal form — only PHS implements it (issue #136)
    @test_throws ArgumentError RadialBasisOperator(
        Laplacian(), data; basis = IMQ(1.0; poly_deg = 2), hermite = neumann
    )
    @test_throws ArgumentError RadialBasisOperator(
        Laplacian(), data; basis = Gaussian(1.0; poly_deg = 2), hermite = neumann
    )
    robin = (is_boundary = is_boundary, bc = [Robin(1.0, 1.0), Robin(1.0, 1.0)], normals = normals)
    @test_throws ArgumentError RadialBasisOperator(
        Laplacian(), data; basis = IMQ(1.0; poly_deg = 2), hermite = robin
    )

    # Tuple-valued actions (gradient family) route through the same guard
    @test_throws ArgumentError RadialBasisOperator(
        Jacobian{2}(), data; basis = IMQ(1.0; poly_deg = 2), hermite = neumann
    )

    # PHS supports the normal form — build must succeed and produce finite values
    op = RadialBasisOperator(
        Laplacian(), data; basis = PHS(3; poly_deg = 2), hermite = neumann
    )
    u = [x[2]^2 for x in data]  # ∂u/∂n = 0 on both boundaries
    @test all(isfinite, op(u))

    op_jac = RadialBasisOperator(
        Jacobian{2}(), data; basis = PHS(3; poly_deg = 2), hermite = neumann
    )
    @test op_jac isa RadialBasisOperator

    # Dirichlet-only never evaluates the normal form — IMQ/Gaussian must not trip the guard
    op_imq = RadialBasisOperator(
        Laplacian(), data; basis = IMQ(1.0; poly_deg = 2), hermite = dirichlet
    )
    @test all(isfinite, op_imq(u))
    op_gauss = RadialBasisOperator(
        Laplacian(), data; basis = Gaussian(1.0; poly_deg = 2), hermite = dirichlet
    )
    @test all(isfinite, op_gauss(u))
end

@testset "boundary input validation" begin
    data, is_boundary, normals = create_2d_domain()
    bc = [Neumann(), Neumann()]

    # is_boundary must flag every data point
    @test_throws DimensionMismatch RadialBasisOperator(
        Laplacian(), data;
        hermite = (is_boundary = is_boundary[1:5], bc = bc, normals = normals),
    )
    # one bc per boundary point
    @test_throws DimensionMismatch RadialBasisOperator(
        Laplacian(), data;
        hermite = (is_boundary = is_boundary, bc = bc[1:1], normals = normals),
    )
    # one normal per boundary point
    @test_throws DimensionMismatch RadialBasisOperator(
        Laplacian(), data;
        hermite = (is_boundary = is_boundary, bc = bc, normals = normals[1:1]),
    )
end

@testset "HermiteStencilData length validation" begin
    stencil_data = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]
    bcs = [Internal(), Internal(), Internal()]
    stencil_normals = [zeros(2) for _ in 1:3]

    hd = HermiteStencilData(stencil_data, fill(false, 3), bcs, stencil_normals)
    @test hd isa HermiteStencilData{Float64}

    @test_throws DimensionMismatch HermiteStencilData(
        stencil_data, fill(false, 2), bcs, stencil_normals
    )
    @test_throws DimensionMismatch HermiteStencilData(
        stencil_data, fill(false, 3), bcs[1:2], stencil_normals
    )
    @test_throws DimensionMismatch HermiteStencilData(
        stencil_data, fill(false, 3), bcs, stencil_normals[1:2]
    )
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
