using StaticArraysCore
using LinearAlgebra, LinearSolve
using SparseArrays
using Test
using Random

import RadialBasisFunctions as RBF

@testset "solve_with_Hermite (small deterministic case)" begin
    # Small configuration: 2 internal, 3 boundary (Neumann, Dirichlet, Neumann)
    node_coords = [
        SVector(1.0, 2.0),
        SVector(2.0, 1.0),
        SVector(1.5, 0.0),
        SVector(0.0, 1.0),
        SVector(2.0, 0.0),
    ]
    is_boundary = [false, false, true, true, true]
    bt_neu = RBF.BoundaryType(Float64[0.0, 1.0])
    bt_dir = RBF.BoundaryType(Float64[1.0, 0.0])
    boundary_types = [bt_neu, bt_dir, bt_neu]
    normals = [
        SVector(0.0, 1.0),  # node 3
        SVector(0.0, 0.0),  # node 4 (unused for Dirichlet)
        SVector(1.0, 0.0),  # node 5
    ]

    # Adjacency: each node + 3 others (k = 4 for internal nodes)
    adjl = [[1, 2, 3, 4], [2, 1, 3, 5], [3, 1, 2, 5], [4, 1, 3, 5], [5, 2, 3, 4]]

    # Basis & operators
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    fdata = RBF.FunctionalData(basis, mon, (RBF.∂(basis, 1),), (RBF.∂_Hermite(mon, 1),))
    region = RBF.RegionData(node_coords, is_boundary, boundary_types, normals, adjl, fdata)

    @testset "BoundaryType classification" begin
        @test RBF.is_Neumann(bt_neu)
        @test RBF.is_Dirichlet(bt_dir)
        @test !RBF.is_Robin(bt_neu)
        @test !RBF.is_Robin(bt_dir)
    end

    @testset "_preallocate_IJV_matrices" begin
        lhs, rhs = RBF._preallocate_IJV_matrices(region)
        # internal/internal: for node1 (1,2) and node2 (2,1) plus self (1,1) (2,2) => 4
        @test length(lhs.I) == 4
        @test length(lhs.J) == 4
        @test size(lhs.V) == (4, 1)
        # internal->boundary: node1 to (3,4); node2 to (3,5) => 4
        @test length(rhs.I) == 4
        @test length(rhs.J) == 4
        @test size(rhs.V) == (4, 1)
        @test all(1 .<= lhs.I .<= 2) && all(1 .<= lhs.J .<= 2)
        @test all(1 .<= rhs.J .<= 3) # boundary remapped to 1:3
    end

    @testset "_calculate_thread_offsets" begin
        lhs_off, rhs_off = RBF._calculate_thread_offsets(region, 2)
        @test length(lhs_off) == 2
        @test length(rhs_off) == 2
        @test lhs_off[1] == 1
        @test rhs_off[1] == 1
    end

    @testset "_update_stencil! (local)" begin

        # node 1
        stencil1 = RBF.StencilData(region)
        RBF._set_stencil_eval_point!(stencil1, node_coords[1])
        RBF._update_stencil!(stencil1, region, 1)
        k = length(region.adjl[1])
        @test k == 4
        #test local_adjl
        @test all(stencil1.local_adjl[i] == region.adjl[1][i] for i in 1:k)
        # local node copies
        @test all(stencil1.local_coords[i] == node_coords[region.adjl[1][i]] for i in 1:k)
        # boundary flags (nodes 3,4 boundary appear locally at positions 3,4)
        @test stencil1.is_boundary[1] == false
        @test stencil1.is_boundary[2] == false
        @test stencil1.is_boundary[3] == true
        @test stencil1.is_boundary[4] == true
        # Some weights computed
        @test any(stencil1.lhs_v .!= 0) || any(stencil1.rhs_v .!= 0)
        #test the normals
        @test stencil1.normals[1] == SVector(0.0, 0.0)
        @test stencil1.normals[2] == SVector(0.0, 0.0)
        @test stencil1.normals[3] == SVector(0.0, 1.0)
        @test stencil1.normals[4] == SVector(0.0, 0.0)
        #test again boundary normals
        @test stencil1.normals[3] ==
            region.normals[region.global_to_boundary[stencil1.local_adjl[3]]]
        @test stencil1.normals[4] ==
            region.normals[region.global_to_boundary[stencil1.local_adjl[4]]]

        # node 2 adjacency includes node 5 boundary instead of 4
        stencil2 = RBF.StencilData(region)
        RBF._set_stencil_eval_point!(stencil2, node_coords[2])
        RBF._update_stencil!(stencil2, region, 2)
        @test stencil2.local_coords[4] == node_coords[5]
        @test all(stencil2.local_coords[i] == node_coords[region.adjl[2][i]] for i in 1:k)
        @test stencil2.is_boundary[1] == false
        @test stencil2.is_boundary[2] == false
        @test stencil2.is_boundary[3] == true
        @test stencil2.is_boundary[4] == true
    end

    @testset "_return_global_matrices & full build" begin
        lhs_mat, rhs_mat = RBF._build_weights(region)
        @test size(lhs_mat) == (2, 2)
        @test size(rhs_mat) == (2, 3)
        @test nnz(lhs_mat) > 0
        @test nnz(rhs_mat) > 0
    end
end

@testset "testing _build_weights_larger_case" begin
    #square [0.1,1.1]
    Random.seed!(45)
    all_coords = [rand(SVector{2}) + SVector(0.1, 0.1) for _ in 1:20]
    #distance matrix
    # distance_matrix = [
    #     norm(all_coords[i] - all_coords[j]) for i in 1:length(all_coords),
    #     j in 1:length(all_coords)
    # ]
    # println("distance_matrix = $(display(distance_matrix))")
    # println("all_coords = $all_coords")
    is_boundary = vcat(fill(false, 15), fill(true, 5))
    boundary_types = [RBF.BoundaryType([1.0, 0.0]) for _ in 1:5]
    normals = [SVector(0.0, 0.0) for _ in 1:5]
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    fdata = RBF.FunctionalData(basis, mon, (RBF.∂(basis, 1),), (RBF.∂_Hermite(mon, 1),))
    k_stencil = RBF.autoselect_k(all_coords, basis)
    k_stencil = 8
    println("k_stencil = $k_stencil")
    adjl = RBF.find_neighbors(all_coords, k_stencil)
    # println("adjl = $adjl")
    region_dx = RBF.RegionData(
        all_coords, is_boundary, boundary_types, normals, adjl, fdata
    )
    lhs_dx, rhs_dx = RBF._build_weights(region_dx)
    @test size(lhs_dx) == (sum(.!is_boundary), sum(.!is_boundary))
    @test size(rhs_dx) == (sum(.!is_boundary), sum(is_boundary))

    #Manufactured solution
    an_sol(coord) = 1.0 + 2.0 * coord[1] + 3.0 * coord[2]
    dx_an = 2.0
    # RHS = -rhs_dx * (dx_an * ones(Float64, sum(is_boundary)))
    RHS = -rhs_dx * (an_sol.(all_coords[is_boundary]))
    prob = LinearProblem(lhs_dx, RHS)
    sol = solve(prob)
    @test sol.u ≈ an_sol.(all_coords[.!is_boundary])
end

@testset "Neumann boundary conditions - small deterministic case" begin
    # Configuration with mixed Dirichlet and Neumann boundary conditions
    # (Pure Neumann problems are under-determined)
    node_coords = [
        SVector(1.0, 2.0),
        SVector(2.0, 1.0),
        SVector(1.5, 0.0),
        SVector(0.0, 1.0),
        SVector(2.0, 0.0),
    ]
    is_boundary = [false, false, true, true, true]
    bt_neu = RBF.BoundaryType(Float64[0.0, 1.0])  # Neumann BC
    bt_dir = RBF.BoundaryType(Float64[1.0, 0.0])  # Dirichlet BC
    boundary_types = [bt_neu, bt_dir, bt_neu]  # Mix of Neumann and Dirichlet
    normals = [
        SVector(0.0, 1.0),  # node 3 (Neumann)
        SVector(-1.0, 0.0), # node 4 (Dirichlet - normal not used)
        SVector(1.0, 0.0),  # node 5 (Neumann)
    ]

    # Adjacency: each node + 3 others (k = 4 for internal nodes)
    adjl = [[1, 2, 3, 4], [2, 1, 3, 5], [3, 1, 2, 5], [4, 1, 3, 5], [5, 2, 3, 4]]

    # Basis & operators
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    fdata = RBF.FunctionalData(basis, mon, (RBF.∂(basis, 1),), (RBF.∂_Hermite(mon, 1),))
    region = RBF.RegionData(node_coords, is_boundary, boundary_types, normals, adjl, fdata)

    @testset "Mixed Neumann/Dirichlet boundary types" begin
        @test RBF.is_Neumann(boundary_types[1])
        @test RBF.is_Dirichlet(boundary_types[2])
        @test RBF.is_Neumann(boundary_types[3])
        @test !RBF.is_Robin(boundary_types[1])
        @test !RBF.is_Robin(boundary_types[2])
        @test !RBF.is_Robin(boundary_types[3])
    end

    @testset "Neumann matrix structure" begin
        lhs_mat, rhs_mat = RBF._build_weights(region)
        @test size(lhs_mat) == (2, 2)
        @test size(rhs_mat) == (2, 3)
        @test nnz(lhs_mat) > 0
        @test nnz(rhs_mat) > 0
    end

    @testset "Mixed Neumann/Dirichlet manufactured solution test" begin
        # Use a linear function: u(x,y) = 1 + 2x + 3y
        # ∂u/∂x = 2, ∂u/∂y = 3
        an_sol(coord) = 1.0 + 2.0 * coord[1] + 3.0 * coord[2]
        du_dx = 2.0
        du_dy = 3.0

        # Calculate boundary values based on boundary type
        boundary_values = Float64[]
        boundary_coords = node_coords[is_boundary]

        for (i, (coord, normal, bt)) in
            enumerate(zip(boundary_coords, normals, boundary_types))
            if RBF.is_Dirichlet(bt)
                # Dirichlet: u = g
                push!(boundary_values, an_sol(coord))
            elseif RBF.is_Neumann(bt)
                # Neumann: ∂u/∂n = g
                normal_derivative = normal[1] * du_dx + normal[2] * du_dy
                push!(boundary_values, normal_derivative)
            end
        end

        lhs_mat, rhs_mat = RBF._build_weights(region)
        RHS = -rhs_mat * boundary_values
        prob = LinearProblem(lhs_mat, RHS)
        sol = solve(prob)

        # Solution should match manufactured solution at internal nodes
        expected = an_sol.(node_coords[.!is_boundary])
        @test sol.u ≈ expected atol = 1e-10
    end
end

@testset "Pure Neumann boundary conditions - singularity test" begin
    # Configuration with all Neumann boundary conditions (singular system)
    node_coords = [
        SVector(1.0, 2.0),
        SVector(2.0, 1.0),
        SVector(1.5, 0.0),
        SVector(0.0, 1.0),
        SVector(2.0, 0.0),
    ]
    is_boundary = [false, false, true, true, true]
    bt_neu = RBF.BoundaryType(Float64[0.0, 1.0])  # Pure Neumann BC
    boundary_types = [bt_neu, bt_neu, bt_neu]  # All boundary nodes are Neumann
    normals = [
        SVector(0.0, 1.0),  # node 3
        SVector(-1.0, 0.0), # node 4
        SVector(1.0, 0.0),  # node 5
    ]

    # Adjacency: each node + 3 others
    adjl = [[1, 2, 3, 4], [2, 1, 3, 5], [3, 1, 2, 5], [4, 1, 3, 5], [5, 2, 3, 4]]

    # Basis & operators
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    fdata = RBF.FunctionalData(basis, mon, (RBF.∂(basis, 1),), (RBF.∂_Hermite(mon, 1),))
    region = RBF.RegionData(node_coords, is_boundary, boundary_types, normals, adjl, fdata)

    @testset "All boundary types are Neumann" begin
        for bt in boundary_types
            @test RBF.is_Neumann(bt)
            @test !RBF.is_Dirichlet(bt)
            @test !RBF.is_Robin(bt)
        end
    end

    @testset "Pure Neumann matrix structure" begin
        lhs_mat, rhs_mat = RBF._build_weights(region)
        @test size(lhs_mat) == (2, 2)
        @test size(rhs_mat) == (2, 3)
        @test nnz(lhs_mat) > 0
        @test nnz(rhs_mat) > 0

        # Pure Neumann systems are typically singular
        # Check that the system can be constructed (but may not be solvable)
        @test isa(lhs_mat, SparseMatrixCSC)
        @test isa(rhs_mat, SparseMatrixCSC)
    end
end

@testset "Robin boundary conditions - small deterministic case" begin
    # Configuration with Robin boundary conditions (α*u + β*(∂u/∂n) = g)
    node_coords = [
        SVector(1.0, 2.0),
        SVector(2.0, 1.0),
        SVector(1.5, 0.0),
        SVector(0.0, 1.0),
        SVector(2.0, 0.0),
    ]
    is_boundary = [false, false, true, true, true]

    # Robin BCs with different coefficients
    bt_robin1 = RBF.BoundaryType(Float64[1.0, 1.0])  # α=1, β=1
    bt_robin2 = RBF.BoundaryType(Float64[2.0, 0.5])  # α=2, β=0.5
    bt_robin3 = RBF.BoundaryType(Float64[0.5, 2.0])  # α=0.5, β=2.0
    boundary_types = [bt_robin1, bt_robin2, bt_robin3]

    normals = [
        SVector(0.0, 1.0),  # node 3
        SVector(-1.0, 0.0), # node 4
        SVector(1.0, 0.0),  # node 5
    ]

    # Adjacency: each node + 3 others (k = 4 for internal nodes)
    adjl = [[1, 2, 3, 4], [2, 1, 3, 5], [3, 1, 2, 5], [4, 1, 3, 5], [5, 2, 3, 4]]

    # Basis & operators
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    fdata = RBF.FunctionalData(basis, mon, (RBF.∂(basis, 1),), (RBF.∂_Hermite(mon, 1),))
    region = RBF.RegionData(node_coords, is_boundary, boundary_types, normals, adjl, fdata)

    @testset "All boundary types are Robin" begin
        for bt in boundary_types
            @test RBF.is_Robin(bt)
            @test !RBF.is_Dirichlet(bt)
            @test !RBF.is_Neumann(bt)
        end
    end

    @testset "Robin coefficient access" begin
        @test RBF.α(bt_robin1) == 1.0
        @test RBF.β(bt_robin1) == 1.0
        @test RBF.α(bt_robin2) == 2.0
        @test RBF.β(bt_robin2) == 0.5
        @test RBF.α(bt_robin3) == 0.5
        @test RBF.β(bt_robin3) == 2.0
    end

    @testset "Robin matrix structure" begin
        lhs_mat, rhs_mat = RBF._build_weights(region)
        @test size(lhs_mat) == (2, 2)
        @test size(rhs_mat) == (2, 3)
        @test nnz(lhs_mat) > 0
        @test nnz(rhs_mat) > 0
    end

    @testset "Robin manufactured solution test" begin
        # Use a linear function: u(x,y) = 1 + 2x + 3y
        # ∂u/∂x = 2, ∂u/∂y = 3
        an_sol(coord) = 1.0 + 2.0 * coord[1] + 3.0 * coord[2]
        du_dx = 2.0
        du_dy = 3.0

        # Calculate Robin boundary values: α*u + β*(∂u/∂n) = g
        boundary_values = Float64[]
        boundary_coords = node_coords[is_boundary]

        for (i, (coord, normal, bt)) in
            enumerate(zip(boundary_coords, normals, boundary_types))
            u_val = an_sol(coord)
            normal_derivative = normal[1] * du_dx + normal[2] * du_dy
            robin_value = RBF.α(bt) * u_val + RBF.β(bt) * normal_derivative
            push!(boundary_values, robin_value)
        end

        lhs_mat, rhs_mat = RBF._build_weights(region)
        RHS = -rhs_mat * boundary_values
        prob = LinearProblem(lhs_mat, RHS)
        sol = solve(prob)

        # Solution should match manufactured solution
        expected = an_sol.(node_coords[.!is_boundary])
        @test sol.u ≈ expected atol = 1e-10
    end
end

@testset "Mixed boundary conditions - comprehensive test" begin
    # Configuration with all three types of boundary conditions
    node_coords = [
        SVector(1.0, 2.0),   # internal
        SVector(2.0, 1.0),   # internal
        SVector(3.0, 1.5),   # internal
        SVector(1.5, 0.0),   # boundary - Dirichlet
        SVector(0.0, 1.0),   # boundary - Neumann
        SVector(2.0, 0.0),   # boundary - Robin
        SVector(3.0, 0.5),   # boundary - Dirichlet
    ]
    is_boundary = [false, false, false, true, true, true, true]

    # Mixed boundary types
    bt_dir = RBF.BoundaryType(Float64[1.0, 0.0])    # Dirichlet
    bt_neu = RBF.BoundaryType(Float64[0.0, 1.0])    # Neumann
    bt_robin = RBF.BoundaryType(Float64[1.5, 0.8])  # Robin
    boundary_types = [bt_dir, bt_neu, bt_robin, bt_dir]

    normals = [
        SVector(0.0, 1.0),   # node 4 (Dirichlet - normal not used)
        SVector(-1.0, 0.0),  # node 5 (Neumann)
        SVector(1.0, 0.0),   # node 6 (Robin)
        SVector(0.0, -1.0),  # node 7 (Dirichlet - normal not used)
    ]

    # Larger adjacency lists for more nodes
    adjl = [
        [1, 2, 3, 4, 5],     # Node 1
        [2, 1, 3, 5, 6],     # Node 2  
        [3, 1, 2, 6, 7],     # Node 3
        [4, 1, 2, 3, 7],     # Node 4
        [5, 1, 2, 4, 6],     # Node 5
        [6, 2, 3, 5, 7],     # Node 6
        [7, 3, 4, 6],        # Node 7
    ]

    # Basis & operators
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    fdata = RBF.FunctionalData(basis, mon, (RBF.∂(basis, 1),), (RBF.∂_Hermite(mon, 1),))
    region = RBF.RegionData(node_coords, is_boundary, boundary_types, normals, adjl, fdata)

    @testset "Mixed boundary type classification" begin
        @test RBF.is_Dirichlet(boundary_types[1])
        @test RBF.is_Neumann(boundary_types[2])
        @test RBF.is_Robin(boundary_types[3])
        @test RBF.is_Dirichlet(boundary_types[4])

        @test !RBF.is_Neumann(boundary_types[1])
        @test !RBF.is_Robin(boundary_types[1])
        @test !RBF.is_Dirichlet(boundary_types[2])
        @test !RBF.is_Robin(boundary_types[2])
        @test !RBF.is_Dirichlet(boundary_types[3])
        @test !RBF.is_Neumann(boundary_types[3])
    end

    @testset "Mixed boundary matrix structure" begin
        lhs_mat, rhs_mat = RBF._build_weights(region)
        @test size(lhs_mat) == (3, 3)  # 3 internal nodes
        @test size(rhs_mat) == (3, 4)  # 3 internal nodes x 4 boundary nodes
        @test nnz(lhs_mat) > 0
        @test nnz(rhs_mat) > 0
    end

    @testset "Mixed boundary manufactured solution test" begin
        # Use a simpler linear function: u(x,y) = 1 + 2x + 3y
        # ∂u/∂x = 2, ∂u/∂y = 3
        an_sol(coord) = 1.0 + 2.0 * coord[1] + 3.0 * coord[2]
        du_dx(coord) = 2.0
        du_dy(coord) = 3.0

        # Calculate boundary values based on boundary type
        boundary_values = Float64[]
        boundary_coords = node_coords[is_boundary]

        for (i, (coord, normal, bt)) in
            enumerate(zip(boundary_coords, normals, boundary_types))
            if RBF.is_Dirichlet(bt)
                # Dirichlet: u = g
                push!(boundary_values, an_sol(coord))
            elseif RBF.is_Neumann(bt)
                # Neumann: ∂u/∂n = g
                normal_derivative = normal[1] * du_dx(coord) + normal[2] * du_dy(coord)
                push!(boundary_values, normal_derivative)
            elseif RBF.is_Robin(bt)
                # Robin: α*u + β*(∂u/∂n) = g
                u_val = an_sol(coord)
                normal_derivative = normal[1] * du_dx(coord) + normal[2] * du_dy(coord)
                robin_value = RBF.α(bt) * u_val + RBF.β(bt) * normal_derivative
                push!(boundary_values, robin_value)
            end
        end

        lhs_mat, rhs_mat = RBF._build_weights(region)
        RHS = -rhs_mat * boundary_values
        prob = LinearProblem(lhs_mat, RHS)
        sol = solve(prob)

        # Solution should match manufactured solution at internal nodes
        expected = an_sol.(node_coords[.!is_boundary])
        @test sol.u ≈ expected atol = 1e-6  # Relaxed tolerance for more complex case
    end
end

@testset "Boundary condition helper constructors" begin
    @testset "Create Dirichlet" begin
        bt = RBF.Dirichlet(Float64)
        @test RBF.is_Dirichlet(bt)
        @test !RBF.is_Neumann(bt)
        @test !RBF.is_Robin(bt)
        @test RBF.α(bt) == 1.0
        @test RBF.β(bt) == 0.0
    end

    @testset "Create Neumann" begin
        bt = RBF.Neumann(Float64)
        @test RBF.is_Neumann(bt)
        @test !RBF.is_Dirichlet(bt)
        @test !RBF.is_Robin(bt)
        @test RBF.α(bt) == 0.0
        @test RBF.β(bt) == 1.0
    end

    @testset "Create Robin" begin
        α_val = 2.5
        β_val = 1.8
        bt = RBF.Robin(α_val, β_val)
        @test RBF.is_Robin(bt)
        @test !RBF.is_Dirichlet(bt)
        @test !RBF.is_Neumann(bt)
        @test RBF.α(bt) ≈ α_val
        @test RBF.β(bt) ≈ β_val
    end
end

# @testset "Alternative operator (y-derivative)" begin
#     ℒrbf_y = (RBF.∂(basis, 2),)
#     ℒmon_y = (RBF.∂_Hermite(mon, 2),)
#     fdata_y = RBF.FunctionalData(basis, mon, ℒrbf_y, ℒmon_y)
#     region_y = RBF.RegionData(
#         node_coords, is_boundary, boundary_types, normals, adjl, fdata_y
#     )
#     lhs_y, rhs_y = RBF._build_weights(region_y)
#     lhs_x, rhs_x = RBF._build_weights(region)
#     @test size(lhs_y) == size(lhs_x)
#     @test size(rhs_y) == size(rhs_x)
#     # Expect some difference; relax to norm-based to avoid sparse ordering pitfalls
#     @test (norm(lhs_y - lhs_x) + norm(rhs_y - rhs_x)) > 1e-10
#     println("Difference between y-derivative and x-derivative weights:")
#     println(lhs_y - lhs_x)
#     println(rhs_y - rhs_x)
# end
# end

# The remaining detailed unit tests below (legacy style) were removed to avoid duplication
# and reliance on old APIs. They can be reintroduced incrementally if needed with RegionData
# abstractions.

# @testset "solve_with_Hermite" begin
#     # Setup test data - with one more boundary node that has Neumann BC
#     data = [
#         SVector(1.0, 2.0),   # internal (node 1)
#         SVector(2.0, 1.0),   # internal (node 2) 
#         SVector(1.5, 0.0),   # boundary with Neumann BC (node 3)
#         SVector(0.0, 1.0),   # boundary with Dirichlet BC (node 4)
#         SVector(2.0, 0.0),    # boundary with Neumann BC (node 5)
#     ]
#     boundary_flag = [false, false, true, true, true]  # First two internal, last three boundary
#     is_Neumann = [false, false, true, false, true]    # Nodes 3 and 5 have Neumann BC
#     normals = [
#         SVector(0.0, 0.0),   # Internal nodes have no normals
#         SVector(0.0, 0.0),
#         SVector(0.0, 1.0),   # Normal pointing in y direction
#         SVector(0.0, 0.0),   # Not used for Dirichlet
#         SVector(1.0, 0.0),    # Normal pointing in x direction
#     ]

#     # Define adjacency lists (each node connects to itself and 3 others)
#     adjl = [
#         [1, 2, 3, 4],        # Node 1 connects to itself, 2, 3, 4
#         [2, 1, 3, 5],        # Node 2 connects to itself, 1, 3, 5
#         [3, 1, 2, 5],        # Node 3 connects to itself, 1, 2, 5
#         [4, 1, 3, 5],        # Node 4 connects to itself, 1, 3, 5
#         [5, 2, 3, 4],         # Node 5 connects to itself, 2, 3, 4
#     ]

#     # Create basis functions
#     basis = RBF.PHS(3; poly_deg=1)
#     mon = RBF.MonomialBasis(2, 1)

#     Lrb = RBF.∂(basis, 1)
#     Lmb = RBF.∂_Hermite(mon, 1)

#     @testset "_preallocate_IJV_matrices" begin
#         # Using RBF directly instead of getfield
#         (I_lhs, J_lhs, V_lhs), (I_rhs, J_rhs, V_rhs) = RBF._preallocate_IJV_matrices(
#             adjl, data, boundary_flag, Lrb
#         )

#         # Check dimensions
#         # 2 internal nodes with self-connections + connections to each other
#         @test length(I_lhs) == 4
#         @test length(J_lhs) == 4
#         @test size(V_lhs) == (4, 1)  # Single operator

#         # 2 internal nodes with 3 boundary neighbors each (not all boundary nodes connect to all internal nodes)
#         # Node 1 connects to boundary nodes 3, 4
#         # Node 2 connects to boundary nodes 3, 5
#         @test length(I_rhs) == 4
#         @test length(J_rhs) == 4
#         @test size(V_rhs) == (4, 1)

#         # Check values - internal to internal connections
#         @test (1, 1) in zip(I_lhs, J_lhs)  # Node 1 connects to itself
#         @test (1, 2) in zip(I_lhs, J_lhs)  # Node 1 connects to node 2
#         @test (2, 2) in zip(I_lhs, J_lhs)  # Node 2 connects to itself
#         @test (2, 1) in zip(I_lhs, J_lhs)  # Node 2 connects to node 1

#         # Check values - internal to boundary connections
#         # Node indices are remapped for boundary nodes: 3->1, 4->2, 5->3
#         @test (1, 1) in zip(I_rhs, J_rhs)  # Node 1 connects to first boundary node (3)
#         @test (1, 2) in zip(I_rhs, J_rhs)  # Node 1 connects to second boundary node (4)
#         @test (2, 1) in zip(I_rhs, J_rhs)  # Node 2 connects to first boundary node (3)
#         @test (2, 3) in zip(I_rhs, J_rhs)  # Node 2 connects to third boundary node (5)
#     end

#     @testset "_calculate_thread_offsets" begin
#         nchunks = 2
#         lhs_offsets, rhs_offsets = RBF._calculate_thread_offsets(
#             adjl, boundary_flag, nchunks
#         )

#         # Check that offsets are calculated correctly
#         @test length(lhs_offsets) == nchunks
#         @test length(rhs_offsets) == nchunks
#         @test lhs_offsets[1] == 0  # First thread starts at index 0
#         @test rhs_offsets[1] == 0  # First thread starts at index 0

#         # Second thread starts after first thread's internal connections
#         @test lhs_offsets[2] == 4
#         # Second thread starts after first thread's boundary connections
#         @test rhs_offsets[2] == 4
#     end

#     @testset "_update_stencil!" begin
#         TD = Float64
#         dim = 2
#         k = 4  # Number of neighbors (self + 3)
#         nmon = 3  # 1 + x + y
#         num_ops = 1

#         stencil = RBF.StencilData(TD, dim, k + nmon, k, num_ops)

#         i = 1  # Test for first node
#         RBF._update_stencil!(
#             stencil,
#             adjl[i],
#             data,
#             boundary_flag,
#             is_Neumann,
#             normals,
#             Lrb,
#             Lmb,
#             data[i],
#             basis,
#             mon,
#             k,
#         )

#         # Check stencil data was updated with correct points
#         @test stencil.d[1] == data[1]  # Self connection
#         @test stencil.d[2] == data[2]  # Connection to node 2
#         @test stencil.d[3] == data[3]  # Connection to node 3
#         @test stencil.d[4] == data[4]  # Connection to node 4

#         # Check boundary flags
#         @test stencil.is_boundary[1] == false  # Node 1 (self)
#         @test stencil.is_boundary[2] == false  # Node 2
#         @test stencil.is_boundary[3] == true   # Node 3
#         @test stencil.is_boundary[4] == true   # Node 4

#         # Check Neumann flags
#         @test stencil.is_Neumann[1] == false  # Node 1 is not Neumann
#         @test stencil.is_Neumann[2] == false  # Node 2 is not Neumann
#         @test stencil.is_Neumann[3] == true   # Node 3 is Neumann
#         @test stencil.is_Neumann[4] == false  # Node 4 is not Neumann

#         # Check that weights were computed
#         @test any(stencil.lhs_v .!= 0)  # Some LHS weights should be non-zero
#         @test any(stencil.rhs_v .!= 0)  # Some RHS weights should be non-zero

#         # Test with second node to verify different adjacency pattern
#         i = 2
#         stencil2 = RBF.StencilData(TD, dim, k + nmon, k, num_ops)
#         RBF._update_stencil!(
#             stencil2,
#             adjl[i],
#             data,
#             boundary_flag,
#             is_Neumann,
#             normals,
#             Lrb,
#             Lmb,
#             data[i],
#             basis,
#             mon,
#             k,
#         )

#         # Check connections for node 2
#         @test stencil2.d[1] == data[2]  # Self connection
#         @test stencil2.d[2] == data[1]  # Connection to node 1
#         @test stencil2.d[3] == data[3]  # Connection to node 3
#         @test stencil2.d[4] == data[5]  # Connection to node 5 (different from node 1)
#     end

#     @testset "_write_coefficients_to_global_matrices!" begin
#         TD = Float64
#         num_ops = 1

#         # Create sample stencil data with 4 neighbors
#         stencil = RBF.StencilData(TD, 2, 7, 4, num_ops)  # 4 neighbors + 3 monomial terms

#         # Set values for all 4 connections in the correct locations
#         stencil.lhs_v[1, 1] = 1.0  # Self connection (node 1 to node 1)
#         stencil.lhs_v[2, 1] = 2.0  # Node 1 to node 2 (internal)
#         stencil.rhs_v[3, 1] = 3.0  # Node 1 to node 3 (boundary)
#         stencil.rhs_v[4, 1] = 4.0  # Node 1 to node 4 (boundary)

#         # Create target matrices
#         V_lhs = zeros(4, num_ops)
#         V_rhs = zeros(4, num_ops)

#         # Start with known indices
#         lhs_idx = 1
#         rhs_idx = 1

#         # Use the full adjacency list with self-connection
#         new_lhs_idx, new_rhs_idx = RBF._write_coefficients_to_global_matrices!(
#             V_lhs, V_rhs, stencil, adjl[1], boundary_flag, lhs_idx, rhs_idx
#         )

#         # Count internal and boundary neighbors
#         internal_count = count(i -> !boundary_flag[i], adjl[1])
#         boundary_count = count(i -> boundary_flag[i], adjl[1])

#         # Check values were written
#         @test any(V_lhs .!= 0)  # At least some values should be non-zero
#         @test any(V_rhs .!= 0)  # At least some values should be non-zero

#         # Check indices were updated correctly 
#         @test new_lhs_idx == lhs_idx + internal_count
#         @test new_rhs_idx == rhs_idx + boundary_count

#         # Test with Neumann boundary condition
#         stencil.is_Neumann[3] = true
#         V_lhs = zeros(4, num_ops)
#         V_rhs = zeros(4, num_ops)

#         lhs_idx = 1
#         rhs_idx = 1

#         RBF._write_coefficients_to_global_matrices!(
#             V_lhs, V_rhs, stencil, adjl[1], boundary_flag, lhs_idx, rhs_idx
#         )

#         # Verify Neumann BC handling
#         @test any(V_lhs .!= 0)
#         @test any(V_rhs .!= 0)
#     end

#     @testset "_return_global_matrices" begin
#         # Create sample matrices
#         I_lhs = [1, 1, 2, 2]
#         J_lhs = [1, 2, 1, 2]
#         V_lhs = [1.0, 2.0, 3.0, 4.0]

#         I_rhs = [1, 1, 2, 2]
#         J_rhs = [1, 2, 1, 3]
#         V_rhs = [5.0, 6.0, 7.0, 8.0]

#         # Call function
#         lhs_matrix, rhs_matrix = RBF._return_global_matrices(
#             I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag
#         )

#         # Check that matrices were created correctly
#         @test isa(lhs_matrix, SparseMatrixCSC)
#         @test isa(rhs_matrix, SparseMatrixCSC)
#         @test size(lhs_matrix) == (2, 2)  # 2 internal nodes
#         @test size(rhs_matrix) == (2, 3)  # 2 internal nodes x 3 boundary nodes

#         # Check values
#         @test lhs_matrix[1, 1] == 1.0
#         @test lhs_matrix[1, 2] == 2.0
#         @test lhs_matrix[2, 1] == 3.0
#         @test lhs_matrix[2, 2] == 4.0

#         @test rhs_matrix[1, 1] == 5.0
#         @test rhs_matrix[1, 2] == 6.0
#         @test rhs_matrix[2, 1] == 7.0
#         @test rhs_matrix[2, 3] == 8.0

#         # Test multiple operators case
#         V_lhs_multi = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0]
#         V_rhs_multi = [5.0 50.0; 6.0 60.0; 7.0 70.0; 8.0 80.0]

#         lhs_matrices, rhs_matrices = RBF._return_global_matrices(
#             I_lhs, J_lhs, V_lhs_multi, I_rhs, J_rhs, V_rhs_multi, boundary_flag
#         )

#         @test length(lhs_matrices) == 2  # Two operators
#         @test length(rhs_matrices) == 2
#         @test size(lhs_matrices[1]) == (2, 2)
#         @test size(rhs_matrices[1]) == (2, 3)

#         # Check second operator values
#         @test lhs_matrices[2][1, 1] == 10.0
#         @test lhs_matrices[2][1, 2] == 20.0
#         @test lhs_matrices[2][2, 1] == 30.0
#         @test lhs_matrices[2][2, 2] == 40.0

#         @test rhs_matrices[2][1, 1] == 50.0
#         @test rhs_matrices[2][1, 2] == 60.0
#         @test rhs_matrices[2][2, 1] == 70.0
#         @test rhs_matrices[2][2, 3] == 80.0
#     end

#     @testset "Full integration test" begin
#         # Test the complete workflow
#         matrices = RBF._build_weights(
#             data, normals, boundary_flag, is_Neumann, adjl, basis, Lrb, Lmb, mon
#         )

#         @test length(matrices) == 2  # Returns (lhs_matrix, rhs_matrix)
#         lhs_matrix, rhs_matrix = matrices

#         @test size(lhs_matrix) == (2, 2)  # 2 internal nodes
#         @test size(rhs_matrix) == (2, 3)  # 2 internal nodes x 3 boundary nodes

#         # Some basic sanity checks
#         @test isa(lhs_matrix, SparseMatrixCSC)
#         @test isa(rhs_matrix, SparseMatrixCSC)
#         @test nnz(lhs_matrix) > 0  # Should have non-zero entries
#         @test nnz(rhs_matrix) > 0

#         # Test with y-derivative operator (instead of x-derivative)
#         Lrbf_y = RBF.∂(basis, 2)  # Use y-derivative (dimension 2)
#         Lmbf_y = RBF.∂_Hermite(mon, 2)    # Use y-derivative (dimension 2)

#         matrices_y = RBF._build_weights(
#             data, normals, boundary_flag, is_Neumann, adjl, basis, Lrbf_y, Lmbf_y, mon
#         )

#         lhs_matrix_y, rhs_matrix_y = matrices_y
#         @test size(lhs_matrix_y) == (2, 2)
#         @test size(rhs_matrix_y) == (2, 3)
#         @test nnz(lhs_matrix_y) > 0
#         @test nnz(rhs_matrix_y) > 0

#         # Make sure y derivative matrices are different from original
#         @test any(lhs_matrix_y.nzval .≠ lhs_matrix.nzval) ||
#             any(rhs_matrix_y.nzval .≠ rhs_matrix.nzval)
#     end
# end