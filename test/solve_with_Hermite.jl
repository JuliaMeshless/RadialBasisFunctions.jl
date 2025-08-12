using StaticArraysCore
using LinearAlgebra
using SparseArrays
using Test

import RadialBasisFunctions as RBF

@testset "solve_with_Hermite (new API)" begin
    # Geometry: 2 internal nodes, 3 boundary nodes (Dirichlet, Neumann, Robin)
    data = [
        SVector(1.0, 2.0),   # 1 internal
        SVector(2.0, 1.0),   # 2 internal
        SVector(1.5, 0.0),   # 3 boundary (Neumann)
        SVector(0.0, 1.0),   # 4 boundary (Dirichlet)
        SVector(2.2, 0.2),   # 5 boundary (Robin)
    ]
    is_boundary = [false, false, true, true, true]

    # Boundary ordering follows global order (3,4,5)
    bt_neu = RBF.BoundaryType(Float64[0.0, 1.0])            # Neumann α=0 β=1
    bt_dir = RBF.BoundaryType(Float64[1.0, 0.0])            # Dirichlet α=1 β=0
    bt_rob = RBF.BoundaryType(Float64[0.4, 0.6])            # Robin α>0 β>0
    boundary_types = [bt_neu, bt_dir, bt_rob]
    normals = [
        SVector(0.0, 1.0),   # for node 3 (Neumann)
        SVector(0.0, 0.0),   # for node 4 (Dirichlet - unused)
        SVector(1.0, 0.0),   # for node 5 (Robin)
    ]

    # Adjacency: uniform length (each node lists 5 entries: itself + all other internal + all boundary)
    adjl = [
        [1, 2, 3, 4, 5], [2, 1, 3, 4, 5], [3, 1, 2, 4, 5], [4, 1, 2, 3, 5], [5, 1, 2, 3, 4]
    ]

    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    ℒrbf = (RBF.∂(basis, 1),)
    ℒmon = (RBF.∂_Hermite(mon, 1),)
    fdata = RBF.FunctionalData(basis, mon, ℒrbf, ℒmon)
    region = RBF.RegionData(data, is_boundary, boundary_types, normals, adjl, fdata)

    @testset "BoundaryType classification" begin
        @test RBF.is_Neumann(bt_neu)
        @test RBF.is_Dirichlet(bt_dir)
        @test RBF.is_Robin(bt_rob)
    end

    @testset "_preallocate_IJV_matrices" begin
        lhs, rhs = RBF._preallocate_IJV_matrices(region)
        @test length(lhs.I) == 2 * 2  # internal/internal including self + other for two nodes
        @test length(rhs.I) == 2 * 3  # each internal connects to 3 boundary
        @test size(lhs.V, 2) == 1
        @test size(rhs.V, 2) == 1
        # internal indices should be in 1:2, boundary remapped to 1:3
        @test all(1 .<= lhs.I .<= 2)
        @test all(1 .<= lhs.J .<= 2)
        @test all(1 .<= rhs.J .<= 3)
    end

    @testset "_calculate_thread_offsets" begin
        lhs_off, rhs_off = RBF._calculate_thread_offsets(region, 2)
        @test length(lhs_off) == 2
        @test length(rhs_off) == 2
        # first offsets start at 1
        @test lhs_off[1] == 1
        @test rhs_off[1] == 1
    end

    @testset "_update_stencil! local matrix entries" begin
        stencil = RBF.StencilData(region)
        # choose internal node 1
        RBF._set_stencil_eval_point!(stencil, data[1])
        RBF._update_stencil!(stencil, region, 1)

        # adjacency length for internal node 1
        k = length(region.adjl[1])
        A = parent(stencil.A)
        # Identify local indices of boundary nodes (3,4,5)
        loc_map = Dict(stencil.local_adjl[i] => i for i in 1:k)
        i_internal = loc_map[1]
        idx_neu = loc_map[3]
        idx_dir = loc_map[4]
        idx_rob = loc_map[5]

        x_int = data[1]
        φ_neu = basis(x_int, data[3])
        φ_dir = basis(x_int, data[4])
        φ_rob = basis(x_int, data[5])
        g_neu = RBF.∇(basis)(x_int, data[3])
        g_rob = RBF.∇(basis)(x_int, data[5])
        # Expected entries (Dirichlet uses α=1 β=0)
        exp_dir = φ_dir
        exp_neu = dot(normals[1], -g_neu) * 1.0  # α=0 β=1, second argument derivative sign
        exp_rob = 0.4 * φ_rob + 0.6 * dot(normals[3], -g_rob)

        @test isapprox(A[i_internal, idx_dir], exp_dir; atol=1e-10)
        @test isapprox(A[i_internal, idx_neu], exp_neu; atol=1e-10)
        @test isapprox(A[i_internal, idx_rob], exp_rob; atol=1e-10)
    end

    @testset "Full integration" begin
        lhs_mat, rhs_mat = RBF._build_weights(region)
        @test size(lhs_mat, 1) == 2
        @test size(lhs_mat, 2) == 2
        @test size(rhs_mat, 1) == 2
        @test size(rhs_mat, 2) == 3
        @test nnz(lhs_mat) > 0
        @test nnz(rhs_mat) > 0
    end

    @testset "Alternative operator (y-derivative)" begin
        ℒrbf_y = (RBF.∂(basis, 2),)
        ℒmon_y = (RBF.∂_Hermite(mon, 2),)
        fdata_y = RBF.FunctionalData(basis, mon, ℒrbf_y, ℒmon_y)
        region_y = RBF.RegionData(data, is_boundary, boundary_types, normals, adjl, fdata_y)
        lhs_y, rhs_y = RBF._build_weights(region_y)
        @test nnz(lhs_y) > 0 && nnz(rhs_y) > 0
        # Expect difference from x-derivative system
        lhs_x, rhs_x = RBF._build_weights(region)
        @test any(lhs_y.nzval .!= lhs_x.nzval) || any(rhs_y.nzval .!= rhs_x.nzval)
    end
end
