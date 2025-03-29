using StaticArraysCore
using LinearAlgebra
using SparseArrays
using Test

# Include the module file
include("../src/RadialBasisFunctions.jl")
import RadialBasisFunctions as RBF

@testset "solve_with_Hermite" begin
    # Setup test data - with one more boundary node that has Neumann BC
    data = [
        SVector(1.0, 2.0),   # internal (node 1)
        SVector(2.0, 1.0),   # internal (node 2) 
        SVector(1.5, 0.0),   # boundary with Neumann BC (node 3)
        SVector(0.0, 1.0),   # boundary with Dirichlet BC (node 4)
        SVector(2.0, 0.0)    # boundary with Neumann BC (node 5)
    ]
    boundary_flag = [false, false, true, true, true]  # First two internal, last three boundary
    is_Neumann = [false, false, true, false, true]    # Nodes 3 and 5 have Neumann BC
    normals = [
        SVector(0.0, 0.0),   # Internal nodes have no normals
        SVector(0.0, 0.0), 
        SVector(0.0, 1.0),   # Normal pointing in y direction
        SVector(0.0, 0.0),   # Not used for Dirichlet
        SVector(1.0, 0.0)    # Normal pointing in x direction
    ]

    # Define adjacency lists (each node connects to itself and 3 others)
    adjl = [
        [1, 2, 3, 4],        # Node 1 connects to itself, 2, 3, 4
        [2, 1, 3, 5],        # Node 2 connects to itself, 1, 3, 5
        [3, 1, 2, 5],        # Node 3 connects to itself, 1, 2, 5
        [4, 1, 3, 5],        # Node 4 connects to itself, 1, 3, 5
        [5, 2, 3, 4]         # Node 5 connects to itself, 2, 3, 4
    ]

    # Create basis functions
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)

    Lrb = RBF.∂_Hermite(basis, 1)
    Lmb = RBF.∂_Hermite(mon, 1)

    @testset "_preallocate_IJV_matrices" begin
        # Using RBF directly instead of getfield
        (I_lhs, J_lhs, V_lhs), (I_rhs, J_rhs, V_rhs) = RBF._preallocate_IJV_matrices(
            adjl, data, boundary_flag, Lrb
        )

        # Check dimensions
        # 2 internal nodes with self-connections + connections to each other
        @test length(I_lhs) == 4  
        @test length(J_lhs) == 4
        @test size(V_lhs) == (4, 1)  # Single operator

        println(I_lhs, J_lhs, V_lhs)

        # 2 internal nodes with 3 boundary neighbors each (not all boundary nodes connect to all internal nodes)
        # Node 1 connects to boundary nodes 3, 4
        # Node 2 connects to boundary nodes 3, 5
        @test length(I_rhs) == 4  
        @test length(J_rhs) == 4
        @test size(V_rhs) == (4, 1)

        println(I_rhs, J_rhs, V_rhs)

        # Check values - internal to internal connections
        @test (1, 1) in zip(I_lhs, J_lhs)  # Node 1 connects to itself
        @test (1, 2) in zip(I_lhs, J_lhs)  # Node 1 connects to node 2
        @test (2, 2) in zip(I_lhs, J_lhs)  # Node 2 connects to itself
        @test (2, 1) in zip(I_lhs, J_lhs)  # Node 2 connects to node 1

        # Check values - internal to boundary connections
        # Node indices are remapped for boundary nodes: 3->1, 4->2, 5->3
        @test (1, 1) in zip(I_rhs, J_rhs)  # Node 1 connects to first boundary node (3)
        @test (1, 2) in zip(I_rhs, J_rhs)  # Node 1 connects to second boundary node (4)
        @test (2, 1) in zip(I_rhs, J_rhs)  # Node 2 connects to first boundary node (3)
        @test (2, 3) in zip(I_rhs, J_rhs)  # Node 2 connects to third boundary node (5)
    end

    @testset "_calculate_thread_offsets" begin
        nchunks = 2
        lhs_offsets, rhs_offsets = RBF._calculate_thread_offsets(adjl, boundary_flag, nchunks)

        # Check that offsets are calculated correctly
        @test length(lhs_offsets) == nchunks
        @test length(rhs_offsets) == nchunks
        @test lhs_offsets[1] == 0  # First thread starts at index 0
        @test rhs_offsets[1] == 0  # First thread starts at index 0
        
        # Second thread starts after first thread's internal connections
        @test lhs_offsets[2] == 4
        # Second thread starts after first thread's boundary connections
        @test rhs_offsets[2] == 4
    end

    @testset "_update_stencil!" begin
        TD = Float64
        dim = 2
        k = 4  # Number of neighbors (self + 3)
        nmon = 3  # 1 + x + y
        num_ops = 1

        stencil = RBF.StencilData(TD, dim, k + nmon, k, num_ops)

        i = 1  # Test for first node
        RBF._update_stencil!(
            stencil,
            adjl[i],
            data,
            boundary_flag,
            is_Neumann,
            normals,
            Lrb,
            Lmb,
            data[i],
            basis,
            mon,
            k,
        )

        # Check stencil data was updated with correct points
        @test stencil.d[1] == data[1]  # Self connection
        @test stencil.d[2] == data[2]  # Connection to node 2
        @test stencil.d[3] == data[3]  # Connection to node 3
        @test stencil.d[4] == data[4]  # Connection to node 4

        # Check boundary flags
        @test stencil.is_boundary[1] == false  # Node 1 (self)
        @test stencil.is_boundary[2] == false  # Node 2
        @test stencil.is_boundary[3] == true   # Node 3
        @test stencil.is_boundary[4] == true   # Node 4

        # Check Neumann flags
        @test stencil.is_Neumann[1] == false  # Node 1 is not Neumann
        @test stencil.is_Neumann[2] == false  # Node 2 is not Neumann
        @test stencil.is_Neumann[3] == true   # Node 3 is Neumann
        @test stencil.is_Neumann[4] == false  # Node 4 is not Neumann

        # Check that weights were computed
        @test any(stencil.lhs_v .!= 0)  # Some LHS weights should be non-zero
        @test any(stencil.rhs_v .!= 0)  # Some RHS weights should be non-zero
        
        # Test with second node to verify different adjacency pattern
        i = 2
        stencil2 = RBF.StencilData(TD, dim, k + nmon, k, num_ops)
        RBF._update_stencil!(
            stencil2,
            adjl[i],
            data,
            boundary_flag,
            is_Neumann,
            normals,
            Lrb,
            Lmb,
            data[i],
            basis,
            mon,
            k,
        )
        
        # Check connections for node 2
        @test stencil2.d[1] == data[2]  # Self connection
        @test stencil2.d[2] == data[1]  # Connection to node 1
        @test stencil2.d[3] == data[3]  # Connection to node 3
        @test stencil2.d[4] == data[5]  # Connection to node 5 (different from node 1)
    end

    @testset "_write_coefficients_to_global_matrices!" begin
        TD = Float64
        num_ops = 1

        # Create sample stencil data with 4 neighbors
        stencil = RBF.StencilData(TD, 2, 7, 4, num_ops)  # 4 neighbors + 3 monomial terms

        # Set values for all 4 connections in the correct locations
        stencil.lhs_v[1, 1] = 1.0  # Self connection (node 1 to node 1)
        stencil.lhs_v[2, 1] = 2.0  # Node 1 to node 2 (internal)
        stencil.rhs_v[3, 1] = 3.0  # Node 1 to node 3 (boundary)
        stencil.rhs_v[4, 1] = 4.0  # Node 1 to node 4 (boundary)

        # Create target matrices
        V_lhs = zeros(4, num_ops)
        V_rhs = zeros(4, num_ops)

        # Start with known indices
        lhs_idx = 1
        rhs_idx = 1

        # Use the full adjacency list with self-connection
        new_lhs_idx, new_rhs_idx = RBF._write_coefficients_to_global_matrices!(
            V_lhs, V_rhs, stencil, adjl[1], boundary_flag, lhs_idx, rhs_idx
        )

        # Count internal and boundary neighbors
        internal_count = count(i -> !boundary_flag[i], adjl[1])
        boundary_count = count(i -> boundary_flag[i], adjl[1])

        # Check values were written
        @test any(V_lhs .!= 0)  # At least some values should be non-zero
        @test any(V_rhs .!= 0)  # At least some values should be non-zero
        
        # Check indices were updated correctly 
        @test new_lhs_idx == lhs_idx + internal_count
        @test new_rhs_idx == rhs_idx + boundary_count
        
        # Test with Neumann boundary condition
        stencil.is_Neumann[3] = true
        V_lhs = zeros(4, num_ops)
        V_rhs = zeros(4, num_ops)
        
        lhs_idx = 1
        rhs_idx = 1
        
        RBF._write_coefficients_to_global_matrices!(
            V_lhs, V_rhs, stencil, adjl[1], boundary_flag, lhs_idx, rhs_idx
        )
        
        # Verify Neumann BC handling
        @test any(V_lhs .!= 0) 
        @test any(V_rhs .!= 0)
    end

    @testset "_return_global_matrices" begin
        # Create sample matrices
        I_lhs = [1, 1, 2, 2]
        J_lhs = [1, 2, 1, 2]
        V_lhs = [1.0, 2.0, 3.0, 4.0]

        I_rhs = [1, 1, 2, 2]
        J_rhs = [1, 2, 1, 3]
        V_rhs = [5.0, 6.0, 7.0, 8.0]

        # Call function
        lhs_matrix, rhs_matrix = RBF._return_global_matrices(
            I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag
        )

        # Check that matrices were created correctly
        @test isa(lhs_matrix, SparseMatrixCSC)
        @test isa(rhs_matrix, SparseMatrixCSC)
        @test size(lhs_matrix) == (2, 2)  # 2 internal nodes
        @test size(rhs_matrix) == (2, 3)  # 2 internal nodes x 3 boundary nodes

        # Check values
        @test lhs_matrix[1,1] == 1.0
        @test lhs_matrix[1,2] == 2.0
        @test lhs_matrix[2,1] == 3.0
        @test lhs_matrix[2,2] == 4.0
        
        @test rhs_matrix[1,1] == 5.0
        @test rhs_matrix[1,2] == 6.0
        @test rhs_matrix[2,1] == 7.0
        @test rhs_matrix[2,3] == 8.0

        # Test multiple operators case
        V_lhs_multi = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0]
        V_rhs_multi = [5.0 50.0; 6.0 60.0; 7.0 70.0; 8.0 80.0]

        lhs_matrices, rhs_matrices = RBF._return_global_matrices(
            I_lhs, J_lhs, V_lhs_multi, I_rhs, J_rhs, V_rhs_multi, boundary_flag
        )

        @test length(lhs_matrices) == 2  # Two operators
        @test length(rhs_matrices) == 2
        @test size(lhs_matrices[1]) == (2, 2)
        @test size(rhs_matrices[1]) == (2, 3)
        
        # Check second operator values
        @test lhs_matrices[2][1,1] == 10.0
        @test lhs_matrices[2][1,2] == 20.0
        @test lhs_matrices[2][2,1] == 30.0
        @test lhs_matrices[2][2,2] == 40.0
        
        @test rhs_matrices[2][1,1] == 50.0
        @test rhs_matrices[2][1,2] == 60.0
        @test rhs_matrices[2][2,1] == 70.0
        @test rhs_matrices[2][2,3] == 80.0
    end

    @testset "Full integration test" begin
        # Test the complete workflow
        matrices = RBF._build_weights(
            data, normals, boundary_flag, is_Neumann, adjl, basis, Lrb, Lmb, mon
        )

        @test length(matrices) == 2  # Returns (lhs_matrix, rhs_matrix)
        lhs_matrix, rhs_matrix = matrices

        @test size(lhs_matrix) == (2, 2)  # 2 internal nodes
        @test size(rhs_matrix) == (2, 3)  # 2 internal nodes x 3 boundary nodes

        # Some basic sanity checks
        @test isa(lhs_matrix, SparseMatrixCSC)
        @test isa(rhs_matrix, SparseMatrixCSC)
        @test nnz(lhs_matrix) > 0  # Should have non-zero entries
        @test nnz(rhs_matrix) > 0
        
        # Test with y-derivative operator (instead of x-derivative)
        Lrbf_y = RBF.∂_Hermite(basis, 2)  # Use y-derivative (dimension 2)
        Lmbf_y = RBF.∂_Hermite(mon, 2)    # Use y-derivative (dimension 2)
        
        matrices_y = RBF._build_weights(
            data, normals, boundary_flag, is_Neumann, adjl, basis, Lrbf_y, Lmbf_y, mon
        )
        
        lhs_matrix_y, rhs_matrix_y = matrices_y
        @test size(lhs_matrix_y) == (2, 2)
        @test size(rhs_matrix_y) == (2, 3)
        @test nnz(lhs_matrix_y) > 0
        @test nnz(rhs_matrix_y) > 0
        
        # Make sure y derivative matrices are different from original
        @test any(lhs_matrix_y.nzval .≠ lhs_matrix.nzval) || 
              any(rhs_matrix_y.nzval .≠ rhs_matrix.nzval)
    end
end
