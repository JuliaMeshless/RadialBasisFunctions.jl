using StaticArraysCore
using LinearAlgebra
using SparseArrays
using Test

# Include the module file
include("../src/RadialBasisFunctions.jl")
import .RadialBasisFunctions as RBF  # Note the dot to import the locally included module

# Directly access internal functions using getfield
_preallocate_IJV_matrices = getfield(RBF, :_preallocate_IJV_matrices)
_calculate_thread_offsets = getfield(RBF, :_calculate_thread_offsets)
_update_stencil! = getfield(RBF, :_update_stencil!)
_write_coefficients_to_global_matrices! = getfield(RBF, :_write_coefficients_to_global_matrices!)
_return_global_matrices = getfield(RBF, :_return_global_matrices)
_build_weights = getfield(RBF, :_build_weights)
StencilData = getfield(RBF, :StencilData)

@testset "solve_with_Hermite" begin
    # Setup test data
    data = [SVector(1.0, 2.0), SVector(2.0, 1.0), SVector(1.5, 0.0), SVector(0.0, 1.0)]
    boundary_flag = [false, false, true, true]  # First two are internal, last two are boundary
    is_Neumann = [false, false, true, false]    # Third point has Neumann condition
    normals = [SVector(0.0, 0.0), SVector(0.0, 0.0), SVector(0.0, 1.0), SVector(0.0, 0.0)]
    
    # Define adjacency lists (each point connects to all others)
    adjl = [[1, 3, 2, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    
    # Create basis functions
    basis = RBF.PHS(3; poly_deg=1)
    mon = RBF.MonomialBasis(2, 1)
    
    Lrb = RBF.∂_Hermite(basis,1)
    Lmb = RBF.∂_Hermite(mon,1)
    
    @testset "_preallocate_IJV_matrices" begin
        # Using the directly accessed function
        (I_lhs, J_lhs, V_lhs), (I_rhs, J_rhs, V_rhs) = _preallocate_IJV_matrices(
            adjl, data, boundary_flag, Lrb)
        
        # Check dimensions
        @test length(I_lhs) == 4  # Two internal nodes, each with ONE internal neighbor
        @test length(J_lhs) == 4
        @test size(V_lhs) == (4, 1)  # Single operator
        
        println(I_lhs, J_lhs, V_lhs)


        @test length(I_rhs) == 4  # Two internal nodes, each with TWO boundary neighbors
        @test length(J_rhs) == 4
        @test size(V_rhs) == (4, 1)
        
        println(I_rhs, J_rhs, V_rhs)

        # Check values
        @test (1,2) in zip(I_lhs, J_lhs)  # Node 1 connects to node 2
        @test (2,1) in zip(I_lhs, J_lhs)  # Node 2 connects to node 1
        
        @test (1,1) in zip(I_rhs, J_rhs)  # Node 1 connects to first boundary node (previously node 3)
        @test (1,2) in zip(I_rhs, J_rhs)  # Node 1 connects to second boundary node (previously node 4)
        @test (2,1) in zip(I_rhs, J_rhs)  # Node 2 connects to first boundary node (previously node 3)
        @test (2,2) in zip(I_rhs, J_rhs)  # Node 2 connects to second boundary node (previously node 4)
    end
    
    @testset "_calculate_thread_offsets" begin
        nchunks = 2
        lhs_offsets, rhs_offsets = _calculate_thread_offsets(adjl, boundary_flag, nchunks)
        
        # Check that offsets are calculated correctly
        @test length(lhs_offsets) == nchunks
        @test length(rhs_offsets) == nchunks
        @test lhs_offsets[1] == 0  # First thread starts at index 0
        @test rhs_offsets[1] == 0  # First thread starts at index 0
        @test lhs_offsets[2] == 4  # Second thread starts after the 2 internal connections from thread 1
        @test rhs_offsets[2] == 4  # Second thread starts after the 4 boundary connections from thread 1
    end
    
    @testset "_update_stencil!" begin
        TD = Float64
        dim = 2
        k = 4  # Number of neighbors
        nmon = 3  # 1 + x + y
        num_ops = 1
        
        stencil = StencilData(TD, dim, k+nmon, k, num_ops)
        
        i = 1  # Test for first node
        _update_stencil!(
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
            k
        )
        
        # Check stencil data was updated
        @test stencil.d[1] == data[1]
        @test stencil.d[2] == data[3]
        @test stencil.d[3] == data[2]
        
        # Check boundary flags
        @test stencil.is_boundary[1] == false
        @test stencil.is_boundary[2] == true
        @test stencil.is_boundary[3] == false
        
        # Check Neumann flags
        @test stencil.is_Neumann[1] == false  # Node 2 is not Neumann
        @test stencil.is_Neumann[2] == true   # Node 3 is Neumann
        @test stencil.is_Neumann[3] == false  # Node 4 is not Neumann
        
        # Check that weights were computed
        @test any(stencil.lhs_v .!= 0)  # Some LHS weights should be non-zero
        @test any(stencil.rhs_v .!= 0)  # Some RHS weights should be non-zero
    end
    
    @testset "_write_coefficients_to_global_matrices!" begin
        TD = Float64
        num_ops = 1
        
        # Create sample stencil data with 4 neighbors instead of 3
        stencil = StencilData(TD, 2, 7, 4, num_ops)  # Increased size (4 neighbors + 3 monomial terms)
        
        # Set values for all 4 connections in the correct locations
        stencil.lhs_v[1, 1] = 1.0  # Self connection (node 1 to node 1)
        stencil.lhs_v[3, 1] = 2.0  # Node 1 to node 2 (internal)
        stencil.rhs_v[2, 1] = 3.0  # Node 1 to node 3 (boundary)
        stencil.rhs_v[4, 1] = 4.0  # Node 1 to node 4 (boundary)
        
        # Create target matrices
        V_lhs = zeros(4, num_ops)
        V_rhs = zeros(4, num_ops)
        
        # Start with known indices
        lhs_idx = 1
        rhs_idx = 1
        
        # Use the full adjacency list with self-connection
        new_lhs_idx, new_rhs_idx = _write_coefficients_to_global_matrices!(
            V_lhs, V_rhs, stencil, adjl[1], boundary_flag, lhs_idx, rhs_idx
        )
        
        # Check values were written (exact checking depends on how your function maps them)
        @test any(V_lhs .!= 0)  # At least some values should be non-zero
        @test any(V_rhs .!= 0)  # At least some values should be non-zero
        
        # Count internal and boundary neighbors
        internal_count = count(i -> !boundary_flag[i], adjl[1])
        boundary_count = count(i -> boundary_flag[i], adjl[1])
        
        # Check indices were updated correctly 
        @test new_lhs_idx == lhs_idx + internal_count
        @test new_rhs_idx == rhs_idx + boundary_count
    end
    
    @testset "_return_global_matrices" begin
        # Create sample matrices
        I_lhs = [1, 1]
        J_lhs = [1, 2]
        V_lhs = [1.0, 2.0]
        
        I_rhs = [1, 1]
        J_rhs = [1, 2]
        V_rhs = [3.0, 4.0]
        
        # Call function
        lhs_matrix, rhs_matrix = _return_global_matrices(
            I_lhs, J_lhs, V_lhs, I_rhs, J_rhs, V_rhs, boundary_flag)
        
        # Check that matrices were created correctly
        @test isa(lhs_matrix, SparseMatrixCSC)
        @test isa(rhs_matrix, SparseMatrixCSC)
        @test size(lhs_matrix) == (2, 2)  # 2 internal nodes
        @test size(rhs_matrix) == (2, 2)  # 2 boundary nodes
        
        # Test multiple operators case
        V_lhs_multi = [1.0 2.0; 3.0 4.0]
        V_rhs_multi = [5.0 6.0; 7.0 8.0]
        
        lhs_matrices, rhs_matrices = _return_global_matrices(
            I_lhs, J_lhs, V_lhs_multi, I_rhs, J_rhs, V_rhs_multi, boundary_flag)
        
        @test length(lhs_matrices) == 2  # Two operators
        @test length(rhs_matrices) == 2
        @test size(lhs_matrices[1]) == (2, 2)
        @test size(rhs_matrices[1]) == (2, 2)
    end
    
    @testset "Full integration test" begin
        # Test the complete workflow
        matrices = _build_weights(
            data, normals, boundary_flag, is_Neumann, adjl, basis, Lrb, Lmb, mon)
        
        @test length(matrices) == 2  # Returns (lhs_matrix, rhs_matrix)
        lhs_matrix, rhs_matrix = matrices
        
        @test size(lhs_matrix) == (2, 2)  # 2 internal nodes
        @test size(rhs_matrix) == (2, 2)  # 2 boundary nodes
        
        # Some basic sanity checks
        @test isa(lhs_matrix, SparseMatrixCSC)
        @test isa(rhs_matrix, SparseMatrixCSC)
        @test nnz(lhs_matrix) > 0  # Should have non-zero entries
        @test nnz(rhs_matrix) > 0
    end
end