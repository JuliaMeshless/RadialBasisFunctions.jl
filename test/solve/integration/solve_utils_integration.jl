"""
Integration tests for solve_utils.jl functionality.
Tests the unified kernel infrastructure and utility functions.
Focuses on what actually works with the current API rather than internal implementation details.

CURRENT LIMITATION: Advanced boundary conditions and Hermite functionality is PHS-only.
When IMQ/Gaussian get the required operators, expand hermite_compatible_bases below.
"""

using Test
using LinearAlgebra
using StaticArraysCore
using SparseArrays
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Solve Utils Integration Tests" begin

    # Test setup - basis function and data configuration
    basis_phs = PHS(3; poly_deg=1)
    basis_imq = IMQ(1.0)
    basis_gaussian = Gaussian(1.0)

    # IMPORTANT: Hermite functionality is currently PHS-only
    hermite_compatible_bases = [basis_phs]  # TODO: Add basis_imq, basis_gaussian when operators are implemented
    all_bases = [basis_phs, basis_imq, basis_gaussian]  # For standard tests

    # Test data configurations using SVector format (consistent with codebase) 
    data_1d = [SVector(0.0), SVector(0.5), SVector(1.0), SVector(1.5), SVector(2.0)]
    data_2d = [
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
        SVector(1.0, 1.0),
        SVector(0.5, 0.5),
        SVector(1.5, 0.5),
    ]
    eval_points_1d = [SVector(0.25), SVector(0.75), SVector(1.25)]
    eval_points_2d = [SVector(0.5, 0.5), SVector(0.25, 0.75)]

    @testset "Utility Function Testing" begin
        @testset "Non-zero Counting for Sparse Allocation" begin
            # Test _count_nonzeros function with various boundary configurations

            # Simple case: no boundaries (all interior)
            adjl_simple = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            is_boundary_none = [false, false, false, false, false]
            bcs_none = [RBF.Dirichlet() for _ in 1:0]  # No boundary conditions needed

            total_nnz, nnz_per_row, row_offsets = RBF._count_nonzeros(
                adjl_simple, is_boundary_none, bcs_none
            )

            @test length(nnz_per_row) == length(adjl_simple)
            @test length(row_offsets) == length(adjl_simple) + 1
            @test total_nnz == sum(nnz_per_row)
            @test all(nnz_per_row .== 3)  # All stencils have 3 points

            # Complex case: mixed boundaries
            is_boundary_mixed = [true, false, true, false, false]
            bcs_mixed = [RBF.Dirichlet(), RBF.Neumann()]  # Only boundary conditions for boundary points

            total_nnz_mixed, nnz_per_row_mixed, row_offsets_mixed = RBF._count_nonzeros(
                adjl_simple, is_boundary_mixed, bcs_mixed
            )

            @test length(nnz_per_row_mixed) == length(adjl_simple)
            @test total_nnz_mixed == sum(nnz_per_row_mixed)
            @test total_nnz_mixed <= total_nnz  # Dirichlet boundaries reduce non-zeros
        end

        @testset "Global to Boundary Index Mapping" begin
            # Test _construct_global_to_boundary function

            # Case with some boundary points
            is_boundary = [false, true, false, true, false, true]
            global_to_boundary = RBF._construct_global_to_boundary(is_boundary)

            @test length(global_to_boundary) == length(is_boundary)
            @test global_to_boundary[2] == 1  # First boundary point
            @test global_to_boundary[4] == 2  # Second boundary point  
            @test global_to_boundary[6] == 3  # Third boundary point
            @test global_to_boundary[1] == 0  # Interior point (not used)
            @test global_to_boundary[3] == 0  # Interior point (not used)
            @test global_to_boundary[5] == 0  # Interior point (not used)

            # Case with no boundary points
            is_boundary_none = [false, false, false]
            global_to_boundary_none = RBF._construct_global_to_boundary(is_boundary_none)
            @test all(global_to_boundary_none .== 0)

            # Case with all boundary points
            is_boundary_all = [true, true, true]
            global_to_boundary_all = RBF._construct_global_to_boundary(is_boundary_all)
            @test global_to_boundary_all == [1, 2, 3]
        end
    end

    @testset "Integration with Standard solve.jl Functions" begin
        @testset "Basic Weight Building Integration" begin
            # Test integration with public APIs from solve.jl
            k = 3

            # Create simple operators and data using SVector format (like rest of codebase)
            data = [SVector(0.0), SVector(0.5), SVector(1.0)]
            eval_points = [SVector(0.25)]

            for basis in all_bases
                # Test basic weight building using public API
                adjl = RBF.find_neighbors(data, eval_points, k)

                # Create identity operator
                ℒ = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))

                # Test that basic weight building works
                @test_nowarn weights = RBF._build_weights(ℒ, data, eval_points, adjl, basis)

                weights = RBF._build_weights(ℒ, data, eval_points, adjl, basis)
                # _build_weights returns a sparse matrix of shape (n_eval_points, n_data_points)
                @test size(weights, 1) == length(eval_points)
                @test size(weights, 2) == length(data)
                @test all(isfinite.(weights.nzval))  # Check non-zero values are finite
            end
        end

        @testset "Neighbor Finding Integration" begin
            @testset "Different Neighborhood Sizes" begin
                # Test utility functions for finding neighbors
                data = [SVector(i * 0.1) for i in 0:10]
                eval_points = [SVector(0.25), SVector(0.75)]

                neighborhood_sizes = [3, 5, 7]

                for k in neighborhood_sizes
                    @test_nowarn adjl = RBF.find_neighbors(data, eval_points, k)

                    adjl = RBF.find_neighbors(data, eval_points, k)
                    @test length(adjl) == length(eval_points)
                    for neighbors in adjl
                        @test length(neighbors) == k
                        @test all(1 .<= neighbors .<= length(data))
                    end
                end
            end
        end
    end
end
