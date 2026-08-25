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
    basis_phs = PHS(3; poly_deg = 1)
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
        @testset "ELL Stencil Weight Storage" begin
            # Weights are stored as dense ELL StencilWeights: vals/idx are k x N_eval,
            # logical size is (N_eval, N_data)
            k = 3
            adjl = RBF.find_neighbors(data_1d, eval_points_1d, k)
            ℒ = RBF.Custom{0}(basis -> (x1, x2) -> basis(x1, x2))

            weights = RBF._build_weights(ℒ, data_1d, eval_points_1d, adjl, basis_phs)

            @test weights isa StencilWeights
            @test size(parent(weights)) == (k, length(eval_points_1d))
            @test size(weights) == (length(eval_points_1d), length(data_1d))
            for i in eachindex(adjl)
                @test Int32.(adjl[i]) == weights.idx[:, i]  # columns follow the adjacency list
            end

            # Mixed boundaries (Hermite path): Dirichlet rows collapse to a single
            # identity entry on the diagonal when converted to CSC
            is_boundary = [true, false, false, false, true]
            bcs = [RBF.Dirichlet(), RBF.Dirichlet()]  # Only boundary conditions for boundary points
            normals = [SVector(-1.0), SVector(1.0)]
            adjl_mixed = RBF.find_neighbors(data_1d, data_1d, k)

            weights_mixed = RBF._build_weights(
                ℒ, data_1d, data_1d, adjl_mixed, basis_phs, is_boundary, bcs, normals
            )

            @test weights_mixed isa StencilWeights
            S = sparse(weights_mixed)
            for i in findall(is_boundary)
                @test nnz(S[i, :]) == 1  # Dirichlet rows have a single stored entry
                @test S[i, i] == 1.0
            end

            # Neumann rows keep the full k-entry stencil (only Dirichlet rows are padded)
            bcs_neumann = [RBF.Dirichlet(), RBF.Neumann()]
            weights_neumann = RBF._build_weights(
                RBF.Laplacian(), data_1d, data_1d, adjl_mixed, basis_phs,
                is_boundary, bcs_neumann, normals,
            )
            @test weights_neumann isa StencilWeights
            @test all(isfinite, parent(weights_neumann))
            Sn = sparse(weights_neumann)
            @test nnz(Sn[1, :]) == 1      # Dirichlet endpoint: identity row
            @test nnz(Sn[5, :]) == k      # Neumann endpoint: full stencil
        end

        @testset "Global to Boundary Index Mapping" begin
            # Test construct_global_to_boundary function

            # Case with some boundary points
            is_boundary = [false, true, false, true, false, true]
            global_to_boundary = RBF.construct_global_to_boundary(is_boundary)

            @test length(global_to_boundary) == length(is_boundary)
            @test global_to_boundary[2] == 1  # First boundary point
            @test global_to_boundary[4] == 2  # Second boundary point
            @test global_to_boundary[6] == 3  # Third boundary point
            @test global_to_boundary[1] == 0  # Interior point (not used)
            @test global_to_boundary[3] == 0  # Interior point (not used)
            @test global_to_boundary[5] == 0  # Interior point (not used)

            # Case with no boundary points
            is_boundary_none = [false, false, false]
            global_to_boundary_none = RBF.construct_global_to_boundary(is_boundary_none)
            @test all(global_to_boundary_none .== 0)

            # Case with all boundary points
            is_boundary_all = [true, true, true]
            global_to_boundary_all = RBF.construct_global_to_boundary(is_boundary_all)
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
                ℒ = RBF.Custom{0}(basis -> (x1, x2) -> basis(x1, x2))

                # Test that basic weight building works
                @test_nowarn weights = RBF._build_weights(ℒ, data, eval_points, adjl, basis)

                weights = RBF._build_weights(ℒ, data, eval_points, adjl, basis)
                # _build_weights returns StencilWeights with logical shape (n_eval_points, n_data_points)
                @test size(weights, 1) == length(eval_points)
                @test size(weights, 2) == length(data)
                @test all(isfinite, parent(weights))  # Check stencil values are finite
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
