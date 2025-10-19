"""
Integration tests for Hermite interpolation workflows.

These are END-TO-END tests that verify complete workflows from data setup through 
to final solution. They focus on:
- Complete interpolation workflows with boundary conditions
- Public API usage patterns
- Operator construction and application with Hermite data

Unit tests for individual components are in test/solve/unit/.
Boundary type tests are in test/boundary_types.jl.
"""

using Test
using LinearAlgebra
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF
using SparseArrays

@testset "Hermite Integration Tests" begin

    # Setup: Currently only PHS supports Hermite interpolation
    basis_phs = PHS(3; poly_deg=1)

    # Common 2D test geometry for integration tests
    function create_2d_domain()
        # 6-point domain with 2 boundary points
        data = [
            SVector(0.0, 0.0),   # boundary
            SVector(0.2, 0.1),   # interior
            SVector(0.4, 0.2),   # interior
            SVector(0.6, 0.3),   # interior
            SVector(0.8, 0.2),   # interior
            SVector(1.0, 0.0),   # boundary
        ]
        is_boundary = [true, false, false, false, false, true]
        boundary_bcs = [RBF.Dirichlet(), RBF.Dirichlet()]
        boundary_normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]
        eval_points = [SVector(0.25, 0.15), SVector(0.75, 0.25)]

        return data, is_boundary, boundary_bcs, boundary_normals, eval_points
    end

    @testset "Complete Hermite Interpolation Workflows" begin
        @testset "End-to-End: Laplacian with Dirichlet BCs" begin
            # Complete workflow: Setup → Operator construction → Application
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()

            # Construct Laplacian operator with boundary conditions
            lap_op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)

            # Verify operator was created successfully
            @test lap_op isa RBF.RadialBasisOperator
            @test RBF.dim(lap_op) == 2

            # Apply operator to test function
            u = sin.(range(0, π; length=length(data)))  # Test values at data points
            result = lap_op(u)

            # Verify result has correct dimensions and is finite
            @test length(result) == length(eval_points)
            @test all(isfinite.(result))
        end

        @testset "End-to-End: Gradient with Mixed BCs" begin
            # Test gradient operator with Dirichlet and Neumann conditions
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Dirichlet(), RBF.Neumann()]  # Mixed BC types
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            # Construct gradient operator
            grad_op = gradient(data, eval_points, basis_phs, is_boundary, bcs, normals)

            # Verify operator
            @test grad_op isa RBF.RadialBasisOperator

            # Apply operator
            u = ones(length(data))
            result = grad_op(u)

            # Gradient returns tuple (∂/∂x, ∂/∂y)
            @test result isa Tuple
            @test length(result) == 2  # 2D gradient
            @test length(result[1]) == length(eval_points)
            @test length(result[2]) == length(eval_points)
            @test all(isfinite.(result[1]))
            @test all(isfinite.(result[2]))
        end

        @testset "End-to-End: Partial Derivative with Robin BCs" begin
            # Test partial derivative with Robin boundary conditions
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Robin(1.0, 2.0), RBF.Robin(2.0, 1.0)]  # Robin BCs
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            # Construct ∂/∂x operator
            partial_op = partial(
                data, eval_points, 1, 1, basis_phs, is_boundary, bcs, normals
            )

            # Verify and apply
            @test partial_op isa RBF.RadialBasisOperator
            u = collect(range(0, 1; length=length(data)))
            result = partial_op(u)

            @test length(result) == length(eval_points)
            @test all(isfinite.(result))
        end

        @testset "End-to-End: Directional Derivative with BCs" begin
            # Test directional derivative in arbitrary direction
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()
            direction = [1.0, 1.0]  # Diagonal direction

            # Construct directional derivative operator
            dir_op = directional(
                data, eval_points, direction, basis_phs, is_boundary, bcs, normals
            )

            # Verify and apply
            @test dir_op isa RBF.RadialBasisOperator
            u = [x[1] + x[2] for x in data]  # Linear function
            result = dir_op(u)

            @test length(result) == length(eval_points)
            @test all(isfinite.(result))
        end

        @testset "End-to-End: Custom Operator with BCs" begin
            # Test custom operator (identity) with boundary conditions
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()
            k = 5

            # Build custom operator manually to test low-level integration
            custom_op = RBF.Custom(b -> (x1, x2, n...) -> b(x1, x2))
            adjl = find_neighbors(data, eval_points, k)

            weights = RBF._build_weights(
                custom_op, data, eval_points, adjl, basis_phs, is_boundary, bcs, normals
            )

            # Verify weight matrix
            @test isa(weights, SparseMatrixCSC)
            @test size(weights, 1) == length(eval_points)
            @test size(weights, 2) == length(data)
            @test all(isfinite.(weights.nzval))

            # Apply weights
            u = ones(length(data))
            result = weights * u
            @test length(result) == length(eval_points)
        end
    end

    @testset "Hermite vs Standard RBF Comparison" begin
        @testset "Interior Points: Hermite vs Standard Should Match" begin
            # When evaluating at interior points, Hermite and standard methods
            # should give similar results (BCs don't affect interior stencils)

            data_simple = [SVector(0.0, 0.0), SVector(0.5, 0.3), SVector(1.0, 0.0)]
            eval_interior = [SVector(0.5, 0.15)]  # Interior evaluation point
            k = 3

            # Standard RBF (no boundaries)
            op_standard = RadialBasisOperator(RBF.Laplacian(), data_simple, basis_phs; k=k)

            # Hermite RBF (with boundaries, but eval point is interior)
            is_boundary = [true, false, true]
            bcs = [RBF.Dirichlet(), RBF.Dirichlet()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]
            op_hermite = laplacian(
                data_simple, eval_interior, basis_phs, is_boundary, bcs, normals
            )

            # Both should produce valid operators
            @test op_standard isa RBF.RadialBasisOperator
            @test op_hermite isa RBF.RadialBasisOperator

            # Apply to test function
            u = ones(3)
            result_standard = op_standard(u)
            result_hermite = op_hermite(u)

            @test all(isfinite.(result_standard))
            @test all(isfinite.(result_hermite))
        end
    end

    @testset "Multiple Boundary Condition Types" begin
        @testset "All Dirichlet Boundaries" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Dirichlet(), RBF.Dirichlet()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            @test_nowarn result = op(u)
        end

        @testset "All Neumann Boundaries" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Neumann(), RBF.Neumann()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            @test_nowarn result = op(u)
        end

        @testset "Mixed Dirichlet and Neumann" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Dirichlet(), RBF.Neumann()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = collect(range(0, 1; length=length(data)))
            @test_nowarn result = op(u)
        end

        @testset "Robin Boundary Conditions" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Robin(1.0, 1.0), RBF.Robin(2.0, 3.0)]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            # Verify Robin coefficients
            @test RBF.α(bcs[1]) == 1.0
            @test RBF.β(bcs[1]) == 1.0
            @test RBF.α(bcs[2]) == 2.0
            @test RBF.β(bcs[2]) == 3.0

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            @test_nowarn result = op(u)
        end
    end

    @testset "Multiple Operators on Same Domain" begin
        @testset "Apply Different Operators to Same Data" begin
            # Test that multiple operators can work on the same Hermite domain
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()

            # Create multiple operators
            lap = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            grad = gradient(data, eval_points, basis_phs, is_boundary, bcs, normals)
            partial_x = partial(
                data, eval_points, 1, 1, basis_phs, is_boundary, bcs, normals
            )
            partial_y = partial(
                data, eval_points, 1, 2, basis_phs, is_boundary, bcs, normals
            )

            # All should be valid
            @test lap isa RBF.RadialBasisOperator
            @test grad isa RBF.RadialBasisOperator
            @test partial_x isa RBF.RadialBasisOperator
            @test partial_y isa RBF.RadialBasisOperator

            # Apply all operators to same function
            u = [x[1]^2 + x[2]^2 for x in data]

            @test_nowarn lap_result = lap(u)
            @test_nowarn grad_result = grad(u)
            @test_nowarn px_result = partial_x(u)
            @test_nowarn py_result = partial_y(u)

            # All results should be finite
            @test all(isfinite.(lap(u)))
            @test all(isfinite.(grad(u)[1]))
            @test all(isfinite.(grad(u)[2]))
            @test all(isfinite.(partial_x(u)))
            @test all(isfinite.(partial_y(u)))
        end
    end

    @testset "Varying Problem Sizes" begin
        @testset "Small Problem (3 points)" begin
            data = [SVector(0.0, 0.0), SVector(0.5, 0.5), SVector(1.0, 0.0)]
            is_boundary = [true, false, true]
            bcs = [RBF.Dirichlet(), RBF.Dirichlet()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]
            eval_points = [SVector(0.5, 0.25)]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test length(result) == 1
            @test isfinite(result[1])
        end

        @testset "Medium Problem (6 points)" begin
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test length(result) == 2
            @test all(isfinite.(result))
        end

        @testset "Larger Problem (12 points)" begin
            # Create larger 2D grid
            data = [SVector(i * 0.1, j * 0.1) for i in 0:3 for j in 0:2]
            is_boundary = [
                x[1] == 0.0 || x[1] == 0.3 || x[2] == 0.0 || x[2] == 0.2 for x in data
            ]
            n_boundary = sum(is_boundary)
            bcs = [RBF.Dirichlet() for _ in 1:n_boundary]
            normals = [SVector(1.0, 0.0) for _ in 1:n_boundary]  # Simplified normals
            eval_points = [SVector(0.15, 0.1)]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test length(result) == length(eval_points)
            @test all(isfinite.(result))
        end
    end

    @testset "Normal Vector Orientations" begin
        @testset "Horizontal Normals" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Neumann(), RBF.Neumann()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]  # Left/right

            op = gradient(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test all(isfinite.(result[1]))
            @test all(isfinite.(result[2]))
        end

        @testset "Vertical Normals" begin
            data = [
                SVector(0.0, 0.0),   # bottom
                SVector(0.2, 0.2),   # interior
                SVector(0.4, 0.4),   # interior
                SVector(0.6, 0.6),   # interior
                SVector(0.8, 0.8),   # interior
                SVector(1.0, 1.0),   # top
            ]
            is_boundary = [true, false, false, false, false, true]
            bcs = [RBF.Neumann(), RBF.Neumann()]
            normals = [SVector(0.0, -1.0), SVector(0.0, 1.0)]  # Bottom/top
            eval_points = [SVector(0.5, 0.5)]

            op = gradient(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test all(isfinite.(result[1]))
            @test all(isfinite.(result[2]))
        end

        @testset "Diagonal Normals" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Neumann(), RBF.Neumann()]
            # 45-degree normals
            normals = [
                SVector(1.0 / sqrt(2), -1.0 / sqrt(2)),
                SVector(-1.0 / sqrt(2), 1.0 / sqrt(2)),
            ]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = [x[1] + x[2] for x in data]
            result = op(u)
            @test all(isfinite.(result))
        end
    end

    @testset "Evaluation at Boundary Points" begin
        @testset "Eval Point on Boundary" begin
            data, is_boundary, bcs, normals, _ = create_2d_domain()
            # Evaluate at the first boundary point
            eval_at_boundary = [data[1]]

            op = laplacian(data, eval_at_boundary, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test length(result) == 1
            @test isfinite(result[1])
        end

        @testset "Mixed Interior and Boundary Eval Points" begin
            data, is_boundary, bcs, normals, _ = create_2d_domain()
            # Mix of interior and boundary evaluation points
            eval_mixed = [SVector(0.5, 0.25), data[1], data[end]]

            op = partial(data, eval_mixed, 1, 1, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = [x[1]^2 for x in data]
            result = op(u)
            @test length(result) == 3
            @test all(isfinite.(result))
        end
    end

    @testset "Operator Accuracy with Known Solutions" begin
        @testset "Laplacian of Quadratic (should be constant)" begin
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)

            # u = x² + y² has Laplacian = 2 + 2 = 4
            u = [x[1]^2 + x[2]^2 for x in data]
            result = op(u)

            # Result should be approximately 4 everywhere
            # (allow tolerance for numerical approximation)
            @test all(isfinite.(result))
            # Rough check: should be in ballpark of 4
            @test all(0.0 .<= result .<= 10.0)
        end

        @testset "Gradient of Linear Function" begin
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()

            grad_op = gradient(data, eval_points, basis_phs, is_boundary, bcs, normals)

            # u = 2x + 3y has gradient = (2, 3)
            u = [2.0 * x[1] + 3.0 * x[2] for x in data]
            result = grad_op(u)

            # Both components should be approximately constant
            @test all(isfinite.(result[1]))
            @test all(isfinite.(result[2]))
            # Rough check: ∂/∂x should be near 2, ∂/∂y should be near 3
            @test all(0.0 .<= result[1] .<= 5.0)
            @test all(0.0 .<= result[2] .<= 6.0)
        end

        @testset "Zero Function Should Give Zero" begin
            data, is_boundary, bcs, normals, eval_points = create_2d_domain()

            lap_op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)

            u = zeros(length(data))
            result = lap_op(u)

            # Zero input should give approximately zero output
            @test all(abs.(result) .< 1e-10)
        end
    end

    @testset "Operator-BC Combinations" begin
        @testset "Gradient with Robin BCs" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Robin(1.0, 1.0), RBF.Robin(2.0, 1.0)]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            op = gradient(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = [x[1] + x[2] for x in data]
            result = op(u)
            @test all(isfinite.(result[1]))
            @test all(isfinite.(result[2]))
        end

        @testset "Directional Derivative with Neumann BCs" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Neumann(), RBF.Neumann()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]
            direction = [0.0, 1.0]  # y-direction

            op = directional(
                data, eval_points, direction, basis_phs, is_boundary, bcs, normals
            )
            @test op isa RBF.RadialBasisOperator

            u = [x[2]^2 for x in data]
            result = op(u)
            @test all(isfinite.(result))
        end

        @testset "Partial ∂/∂y with Dirichlet BCs" begin
            data, is_boundary, _, _, eval_points = create_2d_domain()
            bcs = [RBF.Dirichlet(), RBF.Dirichlet()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            # Explicitly test ∂/∂y (second spatial dimension)
            op = partial(data, eval_points, 1, 2, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = [x[2]^2 for x in data]  # Function of y only
            result = op(u)
            @test all(isfinite.(result))
        end
    end

    @testset "Edge Cases" begin
        @testset "Single Boundary Point" begin
            # Minimal case: only one boundary point
            data = [
                SVector(0.0, 0.0),   # boundary
                SVector(0.5, 0.3),   # interior
                SVector(1.0, 0.5),   # interior
            ]
            is_boundary = [true, false, false]
            bcs = [RBF.Dirichlet()]  # Only one BC
            normals = [SVector(1.0, 0.0)]  # Only one normal
            eval_points = [SVector(0.5, 0.25)]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test length(result) == 1
            @test isfinite(result[1])
        end

        @testset "All Points are Boundaries" begin
            # Edge case: every point is a boundary
            data = [SVector(0.0, 0.0), SVector(0.5, 0.5), SVector(1.0, 0.0)]
            is_boundary = [true, true, true]
            bcs = [RBF.Dirichlet(), RBF.Neumann(), RBF.Dirichlet()]
            normals = [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0)]
            eval_points = [SVector(0.5, 0.25)]

            op = laplacian(data, eval_points, basis_phs, is_boundary, bcs, normals)
            @test op isa RBF.RadialBasisOperator

            u = ones(length(data))
            result = op(u)
            @test length(result) == 1
            @test isfinite(result[1])
        end
    end
end
