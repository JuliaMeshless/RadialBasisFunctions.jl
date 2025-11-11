"""
Integration tests for Hermite interpolation workflows.

These tests verify that RBF operators with Hermite boundary conditions
correctly reproduce polynomial functions at interior points. With poly_deg=2,
the method should exactly reproduce polynomials up to degree 2.

Focus: Exact operator reconstruction in the presence of boundary conditions.
"""

using Test
using LinearAlgebra
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF
using SparseArrays

@testset "Hermite Integration Tests" begin

    # Setup: Currently only PHS supports Hermite interpolation
    basis_phs = PHS(3; poly_deg=2)

    # Common 2D test geometry for integration tests
    function create_2d_domain()
        # 12-point domain with 2 boundary points
        data = [
            SVector(0.0, 0.0),   # boundary
            SVector(0.15, 0.1),  # interior
            SVector(0.2, 0.25),  # interior
            SVector(0.3, 0.15),  # interior
            SVector(0.4, 0.3),   # interior
            SVector(0.5, 0.2),   # interior
            SVector(0.6, 0.35),  # interior
            SVector(0.7, 0.25),  # interior
            SVector(0.75, 0.15), # interior
            SVector(0.85, 0.3),  # interior
            SVector(0.9, 0.2),   # interior
            SVector(1.0, 0.0),   # boundary
        ]
        is_boundary = [
            true, false, false, false, false, false, false, false, false, false, false, true
        ]
        boundary_bcs = [RBF.Dirichlet(), RBF.Dirichlet()]
        boundary_normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

        return data, is_boundary, boundary_bcs, boundary_normals
    end

    @testset "Exact Polynomial Reproduction with Boundary Conditions" begin
        @testset "Laplacian of Quadratic (Dirichlet BCs)" begin
            data, is_boundary, bcs, normals = create_2d_domain()

            # Evaluate operator at data points
            op = laplacian(data, data, basis_phs, is_boundary, bcs, normals)

            # u = x² + y² has Laplacian = 2 + 2 = 4 everywhere
            u = [x[1]^2 + x[2]^2 for x in data]
            result = op(u)

            # At interior points, Laplacian should be exactly 4.0
            interior_indices = findall(.!is_boundary)
            interior_results = result[interior_indices]

            @test all(isfinite.(result))
            @test all(abs.(interior_results .- 4.0) .< 1e-10)

            # At Dirichlet boundary points, operator returns function value
            boundary_indices = findall(is_boundary)
            @test result[boundary_indices[1]] ≈ u[boundary_indices[1]]
            @test result[boundary_indices[2]] ≈ u[boundary_indices[2]]
        end

        @testset "Laplacian of Quadratic (Neumann BCs)" begin
            data, is_boundary, _, _ = create_2d_domain()
            # Boundary points are at (0,0) and (1,0) with normals (1,0) and (-1,0)
            bcs = [RBF.Neumann(), RBF.Neumann()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            # Use u = y² which has ∂u/∂n = 0 at both boundaries (since ∇u = (0, 2y) and normals are in x-direction)
            # Laplacian of y² is 2 everywhere
            op = laplacian(data, data, basis_phs, is_boundary, bcs, normals)
            u = [x[2]^2 for x in data]
            result = op(u)

            # Interior points: Laplacian should be exactly 2.0
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices] .- 2.0) .< 1e-10)
        end

        @testset "Laplacian of Quadratic (Mixed Dirichlet/Neumann)" begin
            data, is_boundary, _, _ = create_2d_domain()
            # First boundary at (0,0): Dirichlet, Second boundary at (1,0): Neumann with normal (-1,0)
            bcs = [RBF.Dirichlet(), RBF.Neumann()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            # Use u = y² which has ∂u/∂n = 0 at second boundary
            # Laplacian of y² is 2 everywhere
            op = laplacian(data, data, basis_phs, is_boundary, bcs, normals)
            u = [x[2]^2 for x in data]
            result = op(u)

            # Interior points: Laplacian should be exactly 2.0
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices] .- 2.0) .< 1e-10)
        end

        @testset "Laplacian of Quadratic (Robin BCs)" begin
            data, is_boundary, _, _ = create_2d_domain()
            # Robin BC: αu + β(∂u/∂n) = 0
            # Use u = y² which has ∂u/∂n = 0 at both boundaries
            # So Robin BC becomes: α*y² + β*0 = 0, which means y² = 0 at boundaries
            # Since boundaries are at y=0, this is satisfied!
            bcs = [RBF.Robin(1.0, 1.0), RBF.Robin(2.0, 3.0)]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            op = laplacian(data, data, basis_phs, is_boundary, bcs, normals)
            u = [x[2]^2 for x in data]
            result = op(u)

            # Interior points: Laplacian should be exactly 2.0
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices] .- 2.0) .< 1e-10)
        end

        @testset "Gradient of Linear Function (Dirichlet BCs)" begin
            data, is_boundary, bcs, normals = create_2d_domain()

            op = gradient(data, data, basis_phs, is_boundary, bcs, normals)

            # u = 2x + 3y has gradient = (2, 3)
            u = [2.0 * x[1] + 3.0 * x[2] for x in data]
            result = op(u)

            # At interior points, gradient should be exactly (2, 3)
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[1][interior_indices] .- 2.0) .< 1e-10)
            @test all(abs.(result[2][interior_indices] .- 3.0) .< 1e-10)
        end

        @testset "Gradient of Linear Function (Neumann BCs)" begin
            data, is_boundary, _, _ = create_2d_domain()
            bcs = [RBF.Neumann(), RBF.Neumann()]
            normals = [SVector(1.0, 0.0), SVector(-1.0, 0.0)]

            # Use u = 3y which has gradient = (0, 3) and ∂u/∂n = 0 at both boundaries
            op = gradient(data, data, basis_phs, is_boundary, bcs, normals)
            u = [3.0 * x[2] for x in data]
            result = op(u)

            # Interior points: gradient should be exactly (0, 3)
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[1][interior_indices] .- 0.0) .< 1e-10)
            @test all(abs.(result[2][interior_indices] .- 3.0) .< 1e-10)
        end

        @testset "Partial ∂/∂x of Linear Function" begin
            data, is_boundary, bcs, normals = create_2d_domain()

            # ∂/∂x operator
            op = partial(data, data, 1, 1, basis_phs, is_boundary, bcs, normals)

            # u = 2x + 3y has ∂u/∂x = 2
            u = [2.0 * x[1] + 3.0 * x[2] for x in data]
            result = op(u)

            # Interior points: exact ∂/∂x = 2.0
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices] .- 2.0) .< 1e-10)
        end

        @testset "Partial ∂/∂y of Linear Function" begin
            data, is_boundary, bcs, normals = create_2d_domain()

            # ∂/∂y operator
            op = partial(data, data, 1, 2, basis_phs, is_boundary, bcs, normals)

            # u = 2x + 3y has ∂u/∂y = 3
            u = [2.0 * x[1] + 3.0 * x[2] for x in data]
            result = op(u)

            # Interior points: exact ∂/∂y = 3.0
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices] .- 3.0) .< 1e-10)
        end

        @testset "Directional Derivative (Single Direction)" begin
            data, is_boundary, bcs, normals = create_2d_domain()

            # Direction vector in x-direction
            v = SVector(1.0, 0.0)
            op = directional(data, data, v, basis_phs, is_boundary, bcs, normals)

            # u = 2x + 3y has gradient = (2, 3), directional derivative in x-direction = 2
            u = [2.0 * x[1] + 3.0 * x[2] for x in data]
            result = op(u)

            # Interior points: exact directional derivative = 2.0
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices] .- 2.0) .< 1e-10)
        end

        @testset "Directional Derivative (Per-Point Vectors)" begin
            data, is_boundary, bcs, normals = create_2d_domain()

            # Per-point direction vectors (x-direction for all points)
            v = [SVector(1.0, 0.0) for _ in 1:length(data)]
            op = directional(data, data, v, basis_phs, is_boundary, bcs, normals)

            # u = 2x + 3y has gradient = (2, 3), directional derivative in x-direction = 2
            u = [2.0 * x[1] + 3.0 * x[2] for x in data]
            result = op(u)

            # Interior points: exact directional derivative = 2.0
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices] .- 2.0) .< 1e-10)
        end

        @testset "Zero Function Gives Zero" begin
            data, is_boundary, bcs, normals = create_2d_domain()

            lap_op = laplacian(data, data, basis_phs, is_boundary, bcs, normals)
            u = zeros(length(data))
            result = lap_op(u)

            # Interior points should give exactly zero
            interior_indices = findall(.!is_boundary)
            @test all(abs.(result[interior_indices]) .< 1e-10)
        end
    end

    @testset "Edge Cases" begin
        @testset "All Interior Points" begin
            # No boundary points - should behave like standard RBF
            data = [SVector(0.1 * i, 0.1 * j) for i in 1:3 for j in 1:3]
            is_boundary = fill(false, length(data))
            bcs = BoundaryCondition{Float64}[]  # Empty - properly typed
            normals = SVector{2,Float64}[]  # Empty

            op = laplacian(data, data, basis_phs, is_boundary, bcs, normals)
            u = [x[1]^2 + x[2]^2 for x in data]
            result = op(u)

            # All points should give Laplacian ≈ 4.0
            @test all(abs.(result .- 4.0) .< 1e-9)
        end
    end
end
