"""
End-to-end test for Laplacian operator with Hermite interpolation.

This test verifies that:
1. Weights are calculated exactly (machine precision) for the Laplacian operator
2. The manufactured solution is recovered exactly (machine precision)

Test setup:
- 2D domain: unit square [0,1]² with 441 points (21×21 grid)
- Mixed boundary conditions:
  - x=1: Neumann BC
  - x=0: Robin BC (α=0.5, β=0.5)
  - y=0,1: Dirichlet BC
- Target function: u(x,y) = x² + y² + 1
- Laplacian: ∇²u = 4.0
"""

using Test
using StaticArraysCore
using LinearAlgebra
using Statistics
using RadialBasisFunctions
import RadialBasisFunctions as RBF

include("end_to_end_utils.jl")

function construct_rhs_laplacian(domain_2d, is_boundary, boundary_conditions, RBF)
    rhs = [target_laplacian(p[1], p[2]) for p in domain_2d]
    bnd_counter = 0
    for i in eachindex(domain_2d)
        if is_boundary[i]
            bnd_counter += 1
            if RBF.is_dirichlet(boundary_conditions[bnd_counter])
                # Dirichlet BC: RHS is function value
                rhs[i] = target_function(domain_2d[i][1], domain_2d[i][2])
            end
            # For Neumann/Robin: RHS is Laplacian value (already set)
        end
    end
    return rhs
end

@testset "Laplacian End-to-End with Hermite" begin

    # Create 2D domain
    domain_2d = create_2d_unit_square_domain(0.05)
    @test length(domain_2d) == 441

    # Identify boundary points
    is_boundary = identify_boundary_points(domain_2d)
    n_boundary = count(is_boundary)
    n_interior = length(domain_2d) - n_boundary
    @test n_boundary == 80
    @test n_interior == 361

    # Compute normals for boundary points
    normals = compute_normals(domain_2d, is_boundary)
    @test length(normals) == n_boundary

    # Basis configuration
    basis_phs = RBF.PHS(3; poly_deg=2)
    mon = RBF.MonomialBasis(2, 2)

    k = RBF.autoselect_k(domain_2d, basis_phs)
    adjl = RBF.find_neighbors(domain_2d, 15)

    # Set up boundary conditions
    boundary_conditions = setup_mixed_boundary_conditions(domain_2d, is_boundary, RBF)
    @test length(boundary_conditions) == n_boundary

    # Construct u_values for Hermite interpolation
    u_values = construct_u_values_hermite(
        domain_2d, is_boundary, boundary_conditions, normals, RBF
    )

    # Build Laplacian operator with Hermite interpolation
    laplacian_op = RBF.Laplacian()

    L_op = RBF.RadialBasisOperator(
        laplacian_op,
        domain_2d,
        domain_2d,
        basis_phs,
        is_boundary,
        boundary_conditions,
        normals;
        k=k,
        adjl=adjl,
    )

    @testset "Test 1: Weights Calculation (Forward Problem)" begin
        # Apply operator to u_values
        laplacian_result = L_op(u_values)

        # At interior points, should get Laplacian value
        laplacian_result_interior = laplacian_result[.!is_boundary]
        expected_laplacian = [
            target_laplacian(p[1], p[2]) for p in domain_2d[.!is_boundary]
        ]
        laplacian_error = laplacian_result_interior - expected_laplacian

        max_error = maximum(abs.(laplacian_error))
        rms_error = sqrt(mean(laplacian_error .^ 2))

        @test max_error < 1e-10  # Machine precision
        @test rms_error < 1e-11  # RMS should be even better

        println("Forward Problem (Weights Calculation):")
        println("  Laplacian max error: ", max_error)
        println("  Laplacian RMS error: ", rms_error)
    end

    @testset "Test 2: Manufactured Solution (Inverse Problem)" begin
        # Construct RHS for manufactured solution
        rhs = construct_rhs_laplacian(domain_2d, is_boundary, boundary_conditions, RBF)

        # Verify RHS construction by comparing with operator output
        laplacian_result = L_op(u_values)
        rhs_error = abs.(rhs - laplacian_result)
        @test maximum(rhs_error) < 1e-10

        # Solve the system
        solution = L_op.weights \ rhs

        # Solution should match u_values
        solution_error = solution - u_values
        max_error = maximum(abs.(solution_error))
        rms_error = sqrt(mean(solution_error .^ 2))

        @test max_error < 1e-10  # Machine precision
        @test rms_error < 1e-11  # RMS should be even better

        println("Inverse Problem (Manufactured Solution):")
        println("  Solution max error: ", max_error)
        println("  Solution RMS error: ", rms_error)
    end
end
