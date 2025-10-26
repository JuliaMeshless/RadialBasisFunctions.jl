"""
End-to-end test for Gradient operator with Hermite interpolation.

This test verifies that:
1. Weights are calculated exactly (machine precision) for the Gradient operator
2. The manufactured solution is recovered exactly (machine precision)

Test setup:
- 2D domain: unit square [0,1]² with 441 points (21×21 grid)
- Mixed boundary conditions:
  - x=1: Neumann BC
  - x=0: Robin BC (α=0.5, β=0.5)
  - y=0,1: Dirichlet BC
- Target function: u(x,y) = x² + y² + 1
- Gradient: ∇u = (2x, 2y)
"""

using Test
using StaticArraysCore
using LinearAlgebra
using Statistics
using RadialBasisFunctions
import RadialBasisFunctions as RBF

include("end_to_end_utils.jl")

function construct_rhs_gradient(domain_2d, is_boundary, boundary_conditions, RBF)
    N = length(domain_2d)
    rhs_x = zeros(N)
    rhs_y = zeros(N)

    # Set gradient values for all points initially
    for i in eachindex(domain_2d)
        grad = target_gradient(domain_2d[i][1], domain_2d[i][2])
        rhs_x[i] = grad[1]
        rhs_y[i] = grad[2]
    end

    # For Dirichlet boundaries, RHS is the function value
    bnd_counter = 0
    for i in eachindex(domain_2d)
        if is_boundary[i]
            bnd_counter += 1
            if RBF.is_dirichlet(boundary_conditions[bnd_counter])
                func_val = target_function(domain_2d[i][1], domain_2d[i][2])
                rhs_x[i] = func_val
                rhs_y[i] = func_val
            end
        end
    end

    return (rhs_x, rhs_y)
end

@testset "Gradient End-to-End with Hermite" begin

    # Create 2D domain
    domain_2d = create_2d_unit_square_domain(0.05)
    @test length(domain_2d) == 441

    # Identify boundary points
    is_boundary = identify_boundary_points(domain_2d)

    # Basis configuration
    basis_phs = RBF.PHS(3; poly_deg=2)
    mon = RBF.MonomialBasis(2, 2)

    k = RBF.autoselect_k(domain_2d, basis_phs)
    adjl = RBF.find_neighbors(domain_2d, 15)
    boundary_conditions = setup_mixed_boundary_conditions(domain_2d, is_boundary, RBF)
    normals = compute_normals(domain_2d, is_boundary)
    # Construct u_values for Hermite interpolation
    u_values = construct_u_values_hermite(
        domain_2d, is_boundary, boundary_conditions, normals, RBF
    )

    gradient_op = RBF.Gradient{2}()

    G_op = RBF.RadialBasisOperator(
        gradient_op,
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
        gradient_result = G_op(u_values)

        # At interior points, should get gradient values
        gradient_x_interior = gradient_result[1][.!is_boundary]
        gradient_y_interior = gradient_result[2][.!is_boundary]

        expected_gradient_x = Float64[]
        expected_gradient_y = Float64[]
        for i in eachindex(domain_2d)
            if !is_boundary[i]
                grad = target_gradient(domain_2d[i][1], domain_2d[i][2])
                push!(expected_gradient_x, grad[1])
                push!(expected_gradient_y, grad[2])
            end
        end

        gradient_error_x = gradient_x_interior - expected_gradient_x
        gradient_error_y = gradient_y_interior - expected_gradient_y

        max_error_x = maximum(abs.(gradient_error_x))
        max_error_y = maximum(abs.(gradient_error_y))
        rms_error_x = sqrt(mean(gradient_error_x .^ 2))
        rms_error_y = sqrt(mean(gradient_error_y .^ 2))

        max_error = max(max_error_x, max_error_y)
        rms_error = max(rms_error_x, rms_error_y)

        @test max_error < 1e-10  # Machine precision
        @test rms_error < 1e-11  # RMS should be even better
    end

    @testset "Test 2: RHS Consistency Verification" begin
        # For gradient operator, verify that RHS construction matches operator output
        # This ensures the manufactured solution setup is correct
        rhs = construct_rhs_gradient(domain_2d, is_boundary, boundary_conditions, RBF)
        gradient_result = G_op(u_values)

        rhs_error_x = abs.(rhs[1] - gradient_result[1])
        rhs_error_y = abs.(rhs[2] - gradient_result[2])
        max_rhs_error = max(maximum(rhs_error_x), maximum(rhs_error_y))

        @test max_rhs_error < 1e-10
    end

    @testset "Test 3: Solution of PDE" begin
        # Test that we can accurately reconstruct gradients at interior points
        # by checking that the weights correctly compute gradients
        gradient_result = G_op(u_values)

        rhs = construct_rhs_gradient(domain_2d, is_boundary, boundary_conditions, RBF)

        # Check matrix conditioning
        println("\n=== Matrix Conditioning ===")
        cond_x = cond(Matrix(G_op.weights[1]))
        cond_y = cond(Matrix(G_op.weights[2]))
        println("Condition number of weights[1]: ", cond_x)
        println("Condition number of weights[2]: ", cond_y)

        # Verify the forward problem is exact
        println("\n=== Verifying Forward Problem ===")
        forward_x = G_op.weights[1] * u_values
        forward_y = G_op.weights[2] * u_values
        forward_error_x = maximum(abs.(forward_x - rhs[1]))
        forward_error_y = maximum(abs.(forward_y - rhs[2]))
        println("Max error in weights[1] * u_values vs rhs[1]: ", forward_error_x)
        println("Max error in weights[2] * u_values vs rhs[2]: ", forward_error_y)

        # Try different solvers
        println("\n=== Testing Different Solvers ===")

        # Method 1: Default backslash (LU decomposition)
        println("\n1. Backslash operator (LU):")
        solution_x_lu = G_op.weights[1] \ rhs[1]
        solution_y_lu = G_op.weights[2] \ rhs[2]
        println("  X max error: ", maximum(abs.(solution_x_lu .- u_values)))
        println("  Y max error: ", maximum(abs.(solution_y_lu .- u_values)))

        # Check residuals
        residual_x = G_op.weights[1] * solution_x - rhs[1]
        residual_y = G_op.weights[2] * solution_y - rhs[2]
        println("\n=== Solution Residuals ===")
        println("Max residual for x: ", maximum(abs.(residual_x)))
        println("Max residual for y: ", maximum(abs.(residual_y)))

        solution_x_error = solution_x - u_values
        solution_y_error = solution_y - u_values

        max_error_x = maximum(abs.(solution_x_error))
        max_error_y = maximum(abs.(solution_y_error))
        rms_error_x = sqrt(mean(solution_x_error .^ 2))
        rms_error_y = sqrt(mean(solution_y_error .^ 2))

        println("\n=== Solution Errors ===")
        println("X-component max error: ", max_error_x)
        println("X-component RMS error: ", rms_error_x)
        println("Y-component max error: ", max_error_y)
        println("Y-component RMS error: ", rms_error_y)

        @test max_error_x < 1e-10  # Machine precision
        @test max_error_y < 1e-10  # Machine precision
        @test rms_error_x < 1e-11  # RMS should be even better
        @test rms_error_y < 1e-11  # RMS should be even
    end
end
