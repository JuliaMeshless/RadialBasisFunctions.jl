
using Test
using StaticArraysCore
using LinearAlgebra
using Statistics
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Gradient End-to-End with Hermite" begin
    domain_2d = create_2d_unit_square_domain(0.05; randomize=true)

    @test length(domain_2d) == 441

    is_boundary = identify_boundary_points(domain_2d)

    basis_phs = RBF.PHS(3; poly_deg=2)
    mon = RBF.MonomialBasis(2, 2)

    k = RBF.autoselect_k(domain_2d, basis_phs)
    adjl = RBF.find_neighbors(domain_2d, k)
    boundary_conditions = setup_test_boundary_conditions(domain_2d, is_boundary, RBF)
    normals = compute_normals(domain_2d, is_boundary)
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
        gradient_result = G_op(u_values)

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

        @test max_error < 1e-10
        @test rms_error < 1e-11
    end

    @testset "Test 2: RHS Consistency Verification" begin
        rhs = construct_rhs(
            target_gradient, domain_2d, is_boundary, boundary_conditions, normals, RBF
        )
        gradient_result = G_op(u_values)

        rhs_error_x = abs.(rhs[1] - gradient_result[1])
        rhs_error_y = abs.(rhs[2] - gradient_result[2])
        max_rhs_error = max(maximum(rhs_error_x), maximum(rhs_error_y))

        @test max_rhs_error < 1e-10
    end

    @testset "Test 3: Solution of PDE" begin
        gradient_result = G_op(u_values)

        rhs = construct_rhs(
            target_gradient, domain_2d, is_boundary, boundary_conditions, normals, RBF
        )

        solution_x = G_op.weights[1] \ rhs[1]
        solution_y = G_op.weights[2] \ rhs[2]

        solution_x_error = solution_x - u_values
        solution_y_error = solution_y - u_values

        max_error_x = maximum(abs.(solution_x_error))
        max_error_y = maximum(abs.(solution_y_error))
        rms_error_x = sqrt(mean(solution_x_error .^ 2))
        rms_error_y = sqrt(mean(solution_y_error .^ 2))

        @test max_error_x < 1e-7
        @test max_error_y < 1e-7
        @test rms_error_x < 1e-8
        @test rms_error_y < 1e-8
    end
end
