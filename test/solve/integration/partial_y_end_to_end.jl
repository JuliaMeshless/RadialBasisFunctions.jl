using Test
using StaticArraysCore
using LinearAlgebra
using Statistics
using RadialBasisFunctions
import RadialBasisFunctions as RBF

# Helper function for ∂u/∂y = 2y
partial_y_func(x, y) = 2.0 * y

@testset "Partial ∂/∂y End-to-End with Hermite" begin
    domain_2d = create_2d_unit_square_domain(0.05; randomize = true)
    @test length(domain_2d) == 441

    is_boundary = identify_boundary_points(domain_2d)
    n_boundary = count(is_boundary)
    n_interior = length(domain_2d) - n_boundary
    @test n_boundary == 80
    @test n_interior == 361

    basis_phs = RBF.PHS(3; poly_deg = 2)
    mon = RBF.MonomialBasis(2, 2)

    k = RBF.autoselect_k(domain_2d, basis_phs)
    adjl = RBF.find_neighbors(domain_2d, k)
    boundary_conditions = setup_test_boundary_conditions(domain_2d, is_boundary, RBF)
    normals = compute_normals(domain_2d, is_boundary)

    u_values = construct_u_values_hermite(
        domain_2d, is_boundary, boundary_conditions, normals, RBF
    )

    partial_y_op = RBF.Partial(1, 2)

    Dy_op = RBF.RadialBasisOperator(
        partial_y_op,
        domain_2d,
        domain_2d,
        basis_phs,
        is_boundary,
        boundary_conditions,
        normals;
        k = k,
        adjl = adjl,
    )

    @testset "Test 1: Weights Calculation (Forward Problem)" begin
        partial_y_result = Dy_op(u_values)

        partial_y_interior = partial_y_result[.!is_boundary]

        expected_partial_y = Float64[]
        for i in eachindex(domain_2d)
            if !is_boundary[i]
                push!(expected_partial_y, 2.0 * domain_2d[i][2])
            end
        end

        errors = abs.(partial_y_interior - expected_partial_y)
        max_error = maximum(errors)
        rms_error = sqrt(mean(errors .^ 2))

        @test max_error < 1.0e-10  # Machine precision
        @test rms_error < 1.0e-11  # RMS should be even better
    end

    @testset "Test 2: RHS Consistency Verification" begin
        rhs = construct_rhs(
            partial_y_func, domain_2d, is_boundary, boundary_conditions, normals, RBF
        )
        partial_y_result = Dy_op(u_values)

        rhs_error = abs.(rhs - partial_y_result)
        max_rhs_error = maximum(rhs_error)

        @test max_rhs_error < 1.0e-10
    end

    @testset "Test 3: Solution of PDE (Inverse Problem)" begin
        rhs = construct_rhs(
            partial_y_func, domain_2d, is_boundary, boundary_conditions, normals, RBF
        )

        solution = Dy_op.weights \ rhs

        solution_error = solution - u_values
        max_error = maximum(abs.(solution_error))
        rms_error = sqrt(mean(solution_error .^ 2))

        @test max_error < 1.0e-7  # Machine precision
        @test rms_error < 1.0e-8  # RMS should be even better
    end
end
