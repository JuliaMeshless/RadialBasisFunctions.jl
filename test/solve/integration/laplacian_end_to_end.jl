using Test
using StaticArraysCore
using LinearAlgebra
using Statistics
using RadialBasisFunctions
import RadialBasisFunctions as RBF

function construct_rhs_laplacian(domain_2d, is_boundary, boundary_conditions, normals, RBF)
    rhs = [target_laplacian(p[1], p[2]) for p in domain_2d]
    bnd_counter = 0
    for i in eachindex(domain_2d)
        if is_boundary[i]
            bnd_counter += 1
            bc = boundary_conditions[bnd_counter]
            if RBF.is_dirichlet(bc)
                rhs[i] = target_function(domain_2d[i][1], domain_2d[i][2])
            elseif RBF.is_neumann(bc) || RBF.is_robin(bc)
                α_val = RBF.α(bc)
                β_val = RBF.β(bc)
                u_val = target_function(domain_2d[i][1], domain_2d[i][2])
                ∂ₙu_val = target_Neumann_bc(
                    domain_2d[i][1], domain_2d[i][2], normals[bnd_counter]
                )
                rhs[i] = α_val * u_val + β_val * ∂ₙu_val
            end
        end
    end
    return rhs
end

@testset "Laplacian End-to-End with Hermite" begin
    domain_2d = create_2d_unit_square_domain(0.05)
    @test length(domain_2d) == 441

    is_boundary = identify_boundary_points(domain_2d)
    n_boundary = count(is_boundary)
    n_interior = length(domain_2d) - n_boundary
    @test n_boundary == 80
    @test n_interior == 361

    normals = compute_normals(domain_2d, is_boundary)
    @test length(normals) == n_boundary

    basis_phs = RBF.PHS(3; poly_deg=2)
    mon = RBF.MonomialBasis(2, 2)

    k = RBF.autoselect_k(domain_2d, basis_phs)
    adjl = RBF.find_neighbors(domain_2d, 15)

    boundary_conditions = setup_test_boundary_conditions(domain_2d, is_boundary, RBF)
    @test length(boundary_conditions) == n_boundary

    u_values = construct_u_values_hermite(
        domain_2d, is_boundary, boundary_conditions, normals, RBF
    )

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
        laplacian_result = L_op(u_values)

        laplacian_result_interior = laplacian_result[.!is_boundary]
        expected_laplacian = [
            target_laplacian(p[1], p[2]) for p in domain_2d[.!is_boundary]
        ]
        laplacian_error = laplacian_result_interior - expected_laplacian

        max_error = maximum(abs.(laplacian_error))
        rms_error = sqrt(mean(laplacian_error .^ 2))

        @test max_error < 1e-10
        @test rms_error < 1e-11

        # println("  Laplacian max error: ", max_error)
        # println("  Laplacian RMS error: ", rms_error)
    end

    @testset "Test 2: Manufactured Solution (Inverse Problem)" begin
        rhs = construct_rhs_laplacian(
            domain_2d, is_boundary, boundary_conditions, normals, RBF
        )

        laplacian_result = L_op(u_values)
        rhs_error = abs.(rhs - laplacian_result)
        @test maximum(rhs_error) < 1e-10

        solution = L_op.weights \ rhs

        solution_error = solution - u_values
        max_error = maximum(abs.(solution_error))
        rms_error = sqrt(mean(solution_error .^ 2))

        @test max_error < 1e-10  # Machine precision
        @test rms_error < 1e-11  # RMS should be even better

        # println("  Solution max error: ", max_error)
        # println("  Solution RMS error: ", rms_error)
    end
end
