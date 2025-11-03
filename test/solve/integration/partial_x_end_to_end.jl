using Test
using StaticArraysCore
using LinearAlgebra
using Statistics
using RadialBasisFunctions
import RadialBasisFunctions as RBF

function construct_rhs_partial_x(domain_2d, is_boundary, boundary_conditions, normals, RBF)
    N = length(domain_2d)
    rhs = zeros(N)

    # Set ∂/∂x values for all points initially
    for i in eachindex(domain_2d)
        rhs[i] = 2.0 * domain_2d[i][1]  # ∂u/∂x = 2x
    end

    # Apply boundary conditions
    bnd_counter = 0
    for i in eachindex(domain_2d)
        if is_boundary[i]
            bnd_counter += 1
            bc = boundary_conditions[bnd_counter]
            if RBF.is_dirichlet(bc)
                # Dirichlet: RHS is the function value
                rhs[i] = target_function(domain_2d[i][1], domain_2d[i][2])
            elseif RBF.is_neumann(bc) || RBF.is_robin(bc)
                # Neumann/Robin: RHS is α*u + β*∂ₙu (matching what we do in _build_rhs!)
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

@testset "Partial ∂/∂x End-to-End with Hermite" begin

    # Create 2D domain
    domain_2d = create_2d_unit_square_domain(0.05; randomize=true)
    @test length(domain_2d) == 441

    # Identify boundary points
    is_boundary = identify_boundary_points(domain_2d)
    n_boundary = count(is_boundary)
    n_interior = length(domain_2d) - n_boundary
    @test n_boundary == 80
    @test n_interior == 361

    # Basis configuration
    basis_phs = RBF.PHS(3; poly_deg=2)
    mon = RBF.MonomialBasis(2, 2)

    k = RBF.autoselect_k(domain_2d, basis_phs)
    adjl = RBF.find_neighbors(domain_2d, k)
    boundary_conditions = setup_test_boundary_conditions(domain_2d, is_boundary, RBF)
    normals = compute_normals(domain_2d, is_boundary)

    # Construct u_values for Hermite interpolation
    u_values = construct_u_values_hermite(
        domain_2d, is_boundary, boundary_conditions, normals, RBF
    )

    # Create ∂/∂x operator (order=1, dim=1)
    partial_x_op = RBF.Partial(1, 1)

    Dx_op = RBF.RadialBasisOperator(
        partial_x_op,
        domain_2d,
        domain_2d,
        basis_phs,
        is_boundary,
        boundary_conditions,
        normals;
        k=k,
        adjl=adjl,
    )

    exact_solution = [target_function(p[1], p[2]) for p in domain_2d]

    @testset "Test 1: Weights Calculation (Forward Problem)" begin
        # Apply operator to u_values
        partial_x_result = Dx_op(u_values)

        # At interior points, should get ∂u/∂x = 2x
        partial_x_interior = partial_x_result[.!is_boundary]

        expected_partial_x = Float64[]
        for i in eachindex(domain_2d)
            if !is_boundary[i]
                push!(expected_partial_x, 2.0 * domain_2d[i][1])
            end
        end

        errors = abs.(partial_x_interior - expected_partial_x)
        max_error = maximum(errors)
        rms_error = sqrt(mean(errors .^ 2))

        @test max_error < 1e-10  # Machine precision
        @test rms_error < 1e-11  # RMS should be even better
    end

    @testset "Test 2: RHS Consistency Verification" begin
        # Verify that RHS construction matches operator output
        rhs = construct_rhs_partial_x(
            domain_2d, is_boundary, boundary_conditions, normals, RBF
        )
        partial_x_result = Dx_op(u_values)

        rhs_error = abs.(rhs - partial_x_result)
        max_rhs_error = maximum(rhs_error)

        @test max_rhs_error < 1e-11
    end

    @testset "Test 3: Solution of PDE (Inverse Problem)" begin
        # Test that we can accurately recover u_values from ∂u/∂x
        rhs = construct_rhs_partial_x(
            domain_2d, is_boundary, boundary_conditions, normals, RBF
        )

        cond_num = cond(Matrix(Dx_op.weights))
        # println("Condition number of weights: ", cond_num)

        forward_result = Dx_op.weights * u_values
        forward_error = maximum(abs.(forward_result - rhs))
        # println("Max error in weights * u_values vs rhs: ", forward_error)

        solution = Dx_op.weights \ rhs

        # Check residuals
        residual = Dx_op.weights * solution - rhs
        # println("Max residual: ", maximum(abs.(residual)))
        solution_error = solution - exact_solution #u_values
        max_error = maximum(abs.(solution_error))
        rms_error = sqrt(mean(solution_error .^ 2))

        println("Max error: ", max_error)
        println("RMS error: ", rms_error)

        @test max_error < 1e-9  # Machine precision
        @test rms_error < 1e-10  # RMS should be even better
    end
end
