"""
Unit tests for RHS vector building functions.
Tests both standard and Hermite variants of _build_rhs! and related functions.

Focus: Tests the integration of operators with RHS building, not the operators themselves.
Operator correctness is tested in test/operators/. Monomial derivatives are tested in test/basis/monomial.jl.
"""

using Test
using LinearAlgebra
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "RHS Vector Building" begin

    # Basis functions
    basis_phs = PHS(3; poly_deg=1)
    basis_imq = IMQ(1.0)
    basis_gaussian = Gaussian(1.0)
    all_bases = [basis_phs, basis_imq, basis_gaussian]

    # Hermite compatibility: Only PHS has directional∂² for Robin-Robin interactions
    hermite_compatible_bases = [basis_phs]

    # 2D test data (consistent with matrix_entries.jl)
    data_2d = [[0.0, 0.0], [0.5, 0.3], [1.0, 0.0], [0.5, 0.7]]
    eval_point_2d = [0.5, 0.5]
    k_2d = 4

    # Monomial bases for different polynomial degrees
    mon_deg1 = MonomialBasis(2, 1)  # [1, x, y]
    mon_deg2 = MonomialBasis(2, 2)  # [1, x, y, xy, x², y²]
    mon_deg3 = MonomialBasis(2, 3)  # [1, x, y, ..., x³, y³]

    nmon_deg1 = 3
    nmon_deg2 = 6
    nmon_deg3 = 10

    """Test RHS building with identity operator for given polynomial degree."""
    function test_identity_rhs(basis, mon, data, eval_point, k, nmon)
        identity_op_rbf = RBF.Custom(b -> (x1, x2) -> b(x1, x2))
        identity_op_mon = RBF.Custom(m -> (arr, x) -> m(arr, x))

        ℒrbf = identity_op_rbf(basis)
        ℒmon = identity_op_mon(mon)

        n = k + nmon
        b = zeros(Float64, n, 1)

        RBF._build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)

        # Check structure
        @test size(b) == (n, 1)
        @test all(isfinite.(b))

        # RBF part should match basis evaluations
        for i in 1:k
            expected = basis(eval_point, data[i])
            @test b[i, 1] ≈ expected
        end

        # Polynomial part should match monomial evaluations
        poly_vals = zeros(nmon)
        mon(poly_vals, eval_point)
        for i in 1:nmon
            @test b[k + i, 1] ≈ poly_vals[i]
        end
    end

    """Test RHS building with partial derivative operator."""
    function test_partial_rhs(basis, mon, data, eval_point, k, nmon, dim)
        partial_op_rbf = RBF.Custom(b -> RBF.∂(b, dim))
        partial_op_mon = RBF.Custom(m -> RBF.∂(m, dim))

        ℒrbf = partial_op_rbf(basis)
        ℒmon = partial_op_mon(mon)

        n = k + nmon
        b = zeros(Float64, n, 1)

        RBF._build_rhs!(b, ℒrbf, ℒmon, data, eval_point, basis, k)

        @test size(b) == (n, 1)
        @test all(isfinite.(b))

        # RBF part should match derivative evaluations
        for i in 1:k
            expected = RBF.∂(basis, dim)(eval_point, data[i])
            @test b[i, 1] ≈ expected
        end

        # Polynomial part should match monomial derivative
        poly_deriv = zeros(nmon)
        RBF.∂(mon, dim)(poly_deriv, eval_point)
        for i in 1:nmon
            @test b[k + i, 1] ≈ poly_deriv[i]
        end
    end

    @testset "Standard RHS Vector" begin
        @testset "Identity operator - polynomial degree $deg" for (deg, mon, nmon) in [
            (1, mon_deg1, nmon_deg1), (2, mon_deg2, nmon_deg2), (3, mon_deg3, nmon_deg3)
        ]
            for basis in all_bases
                test_identity_rhs(basis, mon, data_2d, eval_point_2d, k_2d, nmon)
            end
        end

        @testset "Partial derivative ∂/∂x - polynomial degree $deg" for (deg, mon, nmon) in
                                                                        [
            (1, mon_deg1, nmon_deg1), (2, mon_deg2, nmon_deg2), (3, mon_deg3, nmon_deg3)
        ]
            for basis in all_bases
                test_partial_rhs(basis, mon, data_2d, eval_point_2d, k_2d, nmon, 1)
            end
        end

        @testset "Partial derivative ∂/∂y - polynomial degree $deg" for (deg, mon, nmon) in
                                                                        [
            (1, mon_deg1, nmon_deg1), (2, mon_deg2, nmon_deg2), (3, mon_deg3, nmon_deg3)
        ]
            for basis in all_bases
                test_partial_rhs(basis, mon, data_2d, eval_point_2d, k_2d, nmon, 2)
            end
        end

        @testset "Second derivative ∂²/∂x² - polynomial degree $deg" for (deg, mon, nmon) in
                                                                         [
            (2, mon_deg2, nmon_deg2),  # Need at least deg 2 for non-zero second derivatives
            (3, mon_deg3, nmon_deg3),
        ]
            second_deriv_op = RBF.Custom(b -> RBF.∂²(b, 1))
            second_deriv_mon = RBF.Custom(m -> RBF.∂²(m, 1))

            for basis in all_bases
                ℒrbf = second_deriv_op(basis)
                ℒmon = second_deriv_mon(mon)

                n = k_2d + nmon
                b = zeros(Float64, n, 1)

                RBF._build_rhs!(b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, k_2d)

                @test size(b) == (n, 1)
                @test all(isfinite.(b))

                # Verify polynomial part matches monomial second derivative
                poly_deriv = zeros(nmon)
                RBF.∂²(mon, 1)(poly_deriv, eval_point_2d)
                for i in 1:nmon
                    @test b[k_2d + i, 1] ≈ poly_deriv[i]
                end
            end
        end

        @testset "Gradient operator (tuple) - polynomial degree $deg" for (
            deg, mon, nmon
        ) in [
            (1, mon_deg1, nmon_deg1), (2, mon_deg2, nmon_deg2), (3, mon_deg3, nmon_deg3)
        ]
            for basis in all_bases
                # Gradient as tuple: (∂/∂x, ∂/∂y)
                ℒrbf = (RBF.∂(basis, 1), RBF.∂(basis, 2))
                ℒmon = (RBF.∂(mon, 1), RBF.∂(mon, 2))

                n = k_2d + nmon
                b = zeros(Float64, n, 2)  # 2 columns for gradient components

                RBF._build_rhs!(b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, k_2d)

                @test size(b) == (n, 2)
                @test all(isfinite.(b))

                # Verify polynomial part for each gradient component
                for dim in 1:2
                    poly_deriv = zeros(nmon)
                    RBF.∂(mon, dim)(poly_deriv, eval_point_2d)
                    for i in 1:nmon
                        @test b[k_2d + i, dim] ≈ poly_deriv[i]
                    end
                end
            end
        end

        @testset "Laplacian operator - polynomial degree $deg" for (deg, mon, nmon) in [
            (2, mon_deg2, nmon_deg2),  # Need at least deg 2 for non-zero Laplacian
            (3, mon_deg3, nmon_deg3),
        ]
            laplacian_op = RBF.Custom(b -> RBF.∇²(b))
            laplacian_mon = RBF.Custom(m -> RBF.∇²(m))

            for basis in all_bases
                ℒrbf = laplacian_op(basis)
                ℒmon = laplacian_mon(mon)

                n = k_2d + nmon
                b = zeros(Float64, n, 1)

                RBF._build_rhs!(b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, k_2d)

                @test size(b) == (n, 1)
                @test all(isfinite.(b))

                # Verify polynomial part
                poly_laplacian = zeros(nmon)
                RBF.∇²(mon)(poly_laplacian, eval_point_2d)
                for i in 1:nmon
                    @test b[k_2d + i, 1] ≈ poly_laplacian[i]
                end
            end
        end
    end

    @testset "Hermite RHS Vector" begin
        # Note: Only PHS has directional∂² for Robin-Robin interactions

        @testset "HermiteStencilData structure validation" begin
            # Test data structure creation and consistency
            is_boundary = [false, true, true, false]
            bcs = [Internal(), Dirichlet(), Neumann(), Internal()]
            normals = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
            hermite_data = RBF.HermiteStencilData(data_2d, is_boundary, bcs, normals)

            @test hermite_data isa RBF.HermiteStencilData{Float64}
            @test length(hermite_data.data) == k_2d
            @test length(hermite_data.is_boundary) == k_2d
            @test length(hermite_data.boundary_conditions) == k_2d
            @test length(hermite_data.normals) == k_2d

            @test hermite_data.is_boundary[2] == true
            @test hermite_data.is_boundary[3] == true
            @test is_dirichlet(hermite_data.boundary_conditions[2])
            @test is_neumann(hermite_data.boundary_conditions[3])
        end
    end
end
