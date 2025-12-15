"""
Unit tests for RHS vector building functions.
Tests both standard and Hermite variants of _build_rhs!

Focus: Tests the integration of operators with RHS building, not the operators themselves.
Operator correctness is tested in test/operators/.
"""

using Test
using LinearAlgebra
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "RHS Vector Building" begin
    # Test setup
    basis = PHS(3; poly_deg=1)
    data_2d = [[0.0, 0.0], [0.5, 0.3], [1.0, 0.0], [0.5, 0.7]]
    eval_point_2d = [0.5, 0.5]
    k = 4
    mon = MonomialBasis(2, 1)
    nmon = 3
    n = k + nmon

    @testset "Standard RHS - Identity Operator" begin
        # Test with identity operator (simplest case)
        identity_op_rbf = RBF.Custom(b -> (x1, x2) -> b(x1, x2))
        identity_op_mon = RBF.Custom(m -> (arr, x) -> m(arr, x))

        ℒrbf = identity_op_rbf(basis)
        ℒmon = identity_op_mon(mon)

        b = zeros(Float64, n)
        RBF._build_rhs!(b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, mon, k)

        # Basic validation
        @test size(b) == (n,)
        @test all(isfinite.(b))

        # RBF part should match basis evaluations
        for i in 1:k
            @test b[i] ≈ basis(eval_point_2d, data_2d[i])
        end

        # Polynomial part should match monomial evaluations
        poly_vals = zeros(nmon)
        mon(poly_vals, eval_point_2d)
        for i in 1:nmon
            @test b[k + i] ≈ poly_vals[i]
        end
    end

    @testset "Standard RHS - Partial Derivative" begin
        # Test with partial derivative operator
        partial_op_rbf = RBF.Custom(b -> RBF.∂(b, 1))
        partial_op_mon = RBF.Custom(m -> RBF.∂(m, 1))

        ℒrbf = partial_op_rbf(basis)
        ℒmon = partial_op_mon(mon)

        b = zeros(Float64, n)
        RBF._build_rhs!(b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, mon, k)

        # Basic validation
        @test size(b) == (n,)
        @test all(isfinite.(b))

        # RBF part should match derivative evaluations
        for i in 1:k
            @test b[i] ≈ RBF.∂(basis, 1)(eval_point_2d, data_2d[i])
        end

        # Polynomial part should match monomial derivative
        poly_deriv = zeros(nmon)
        RBF.∂(mon, 1)(poly_deriv, eval_point_2d)
        for i in 1:nmon
            @test b[k + i] ≈ poly_deriv[i]
        end
    end

    @testset "Standard RHS - Multiple Operators (Gradient)" begin
        # Test with multiple operators (gradient as tuple)
        ℒrbf = (RBF.∂(basis, 1), RBF.∂(basis, 2))
        ℒmon = (RBF.∂(mon, 1), RBF.∂(mon, 2))

        b = zeros(Float64, n, 2)
        RBF._build_rhs!(b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, mon, k)

        # Basic validation
        @test size(b) == (n, 2)
        @test all(isfinite.(b))

        # Verify polynomial part for each gradient component
        for dim in 1:2
            poly_deriv = zeros(nmon)
            RBF.∂(mon, dim)(poly_deriv, eval_point_2d)
            for i in 1:nmon
                @test b[k + i, dim] ≈ poly_deriv[i]
            end
        end
    end

    @testset "Hermite RHS with Dirichlet" begin
        # Test Hermite RHS with Dirichlet boundaries (behaves like standard)
        # Note: Full Hermite testing with Neumann/Robin is in integration tests
        is_boundary = [false, true, false, false]
        bcs = [Internal(), Dirichlet(), Internal(), Internal()]
        normals = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        hermite_data = RBF.HermiteStencilData(data_2d, is_boundary, bcs, normals)

        # Use identity operator for simplicity
        identity_op_rbf = RBF.Custom(b -> (x1, x2) -> b(x1, x2))
        identity_op_mon = RBF.Custom(m -> (arr, x) -> m(arr, x))

        ℒrbf = identity_op_rbf(basis)
        ℒmon = identity_op_mon(mon)

        b = zeros(Float64, n)

        # Note: eval_point must be in the stencil for Hermite
        # Use first point (interior) as eval point
        eval_pt = data_2d[1]
        RBF._build_rhs!(b, ℒrbf, ℒmon, hermite_data, eval_pt, basis, mon, k)

        # Basic validation
        @test size(b) == (n,)
        @test all(isfinite.(b))

        # With Dirichlet boundaries, should match standard RHS
        b_standard = zeros(Float64, n)
        RBF._build_rhs!(b_standard, ℒrbf, ℒmon, data_2d, eval_pt, basis, mon, k)
        @test b ≈ b_standard
    end

    @testset "HermiteStencilData Structure" begin
        # Test that HermiteStencilData is properly constructed
        is_boundary = [false, true, true, false]
        bcs = [Internal(), Dirichlet(), Neumann(), Internal()]
        normals = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

        hermite_data = RBF.HermiteStencilData(data_2d, is_boundary, bcs, normals)

        @test hermite_data isa RBF.HermiteStencilData{Float64}
        @test length(hermite_data.data) == k
        @test length(hermite_data.is_boundary) == k
        @test length(hermite_data.boundary_conditions) == k
        @test length(hermite_data.normals) == k

        @test hermite_data.is_boundary[2] == true
        @test hermite_data.is_boundary[3] == true
        @test RBF.is_dirichlet(hermite_data.boundary_conditions[2])
        @test RBF.is_neumann(hermite_data.boundary_conditions[3])
    end
end
