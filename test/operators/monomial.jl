using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra
using Test
using ForwardDiff

@testset "MonomialBasis - Basic functionality" begin
    @test_throws ArgumentError RadialBasisFunctions.ℒMonomialBasis(1, -1, identity)
    m = RadialBasisFunctions.ℒMonomialBasis(1, 0, identity)
    @test repr(m) == "ℒMonomialBasis of degree 0 in 1 dimensions"
end

@testset "MonomialBasis - Normal derivatives" begin
    @testset "1D MonomialBasis" begin
        # Create a 1D monomial basis of degree 3
        mb = MonomialBasis(1, 3)
        x = SVector(2.0)

        # Define a "normal" in 1D (just direction)
        normal = SVector(1.0)

        # Get normal derivative operator
        normal_op = RBF.∂_normal(mb, normal)

        # Evaluate the normal derivative
        result = zeros(binomial(1 + 3, 1))
        normal_op.f(result, x)

        # Expected results: [0, 1, 2*x, 3*x^2] * normal[1]
        expected = [0.0, 1.0, 4.0, 12.0]

        @test result ≈ expected
    end

    @testset "2D MonomialBasis" begin
        # Create a 2D monomial basis of degree 2
        mb = MonomialBasis(2, 2)
        x = SVector(1.0, 2.0)

        # Define a normal vector
        normal = SVector(1.0, 1.0) ./ sqrt(2)  # Normalized for better numerical stability

        # Get normal derivative operator
        normal_op = RBF.∂_normal(mb, normal)

        # Evaluate the normal derivative
        result = zeros(binomial(2 + 2, 2))
        normal_op.f(result, x)

        # Calculate expected result manually
        # For a polynomial basis [1, x, y, x*y, x^2, y^2]
        # The partial derivatives are:
        # ∂/∂x = [0, 1, 0, y, 2*x, 0]
        # ∂/∂y = [0, 0, 1, x, 0, 2*y]
        # So the normal derivative is normal[1]*∂/∂x + normal[2]*∂/∂y
        n1 = normal[1]
        n2 = normal[2]
        expected = [0.0, n1, n2, n1 * x[2] + n2 * x[1], 2 * n1 * x[1], 2 * n2 * x[2]]

        @test result ≈ expected

        # Verify against ForwardDiff
        mb_eval(x) = [1, x[1], x[2], x[1] * x[2], x[1]^2, x[2]^2]  # Explicit polynomial basis
        grad_at_x = ForwardDiff.jacobian(mb_eval, x)
        fd_result = grad_at_x * normal

        @test result ≈ fd_result
    end

    @testset "3D MonomialBasis" begin
        # Create a 3D monomial basis of degree 1
        mb = MonomialBasis(3, 1)
        x = SVector(1.0, 2.0, 3.0)

        # Define a normal vector
        normal = SVector(1.0, 1.0, 1.0) ./ sqrt(3)  # Normalized

        # Get normal derivative operator
        normal_op = RBF.∂_normal(mb, normal)

        # Evaluate the normal derivative
        result = zeros(binomial(3 + 1, 3))
        normal_op.f(result, x)

        # Calculate expected result manually
        # For a polynomial basis [1, x, y, z]
        # The partial derivatives are:
        # ∂/∂x = [0, 1, 0, 0]
        # ∂/∂y = [0, 0, 1, 0]
        # ∂/∂z = [0, 0, 0, 1]
        # So the normal derivative is normal[1]*∂/∂x + normal[2]*∂/∂y + normal[3]*∂/∂z
        n1 = normal[1]
        n2 = normal[2]
        n3 = normal[3]
        expected = [0.0, n1, n2, n3]

        @test result ≈ expected

        # Verify against ForwardDiff
        mb_eval(x) = [1, x[1], x[2], x[3]]  # Explicit polynomial basis
        grad_at_x = ForwardDiff.jacobian(mb_eval, x)
        fd_result = grad_at_x * normal

        @test result ≈ fd_result
    end
end

@testset "MonomialBasis - Hermite derivatives" begin
    @testset "Regular vs. Normal derivatives" begin
        # Create a 2D monomial basis of degree 2
        mb = MonomialBasis(2, 2)
        x = SVector(1.0, 2.0)

        # Define dimension and normal vector
        dim = 1
        normal = SVector(1.0, 1.0) ./ sqrt(2)

        # Get Hermite derivative operator
        hermite_op = RBF.∂_Hermite(mb, dim)

        # Test regular case (no normal)
        regular_result = hermite_op(x)

        # This should be equivalent to regular partial derivative
        partial_op = RBF.∂(mb, dim)
        expected_regular = zeros(binomial(2 + 2, 2))
        partial_op.f(expected_regular, x)

        @test regular_result ≈ expected_regular

        # Test Neumann boundary case (with normal)
        neumann_result = hermite_op(x, normal)

        # This should be equivalent to normal derivative
        normal_op = RBF.∂_normal(mb, normal)
        expected_neumann = zeros(binomial(2 + 2, 2))
        normal_op.f(expected_neumann, x)

        @test neumann_result ≈ expected_neumann
    end

    @testset "Directional derivatives comparison" begin
        # Create a 3D monomial basis of degree 2
        mb = MonomialBasis(3, 2)
        x = SVector(1.0, 2.0, 3.0)

        # Try different coordinate directions
        for dim in 1:3
            # Get Hermite derivative in coordinate direction
            hermite_op = RBF.∂_Hermite(mb, dim)

            # Create a normal vector in that direction
            normal = zeros(3)
            normal[dim] = 1.0
            normal = SVector{3}(normal)

            # Calculate both ways
            reg_result = hermite_op(x)
            neumann_result = hermite_op(x, normal)

            # They should be the same
            @test reg_result ≈ neumann_result
        end
    end
end
