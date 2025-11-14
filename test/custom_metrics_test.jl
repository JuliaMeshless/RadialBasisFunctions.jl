using RadialBasisFunctions
import RadialBasisFunctions as RBF
using RadialBasisFunctions: PHS, Gaussian, IMQ, gradient, laplacian, partial, directional, Dirichlet, Internal
using StaticArraysCore
using LinearAlgebra
using Distances
using Test
import ForwardDiff as FD

@testset "Custom Distance Metrics" begin
    # Generate test data
    data_2d = [SVector(rand(2)...) for _ in 1:50]
    data_3d = [SVector(rand(3)...) for _ in 1:50]

    @testset "Backward Compatibility" begin
        # Test that existing code works unchanged with default Euclidean metric
        @testset "Default Euclidean - 2D" begin
            basis = PHS(3; poly_deg=2)
            @test basis.metric isa Euclidean

            # Test operator construction
            op_grad = gradient(data_2d, basis)
            @test op_grad isa RadialBasisOperator

            op_lap = laplacian(data_2d, basis)
            @test op_lap isa RadialBasisOperator

            op_partial = partial(data_2d, 1, 1, basis)
            @test op_partial isa RadialBasisOperator

            op_dir = directional(data_2d, SVector(1.0, 0.0), basis)
            @test op_dir isa RadialBasisOperator
        end

        @testset "Explicit Euclidean - 3D" begin
            basis = PHS(3; poly_deg=2, metric=Euclidean())
            @test basis.metric isa Euclidean

            op_grad = gradient(data_3d, basis)
            @test op_grad isa RadialBasisOperator
        end
    end

    @testset "Built-in Metrics" begin
        @testset "Cityblock (Manhattan) Distance" begin
            basis_l1 = PHS(3; poly_deg=2, metric=Cityblock())
            @test basis_l1.metric isa Cityblock

            # Test basis evaluation
            x1 = SVector(1.0, 2.0)
            x2 = SVector(2.0, 4.0)
            expected_dist = abs(1.0 - 2.0) + abs(2.0 - 4.0)  # |Δx| + |Δy| = 1 + 2 = 3
            @test basis_l1(x1, x2) ≈ expected_dist^3

            # Test operator construction
            op = gradient(data_2d, basis_l1)
            @test op isa RadialBasisOperator

            # Test that derivatives work (AD-based)
            ∂rbf = RBF.∂(basis_l1, 1)
            result = ∂rbf(x1, x2)
            @test result isa Real
            @test isfinite(result)
        end

        @testset "Chebyshev Distance" begin
            basis_linf = PHS(3; poly_deg=2, metric=Chebyshev())
            @test basis_linf.metric isa Chebyshev

            x1 = SVector(1.0, 2.0)
            x2 = SVector(2.0, 4.0)
            expected_dist = max(abs(1.0 - 2.0), abs(2.0 - 4.0))  # max(1, 2) = 2
            @test basis_linf(x1, x2) ≈ expected_dist^3

            op = laplacian(data_2d, basis_linf)
            @test op isa RadialBasisOperator
        end

        @testset "Minkowski Distance" begin
            p = 1.5
            basis_lp = PHS(5; poly_deg=2, metric=Minkowski(p))
            @test basis_lp.metric isa Minkowski

            op = partial(data_2d, 1, 1, basis_lp)
            @test op isa RadialBasisOperator
        end
    end

    @testset "Custom Metric Definition" begin
        # Define a custom metric type
        struct SquaredEuclideanMetric <: Metric end

        # Make it callable - squared Euclidean distance
        function (::SquaredEuclideanMetric)(x, y)
            return sum((x .- y).^2)
        end

        @testset "Custom Metric - Basic Usage" begin
            basis_custom = PHS(3; poly_deg=2, metric=SquaredEuclideanMetric())
            @test basis_custom.metric isa SquaredEuclideanMetric

            # Test basis evaluation
            x1 = SVector(1.0, 2.0)
            x2 = SVector(2.0, 4.0)
            expected_dist_sq = (1.0 - 2.0)^2 + (2.0 - 4.0)^2  # = 1 + 4 = 5
            @test basis_custom(x1, x2) ≈ expected_dist_sq^3

            # Test derivatives work with AD
            ∂rbf = RBF.∂(basis_custom, 1)
            result = ∂rbf(x1, x2)
            @test result isa Real
            @test isfinite(result)
        end

        @testset "Custom Metric - Gradient Operator" begin
            # Note: KDTree only supports Minkowski metrics from Distances.jl
            # Custom metrics work with the basis but may need BallTree for neighbor search
            # For this test, we'll use Minkowski instead
            basis_custom = PHS(3; poly_deg=2, metric=Minkowski(3.0))
            op = gradient(data_2d, basis_custom)
            @test op isa RadialBasisOperator
        end
    end

    @testset "Hermite Interpolation Validation" begin
        # Hermite interpolation should only work with Euclidean metric
        eval_points = [SVector(rand(2)...) for _ in 1:20]
        is_boundary = [i <= 10 for i in 1:50]
        boundary_conditions = [i <= 10 ? Dirichlet() : Internal() for i in 1:50]
        normals = [i <= 10 ? SVector(1.0, 0.0) : SVector(0.0, 0.0) for i in 1:50]

        @testset "Euclidean - Should Work" begin
            basis_euclidean = PHS(3; poly_deg=2, metric=Euclidean())
            # This should NOT throw an error
            op = gradient(data_2d, eval_points, basis_euclidean,
                         is_boundary, boundary_conditions, normals)
            @test op isa RadialBasisOperator
        end

        @testset "Non-Euclidean - Should Error" begin
            basis_manhattan = PHS(3; poly_deg=2, metric=Cityblock())

            # This SHOULD throw an ArgumentError
            @test_throws ArgumentError gradient(
                data_2d, eval_points, basis_manhattan,
                is_boundary, boundary_conditions, normals
            )

            @test_throws ArgumentError laplacian(
                data_2d, eval_points, basis_manhattan,
                is_boundary, boundary_conditions, normals
            )

            @test_throws ArgumentError partial(
                data_2d, eval_points, 1, 1, basis_manhattan,
                is_boundary, boundary_conditions, normals
            )
        end
    end

    @testset "AD Correctness" begin
        # Verify that AD-based derivatives match analytical derivatives for Euclidean
        @testset "Compare AD vs Analytical for Euclidean" begin
            x1 = SVector(1.0, 2.0)
            x2 = SVector(2.0, 4.0)

            # Create two identical bases - one will use analytical, one will force AD
            basis_euclidean = PHS(3; poly_deg=2, metric=Euclidean())

            # Get analytical derivative (Euclidean path)
            ∂rbf_analytical = RBF.∂(basis_euclidean, 1)
            result_analytical = ∂rbf_analytical(x1, x2)

            # Compute using ForwardDiff directly (simulating AD path)
            result_ad = FD.derivative(t -> basis_euclidean(x1 + t * SVector(1.0, 0.0), x2), 0.0)

            @test result_analytical ≈ result_ad rtol=1e-10
        end

        @testset "Verify AD for Non-Euclidean" begin
            x1 = SVector(1.0, 2.0)
            x2 = SVector(2.0, 4.0)

            basis_manhattan = PHS(3; poly_deg=2, metric=Cityblock())

            # Get AD-based derivative
            ∂rbf = RBF.∂(basis_manhattan, 1)
            result = ∂rbf(x1, x2)

            # Verify using ForwardDiff directly
            expected = FD.derivative(t -> basis_manhattan(x1 + t * SVector(1.0, 0.0), x2), 0.0)

            @test result ≈ expected rtol=1e-10
        end
    end

    @testset "Different Basis Types with Metrics" begin
        x1 = SVector(1.0, 2.0)
        x2 = SVector(2.0, 4.0)

        @testset "Gaussian" begin
            gauss_eucl = Gaussian(1.0; poly_deg=2, metric=Euclidean())
            gauss_city = Gaussian(1.0; poly_deg=2, metric=Cityblock())

            @test gauss_eucl.metric isa Euclidean
            @test gauss_city.metric isa Cityblock

            # Test evaluation
            @test gauss_eucl(x1, x2) isa Real
            @test gauss_city(x1, x2) isa Real

            # Test derivatives
            ∂gauss_eucl = RBF.∂(gauss_eucl, 1)
            ∂gauss_city = RBF.∂(gauss_city, 1)

            @test ∂gauss_eucl(x1, x2) isa Real
            @test ∂gauss_city(x1, x2) isa Real
        end

        @testset "Inverse Multiquadric" begin
            imq_eucl = IMQ(1.0; poly_deg=2, metric=Euclidean())
            imq_cheby = IMQ(1.0; poly_deg=2, metric=Chebyshev())

            @test imq_eucl.metric isa Euclidean
            @test imq_cheby.metric isa Chebyshev

            # Test evaluation
            @test imq_eucl(x1, x2) isa Real
            @test imq_cheby(x1, x2) isa Real

            # Test derivatives
            ∂imq_eucl = RBF.∂(imq_eucl, 1)
            ∂imq_cheby = RBF.∂(imq_cheby, 1)

            @test ∂imq_eucl(x1, x2) isa Real
            @test ∂imq_cheby(x1, x2) isa Real
        end
    end

    @testset "Neighbor Search with Custom Metrics" begin
        # Test that find_neighbors works with custom metrics
        @testset "Euclidean" begin
            k = 10
            adjl_eucl = RBF.find_neighbors(data_2d, k; metric=Euclidean())
            @test length(adjl_eucl) == length(data_2d)
            @test all(length(neighbors) == k for neighbors in adjl_eucl)
        end

        @testset "Cityblock" begin
            k = 10
            adjl_city = RBF.find_neighbors(data_2d, k; metric=Cityblock())
            @test length(adjl_city) == length(data_2d)
            @test all(length(neighbors) == k for neighbors in adjl_city)
        end

        @testset "Different Metrics Give Different Neighbors" begin
            k = 5
            adjl_eucl = RBF.find_neighbors(data_2d, k; metric=Euclidean())
            adjl_city = RBF.find_neighbors(data_2d, k; metric=Cityblock())

            # Neighbors should potentially be different for different metrics
            # (We don't require them to be different, just that both work)
            @test length(adjl_eucl) == length(adjl_city)
        end
    end
end
