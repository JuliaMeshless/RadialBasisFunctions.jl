using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra
import ForwardDiff as FD

@testset "Constructors and Printing" begin
    phs = PHS()
    @test phs isa PHS3
    @test phs.poly_deg == 2

    phs = PHS(5; poly_deg=0)
    @test phs.poly_deg == 0

    @test_throws ArgumentError PHS(2; poly_deg=-1)
    @test_throws ArgumentError PHS(3; poly_deg=-2)

    @test repr(phs) == """
    Polyharmonic spline (r⁵)
    └─Polynomial augmentation: degree 0"""
end

@testset "PHS, n=1" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(1; poly_deg=-1)
    @test phs isa PHS1
    @test phs.poly_deg == -1
    phs = PHS1(; poly_deg=-1)
    @test phs isa PHS1
    @test phs.poly_deg == -1

    @testset "Distances" begin
        @test phs(x₁, x₂) ≈ sqrt((x₁[1] - x₂[1])^2 + (x₁[2] - x₂[2])^2)^1
    end

    @testset "Derivatives" begin
        dim = 1
        ∂rbf = RBF.∂(phs, dim)
        ∂²rbf = RBF.∂²(phs, dim)
        ∇rbf = RBF.∇(phs)

        @test ∂rbf(x₁, x₂) ≈ -1 / sqrt(5)
        @test all(∇rbf(x₁, x₂) .≈ (-1 / sqrt(5), -2 / sqrt(5)))
        @test ∂²rbf(x₁, x₂) ≈ 4 / (5 * sqrt(5))
    end

    @testset "Hermite" begin
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalize for better numerical stability
        dim = 1

        # First derivative - functor handles both 2-arg and 3-arg calls
        ∂rbf = RBF.∂(phs, dim)
        @test ∂rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂rbf(x₁, x_2), x₂) ⋅ normal)

        # Gradient
        ∇rbf = RBF.∇(phs)
        @test all(
            ∇rbf(x₁, x₂, normal) .≈ [
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[1], x₂) ⋅ normal),
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[2], x₂) ⋅ normal),
            ],
        )

        # Second derivative
        ∂²rbf = RBF.∂²(phs, dim)
        @test ∂²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂²rbf(x₁, x_2), x₂) ⋅ normal)

        # Laplacian
        ∇²rbf = RBF.∇²(phs)
        @test ∇²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∇²rbf(x₁, x_2), x₂) ⋅ normal)
    end

    @testset "Directional First Derivatives" begin
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Test D equals dot(v, ∇)
        @test RBF.D(phs, v1)(x₁, x₂) ≈ dot(v1, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, v2)(x₁, x₂) ≈ dot(v2, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, normal)(x₁, x₂) ≈ dot(normal, RBF.∇(phs)(x₁, x₂))

        # Test against ForwardDiff
        expected = FD.gradient(x -> phs(x, x₂), x₁) ⋅ normal
        @test RBF.D(phs, normal)(x₁, x₂) ≈ expected rtol = 1e-5
    end

    @testset "Directional Second Derivatives" begin
        # Define two direction vectors
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Same direction test (both normal)
        dir_deriv = RBF.D²(phs, normal, normal)

        # Calculate expected value manually with ForwardDiff
        first_normal_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ normal
        second_normal = FD.gradient(y -> first_normal_deriv(y), x₂) ⋅ normal

        @test dir_deriv(x₁, x₂) ≈ second_normal rtol = 1e-5

        # Test with orthogonal directions
        dir_deriv_xy = RBF.D²(phs, v1, v2)

        # Calculate mixed partial derivative with ForwardDiff
        first_x_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ v1
        second_mixed = FD.gradient(y -> first_x_deriv(y), x₂) ⋅ v2

        @test dir_deriv_xy(x₁, x₂) ≈ second_mixed rtol = 1e-5
    end
end

@testset "PHS, n=3" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(3; poly_deg=-1)
    @test phs isa PHS3
    @test phs.poly_deg == -1
    phs = PHS3(; poly_deg=-1)
    @test phs isa PHS3
    @test phs.poly_deg == -1

    @testset "Distances" begin
        @test phs(x₁, x₂) ≈ sqrt((x₁[1] - x₂[1])^2 + (x₁[2] - x₂[2])^2)^3
    end

    @testset "Derivatives" begin
        dim = 1
        ∂rbf = RBF.∂(phs, dim)
        ∂²rbf = RBF.∂²(phs, dim)
        ∇rbf = RBF.∇(phs)

        @test ∂rbf(x₁, x₂) ≈ -3 * sqrt(5)
        @test all(∇rbf(x₁, x₂) .≈ (-3 * sqrt(5), -6 * sqrt(5)))
        @test ∂²rbf(x₁, x₂) ≈ 18 / sqrt(5)
    end

    @testset "Hermite" begin
        normal = SVector(1.0, 1) ./ sqrt(2)
        dim = 1

        # First derivative
        ∂rbf = RBF.∂(phs, dim)
        @test ∂rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂rbf(x₁, x_2), x₂) ⋅ normal)

        # Gradient
        ∇rbf = RBF.∇(phs)
        @test all(
            ∇rbf(x₁, x₂, normal) .≈ [
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[1], x₂) ⋅ normal),
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[2], x₂) ⋅ normal),
            ],
        )

        # Second derivative
        ∂²rbf = RBF.∂²(phs, dim)
        @test ∂²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂²rbf(x₁, x_2), x₂) ⋅ normal)

        # Laplacian
        ∇²rbf = RBF.∇²(phs)
        @test ∇²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∇²rbf(x₁, x_2), x₂) ⋅ normal)
    end

    @testset "Directional First Derivatives" begin
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Test D equals dot(v, ∇)
        @test RBF.D(phs, v1)(x₁, x₂) ≈ dot(v1, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, v2)(x₁, x₂) ≈ dot(v2, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, normal)(x₁, x₂) ≈ dot(normal, RBF.∇(phs)(x₁, x₂))

        # Test against ForwardDiff
        expected = FD.gradient(x -> phs(x, x₂), x₁) ⋅ normal
        @test RBF.D(phs, normal)(x₁, x₂) ≈ expected rtol = 1e-5
    end

    @testset "Directional Second Derivatives" begin
        # Define two direction vectors
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Same direction test (both normal)
        dir_deriv = RBF.D²(phs, normal, normal)

        # Calculate expected value manually with ForwardDiff
        first_normal_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ normal
        second_normal = FD.gradient(y -> first_normal_deriv(y), x₂) ⋅ normal

        @test dir_deriv(x₁, x₂) ≈ second_normal rtol = 1e-5

        # Test with orthogonal directions
        dir_deriv_xy = RBF.D²(phs, v1, v2)

        # Calculate mixed partial derivative with ForwardDiff
        first_x_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ v1
        second_mixed = FD.gradient(y -> first_x_deriv(y), x₂) ⋅ v2

        @test dir_deriv_xy(x₁, x₂) ≈ second_mixed rtol = 1e-5
    end
end

@testset "PHS, n=5" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(5; poly_deg=-1)
    @test phs isa PHS5
    @test phs.poly_deg == -1
    phs = PHS5(; poly_deg=-1)
    @test phs isa PHS5
    @test phs.poly_deg == -1

    @testset "Distances" begin
        @test phs(x₁, x₂) ≈ sqrt((x₁[1] - x₂[1])^2 + (x₁[2] - x₂[2])^2)^5
    end

    @testset "Derivatives" begin
        dim = 1
        ∂rbf = RBF.∂(phs, dim)
        ∂²rbf = RBF.∂²(phs, dim)
        ∇rbf = RBF.∇(phs)

        @test ∂rbf(x₁, x₂) ≈ -25 * sqrt(5)
        @test all(∇rbf(x₁, x₂) .≈ (-25 * sqrt(5), -50 * sqrt(5)))
        @test ∂²rbf(x₁, x₂) ≈ 40 * sqrt(5)
    end

    @testset "Hermite" begin
        normal = SVector(1.0, 1) ./ sqrt(2)
        dim = 1

        # First derivative
        ∂rbf = RBF.∂(phs, dim)
        @test ∂rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂rbf(x₁, x_2), x₂) ⋅ normal)

        # Gradient
        ∇rbf = RBF.∇(phs)
        @test all(
            ∇rbf(x₁, x₂, normal) .≈ [
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[1], x₂) ⋅ normal),
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[2], x₂) ⋅ normal),
            ],
        )

        # Second derivative
        ∂²rbf = RBF.∂²(phs, dim)
        @test ∂²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂²rbf(x₁, x_2), x₂) ⋅ normal)

        # Laplacian
        ∇²rbf = RBF.∇²(phs)
        @test ∇²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∇²rbf(x₁, x_2), x₂) ⋅ normal)
    end

    @testset "Directional First Derivatives" begin
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Test D equals dot(v, ∇)
        @test RBF.D(phs, v1)(x₁, x₂) ≈ dot(v1, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, v2)(x₁, x₂) ≈ dot(v2, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, normal)(x₁, x₂) ≈ dot(normal, RBF.∇(phs)(x₁, x₂))

        # Test against ForwardDiff
        expected = FD.gradient(x -> phs(x, x₂), x₁) ⋅ normal
        @test RBF.D(phs, normal)(x₁, x₂) ≈ expected rtol = 1e-5
    end

    @testset "Directional Second Derivatives" begin
        # Define two direction vectors
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Same direction test (both normal)
        dir_deriv = RBF.D²(phs, normal, normal)

        # Calculate expected value manually with ForwardDiff
        first_normal_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ normal
        second_normal = FD.gradient(y -> first_normal_deriv(y), x₂) ⋅ normal

        @test dir_deriv(x₁, x₂) ≈ second_normal rtol = 1e-5

        # Test with orthogonal directions
        dir_deriv_xy = RBF.D²(phs, v1, v2)

        # Calculate mixed partial derivative with ForwardDiff
        first_x_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ v1
        second_mixed = FD.gradient(y -> first_x_deriv(y), x₂) ⋅ v2

        @test dir_deriv_xy(x₁, x₂) ≈ second_mixed rtol = 1e-5
    end
end

@testset "PHS, n=7" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(7; poly_deg=-1)
    @test phs isa PHS7
    @test phs.poly_deg == -1
    phs = PHS7(; poly_deg=-1)
    @test phs isa PHS7
    @test phs.poly_deg == -1
    @testset "Distances" begin
        @test phs(x₁, x₂) ≈ sqrt((x₁[1] - x₂[1])^2 + (x₁[2] - x₂[2])^2)^7
    end

    @testset "Derivatives" begin
        dim = 1
        ∂rbf = RBF.∂(phs, dim)
        ∂²rbf = RBF.∂²(phs, dim)
        ∇rbf = RBF.∇(phs)

        @test ∂rbf(x₁, x₂) ≈ -175 * sqrt(5)
        @test all(∇rbf(x₁, x₂) .≈ (-175 * sqrt(5), -350 * sqrt(5)))
        @test ∂²rbf(x₁, x₂) ≈ 350 * sqrt(5)
    end

    @testset "Hermite" begin
        normal = SVector(1.0, 1) ./ sqrt(2)
        dim = 1

        # First derivative
        ∂rbf = RBF.∂(phs, dim)
        @test ∂rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂rbf(x₁, x_2), x₂) ⋅ normal)

        # Gradient
        ∇rbf = RBF.∇(phs)
        @test all(
            ∇rbf(x₁, x₂, normal) .≈ [
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[1], x₂) ⋅ normal),
                (FD.gradient(x_2 -> ∇rbf(x₁, x_2)[2], x₂) ⋅ normal),
            ],
        )

        # Second derivative
        ∂²rbf = RBF.∂²(phs, dim)
        @test ∂²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∂²rbf(x₁, x_2), x₂) ⋅ normal)

        # Laplacian
        ∇²rbf = RBF.∇²(phs)
        @test ∇²rbf(x₁, x₂, normal) ≈ (FD.gradient(x_2 -> ∇²rbf(x₁, x_2), x₂) ⋅ normal)
    end

    @testset "Directional First Derivatives" begin
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Test D equals dot(v, ∇)
        @test RBF.D(phs, v1)(x₁, x₂) ≈ dot(v1, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, v2)(x₁, x₂) ≈ dot(v2, RBF.∇(phs)(x₁, x₂))
        @test RBF.D(phs, normal)(x₁, x₂) ≈ dot(normal, RBF.∇(phs)(x₁, x₂))

        # Test against ForwardDiff
        expected = FD.gradient(x -> phs(x, x₂), x₁) ⋅ normal
        @test RBF.D(phs, normal)(x₁, x₂) ≈ expected rtol = 1e-5
    end

    @testset "Directional Second Derivatives" begin
        # Define two direction vectors
        v1 = SVector(1.0, 0.0)  # x direction
        v2 = SVector(0.0, 1.0)  # y direction
        normal = SVector(1.0, 1) ./ sqrt(2)  # Normalized diagonal direction

        # Same direction test (both normal)
        dir_deriv = RBF.D²(phs, normal, normal)

        # Calculate expected value manually with ForwardDiff
        first_normal_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ normal
        second_normal = FD.gradient(y -> first_normal_deriv(y), x₂) ⋅ normal

        @test dir_deriv(x₁, x₂) ≈ second_normal rtol = 1e-5

        # Test with orthogonal directions
        dir_deriv_xy = RBF.D²(phs, v1, v2)

        # Calculate mixed partial derivative with ForwardDiff
        first_x_deriv(y) = FD.gradient(x -> phs(x, y), x₁) ⋅ v1
        second_mixed = FD.gradient(y -> first_x_deriv(y), x₂) ⋅ v2

        @test dir_deriv_xy(x₁, x₂) ≈ second_mixed rtol = 1e-5
    end
end
