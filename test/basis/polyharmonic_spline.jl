using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra
import ForwardDiff as FD

@testset "Constructors and Printing" begin
    phs = PHS()
    @test phs isa PHS3
    @test phs.poly_deg == 2

    phs = PHS(5; poly_deg = 0)
    @test phs.poly_deg == 0

    @test_throws ArgumentError PHS(2; poly_deg = -1)
    @test_throws ArgumentError PHS(3; poly_deg = -2)

    @test repr(phs) == """
        Polyharmonic spline (r⁵)
        └─Polynomial augmentation: degree 0"""
end

@testset "PHS, n=1" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(1; poly_deg = -1)
    @test phs isa PHS1
    @test phs.poly_deg == -1
    phs = PHS1(; poly_deg = -1)
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

    @testset "Hessian" begin
        Hrbf = RBF.H(phs)
        H_val = Hrbf(x₁, x₂)

        # Test matrix size
        @test size(H_val) == (2, 2)

        # Test Hessian symmetry
        @test H_val ≈ H_val' rtol = 1.0e-10

        # Test against ForwardDiff Hessian
        H_fd = FD.hessian(x -> phs(x, x₂), x₁)
        @test H_val ≈ H_fd rtol = 1.0e-5
    end
end

@testset "PHS, n=3" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(3; poly_deg = -1)
    @test phs isa PHS3
    @test phs.poly_deg == -1
    phs = PHS3(; poly_deg = -1)
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

    @testset "Hessian" begin
        Hrbf = RBF.H(phs)
        H_val = Hrbf(x₁, x₂)

        # Test matrix size
        @test size(H_val) == (2, 2)

        # Test Hessian symmetry
        @test H_val ≈ H_val' rtol = 1.0e-10

        # Test against ForwardDiff Hessian
        H_fd = FD.hessian(x -> phs(x, x₂), x₁)
        @test H_val ≈ H_fd rtol = 1.0e-5
    end
end

@testset "PHS, n=5" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(5; poly_deg = -1)
    @test phs isa PHS5
    @test phs.poly_deg == -1
    phs = PHS5(; poly_deg = -1)
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

    @testset "Hessian" begin
        Hrbf = RBF.H(phs)
        H_val = Hrbf(x₁, x₂)

        # Test matrix size
        @test size(H_val) == (2, 2)

        # Test Hessian symmetry
        @test H_val ≈ H_val' rtol = 1.0e-10

        # Test against ForwardDiff Hessian
        H_fd = FD.hessian(x -> phs(x, x₂), x₁)
        @test H_val ≈ H_fd rtol = 1.0e-5
    end
end

@testset "Laplacian dimension dependence" begin
    # ∇² must equal the trace of the Hessian in every dimension — the closed-form constants
    # are dimension-dependent (Δrᵏ = k(k+d−2)·r^(k−2)), so 2D and 3D must both be exercised.
    for n in (1, 3, 5, 7)
        phs = PHS(n; poly_deg = -1)
        ∇²rbf = RBF.∇²(phs)
        for (x₁, x₂) in (
                (SVector(1.0, 2), SVector(2.0, 4)),          # 2D
                (SVector(1.0, 2, 3), SVector(2.0, 4, 1)),    # 3D
            )
            @test ∇²rbf(x₁, x₂) ≈ tr(FD.hessian(x -> phs(x, x₂), x₁)) rtol = 1.0e-8
            @test ∇²rbf(x₁, x₂) ≈ sum(RBF.∂²(phs, d)(x₁, x₂) for d in eachindex(x₁))
        end
    end
end

@testset "PHS, n=7" begin
    x₁ = SVector(1.0, 2)
    x₂ = SVector(2.0, 4)
    phs = PHS(7; poly_deg = -1)
    @test phs isa PHS7
    @test phs.poly_deg == -1
    phs = PHS7(; poly_deg = -1)
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

    @testset "Hessian" begin
        Hrbf = RBF.H(phs)
        H_val = Hrbf(x₁, x₂)

        # Test matrix size
        @test size(H_val) == (2, 2)

        # Test Hessian symmetry
        @test H_val ≈ H_val' rtol = 1.0e-10

        # Test against ForwardDiff Hessian
        H_fd = FD.hessian(x -> phs(x, x₂), x₁)
        @test H_val ≈ H_fd rtol = 1.0e-5
    end
end
