using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra
import ForwardDiff as FD

@testset "Constructors and Printing" begin
    imq = IMQ()
    @test imq isa IMQ
    @test imq.ε == 1
    @test imq.poly_deg == 2

    imq = IMQ(5.0; poly_deg = 0)
    @test imq.ε ≈ 5
    @test imq.poly_deg == 0

    @test_throws ArgumentError IMQ(-1)

    @test repr(imq) == """
        Inverse Multiquadrics, 1/sqrt((r*ε)²+1)
        ├─Shape factor: ε = 5.0
        └─Polynomial augmentation: degree 0"""
end

x₁ = SVector(1.0, 2)
x₂ = SVector(2.0, 4)
imq = IMQ(2; poly_deg = -1)

@testset "Distances" begin
    r = sqrt((x₁[1] - x₂[1])^2 + (x₁[2] - x₂[2])^2)
    @test imq(x₁, x₂) ≈ 1 / sqrt((imq.ε * r)^2 + 1)
end

@testset "Derivatives" begin
    dim = 1
    ∂rbf = RBF.∂(imq, dim)
    ∂²rbf = RBF.∂²(imq, dim)
    ∇rbf = RBF.∇(imq)

    @test ∂rbf(x₁, x₂) ≈ 4 / (21 * sqrt(21))
    @test all(∇rbf(x₁, x₂) .≈ (4 / (21 * sqrt(21)), 8 / (21 * sqrt(21))))
    @test ∂²rbf(x₁, x₂) ≈ -4 / (49 * sqrt(21))
end

@testset "Hessian" begin
    Hrbf = RBF.H(imq)
    H_val = Hrbf(x₁, x₂)

    # Test matrix size
    @test size(H_val) == (2, 2)

    # Test Hessian symmetry
    @test H_val ≈ H_val' rtol = 1.0e-10

    # Test against ForwardDiff Hessian
    H_fd = FD.hessian(x -> imq(x, x₂), x₁)
    @test H_val ≈ H_fd rtol = 1.0e-5
end
