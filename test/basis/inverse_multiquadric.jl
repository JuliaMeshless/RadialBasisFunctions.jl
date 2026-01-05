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

    imq = IMQ(5.0; poly_deg=0)
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
imq = IMQ(2; poly_deg=-1)

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

@testset "Directional First Derivatives" begin
    v1 = SVector(1.0, 0.0)
    v2 = SVector(0.0, 1.0)
    normal = SVector(1.0, 1) ./ sqrt(2)

    # Test D equals dot(v, ∇)
    @test RBF.D(imq, v1)(x₁, x₂) ≈ dot(v1, RBF.∇(imq)(x₁, x₂))
    @test RBF.D(imq, v2)(x₁, x₂) ≈ dot(v2, RBF.∇(imq)(x₁, x₂))
    @test RBF.D(imq, normal)(x₁, x₂) ≈ dot(normal, RBF.∇(imq)(x₁, x₂))

    # Test against ForwardDiff
    expected = FD.gradient(x -> imq(x, x₂), x₁) ⋅ normal
    @test RBF.D(imq, normal)(x₁, x₂) ≈ expected rtol = 1e-5
end
