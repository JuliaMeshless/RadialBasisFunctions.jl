using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra

x = [SVector(1.0, 2.0), SVector(2.0, 1.0), SVector(1.5, 0.0)]

rb = PHS(3; poly_deg=1)
mb = MonomialBasis(2, 1)
L(x) = RBF.∂(x, 1, 1)
Lrb = L(rb)
Lmb = L(mb)

k = length(x)
n = k + 3

@testset "Coefficient Matrix" begin
    A = Symmetric(zeros(n, n))
    RBF._build_collocation_matrix!(A, x, rb, mb, k)
    @testset "RBFs" begin
        @test A[1, 2] ≈ (sqrt(sum((x[1] .- x[2]) .^ 2)))^3
        @test A[1, 3] ≈ (sqrt(sum((x[1] .- x[3]) .^ 2)))^3
        @test A[2, 3] ≈ (sqrt(sum((x[2] .- x[3]) .^ 2)))^3
    end
    @testset "Monomials" begin
        @test all(isapprox.(A[1, 4:6], [1.0, x[1][1], x[1][2]]))
        @test all(isapprox.(A[2, 4:6], [1.0, x[2][1], x[2][2]]))
        @test all(isapprox.(A[3, 4:6], [1.0, x[3][1], x[3][2]]))
    end
end

@testset "Right-hand side" begin
    b = zeros(n)
    eval_point = SVector(0.0, 0.0)
    RBF._build_rhs!(b, Lrb, Lmb, x, eval_point, rb, k)
    @testset "RBFs" begin
        @test b[1] ≈ Lrb(eval_point, x[1])
        @test b[2] ≈ Lrb(eval_point, x[2])
        @test b[3] ≈ Lrb(eval_point, x[3])
    end
    @testset "Monomials" begin
        bb = zeros(3)
        Lmb(bb, x[1])
        @test all(b[4:6] .≈ bb)
    end
end

@testset "Calculate Matrix Entry" begin
    A_test = zeros(3, 3)

    RBF._calculate_matrix_entry!(A_test, 2, 3, x, rb)

    @test A_test[2, 3] ≈ rb(x[2], x[3])
    @test A_test[2, 3] ≈ (sqrt(sum((x[2] .- x[3]) .^ 2)))^3

    @test all(A_test[1, :] .== 0)
    @test A_test[2, 1] == 0
    @test A_test[2, 2] == 0
    @test all(A_test[3, :] .== 0)
end
