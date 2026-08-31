"""
Unit tests for collocation matrix assembly functions.
Tests _build_collocation_matrix!
"""

using Test
using LinearAlgebra
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Collocation Matrix Building" begin
    # Test setup
    basis = PHS(3; poly_deg = 1)
    data_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    k = 4
    mon = MonomialBasis(2, 1)
    nmon = 3
    n = k + nmon

    @testset "Standard Collocation Matrix" begin
        # Build standard collocation matrix for interior points
        A = Symmetric(zeros(Float64, n, n), :U)
        RBF._build_collocation_matrix!(A, data_2d, basis, mon, k)
        AA = parent(A)

        # Basic validation
        @test issymmetric(A)
        @test all(isfinite.(AA))
        @test size(A) == (n, n)

        # Check that only upper triangular part was filled
        for i in 1:n, j in 1:(i - 1)
            @test AA[i, j] == 0.0
        end

        # Spot check: RBF entry
        @test AA[1, 2] ≈ basis(data_2d[1], data_2d[2])

        # Spot check: polynomial entry for point [0,0]
        @test AA[1, k + 1] ≈ 1.0  # Constant term
        @test AA[1, k + 2] ≈ 0.0  # x term
        @test AA[1, k + 3] ≈ 0.0  # y term
    end
end
