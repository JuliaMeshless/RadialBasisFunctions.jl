"""
Unit tests for collocation matrix assembly functions.
Tests both standard and Hermite variants of _build_collocation_matrix!
"""

using Test
using LinearAlgebra
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Collocation Matrix Building" begin
    # Test setup
    basis = PHS(3; poly_deg=1)
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

    @testset "Hermite Collocation Matrix" begin
        # Test with mixed boundary conditions (Neumann + Robin)
        is_boundary = [false, true, false, true]
        bcs = [Internal(), Neumann(), Internal(), Robin(0.5, 1.0)]
        normals = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]

        hermite_data = RBF.HermiteStencilData(data_2d, is_boundary, bcs, normals)

        # Build Hermite matrix
        A_hermite = Symmetric(zeros(Float64, n, n), :U)
        RBF._build_collocation_matrix!(A_hermite, hermite_data, basis, mon, k)

        # Basic validation
        @test issymmetric(A_hermite)
        @test all(isfinite.(parent(A_hermite)))
        @test size(A_hermite) == (n, n)

        # Should differ from standard due to boundary conditions
        A_standard = Symmetric(zeros(Float64, n, n), :U)
        RBF._build_collocation_matrix!(A_standard, data_2d, basis, mon, k)
        @test parent(A_hermite) != parent(A_standard)
    end

    @testset "Hermite with All Interior" begin
        # Edge case: Hermite data but no actual boundaries should match standard
        is_boundary = [false, false, false, false]
        bcs = [Internal(), Internal(), Internal(), Internal()]
        normals = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        hermite_data = RBF.HermiteStencilData(data_2d, is_boundary, bcs, normals)

        A_hermite = Symmetric(zeros(Float64, n, n), :U)
        A_standard = Symmetric(zeros(Float64, n, n), :U)

        RBF._build_collocation_matrix!(A_hermite, hermite_data, basis, mon, k)
        RBF._build_collocation_matrix!(A_standard, data_2d, basis, mon, k)

        # Should be identical
        @test parent(A_hermite) ≈ parent(A_standard)
    end
end
