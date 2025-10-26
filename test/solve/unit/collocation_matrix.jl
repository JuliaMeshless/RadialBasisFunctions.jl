"""
Unit tests for collocation matrix assembly functions.
Tests both standard and Hermite variants of _build_collocation_matrix!

Test Strategy:
- Primary tests use 2D data (more realistic, better coverage)
- Minimal 1D tests for edge cases and dimension-specific behavior
- 3D smoke test to verify scalability
"""

using Test
using LinearAlgebra
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Collocation Matrix Building" begin

    # Test setup - common data for all tests
    basis_phs = PHS(3; poly_deg=1)
    basis_imq = IMQ(1.0)
    basis_gaussian = Gaussian(1.0)
    basis_no_poly = PHS(3; poly_deg=0)

    # LIMITATION: Hermite only supports PHS (needs directional∂² for IMQ/Gaussian)
    hermite_compatible_bases = [basis_phs]
    all_bases = [basis_phs, basis_imq, basis_gaussian]

    # Primary 2D test data (used for most tests)
    data_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    k_2d, mon_2d, nmon_2d = 4, MonomialBasis(2, 1), 3
    n_2d = k_2d + nmon_2d

    # Minimal 1D data (for edge cases only)
    data_1d = [[0.0], [0.5], [1.0]]
    k_1d, mon_1d = 3, MonomialBasis(1, 1)

    # 3D data (for smoke test)
    data_3d = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    k_3d, mon_3d, nmon_3d = 4, MonomialBasis(3, 1), 4
    n_3d = k_3d + nmon_3d

    # Helper: Build both standard and Hermite matrices for comparison
    function build_matrix_pair(data, hermite_data, basis, mon, k, n)
        A_std = Symmetric(zeros(Float64, n, n), :U)
        A_herm = Symmetric(zeros(Float64, n, n), :U)
        RBF._build_collocation_matrix!(A_std, data, basis, mon, k)
        RBF._build_collocation_matrix!(A_herm, hermite_data, basis, mon, k)
        return A_std, A_herm
    end

    @testset "Standard Collocation Matrix (2D)" begin
        @testset "RBF block assembly" begin
            # Test 2D RBF block is filled correctly
            A_2d = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            RBF._build_collocation_matrix!(A_2d, data_2d, basis_phs, mon_2d, k_2d)
            AA_2d = parent(A_2d)

            # Check RBF entries
            @test AA_2d[1, 1] ≈ basis_phs(data_2d[1], data_2d[1])
            @test AA_2d[1, 2] ≈ basis_phs(data_2d[1], data_2d[2])
            @test AA_2d[2, 3] ≈ basis_phs(data_2d[2], data_2d[3])
            @test all(isfinite.(AA_2d))
        end

        @testset "Polynomial augmentation (2D)" begin
            # Test 2D polynomial block
            A_2d = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            RBF._build_collocation_matrix!(A_2d, data_2d, basis_phs, mon_2d, k_2d)
            AA_2d = parent(A_2d)

            # Check polynomial entries for first point [0,0]
            @test AA_2d[1, k_2d + 1] ≈ 1.0  # Constant term
            @test AA_2d[1, k_2d + 2] ≈ 0.0  # x term
            @test AA_2d[1, k_2d + 3] ≈ 0.0  # y term

            # Check polynomial entries for point [1,1]
            @test AA_2d[4, k_2d + 1] ≈ 1.0  # Constant term
            @test AA_2d[4, k_2d + 2] ≈ 1.0  # x term
            @test AA_2d[4, k_2d + 3] ≈ 1.0  # y term

            # Lower-right polynomial constraint block should be zero
            for i in (k_2d + 1):n_2d, j in (k_2d + 1):n_2d
                if i <= j
                    @test AA_2d[i, j] == 0.0
                end
            end
        end

        @testset "Matrix symmetry (2D)" begin
            # Test that built matrix maintains symmetry
            A_sym = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            RBF._build_collocation_matrix!(A_sym, data_2d, basis_phs, mon_2d, k_2d)

            @test issymmetric(A_sym)

            # Check that only upper triangular part was filled
            AA_sym = parent(A_sym)
            for i in 1:n_2d, j in 1:(i - 1)
                @test AA_sym[i, j] == 0.0  # Lower triangle zero in parent
            end
        end
    end

    @testset "Hermite Collocation Matrix (2D)" begin
        # NOTE: Currently only testing PHS basis functions due to directional∂² limitation
        # When directional∂² is implemented for IMQ/Gaussian, expand these tests

        @testset "Interior points (no boundary)" begin
            # Test that all-interior Hermite matches standard
            is_boundary_2d = [false, false, false, false]
            bcs_2d = [Internal(), Internal(), Internal(), Internal()]
            normals_2d = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

            hermite_data_2d = RBF.HermiteStencilData(
                data_2d, is_boundary_2d, bcs_2d, normals_2d
            )

            # Build both matrices
            A_standard = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            A_hermite = Symmetric(zeros(Float64, n_2d, n_2d), :U)

            RBF._build_collocation_matrix!(A_standard, data_2d, basis_phs, mon_2d, k_2d)
            RBF._build_collocation_matrix!(
                A_hermite, hermite_data_2d, basis_phs, mon_2d, k_2d
            )

            # Should be identical
            @test parent(A_standard) ≈ parent(A_hermite)
        end

        @testset "Single boundary conditions" begin
            is_boundary = [false, true, false, false]
            normals = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

            # Dirichlet: should match standard
            hermite_dirichlet = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Internal(), Dirichlet(), Internal(), Internal()],
                normals,
            )
            A_std, A_dir = build_matrix_pair(
                data_2d, hermite_dirichlet, basis_phs, mon_2d, k_2d, n_2d
            )
            @test parent(A_dir) ≈ parent(A_std)

            # Neumann: should differ from standard
            hermite_neumann = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Internal(), Neumann(), Internal(), Internal()],
                normals,
            )
            A_std, A_neu = build_matrix_pair(
                data_2d, hermite_neumann, basis_phs, mon_2d, k_2d, n_2d
            )
            @test parent(A_neu) != parent(A_std)
            @test all(isfinite.(parent(A_neu)))
            @test issymmetric(A_neu)

            # Robin: should differ from standard and be coefficient-sensitive
            hermite_robin1 = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Internal(), Robin(0.5, 2.0), Internal(), Internal()],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],  # y-direction
            )
            hermite_robin2 = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Internal(), Robin(1.0, 0.5), Internal(), Internal()],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            )
            A_std, A_rob1 = build_matrix_pair(
                data_2d, hermite_robin1, basis_phs, mon_2d, k_2d, n_2d
            )
            _, A_rob2 = build_matrix_pair(
                data_2d, hermite_robin2, basis_phs, mon_2d, k_2d, n_2d
            )

            @test parent(A_rob1) != parent(A_std)
            @test parent(A_rob1) != parent(A_rob2)  # Coefficient sensitivity
            @test all(isfinite.(parent(A_rob1)))
            @test issymmetric(A_rob1)
        end

        @testset "Multiple boundary points with mixed BCs" begin
            # Test with multiple boundary points of different types
            is_boundary_2d = [false, true, false, true]
            bcs_2d = [Internal(), Neumann(), Internal(), Robin(0.5, 1.0)]
            normals_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]

            hermite_data_2d = RBF.HermiteStencilData(
                data_2d, is_boundary_2d, bcs_2d, normals_2d
            )

            # Build matrices
            A_hermite = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            A_standard = Symmetric(zeros(Float64, n_2d, n_2d), :U)

            RBF._build_collocation_matrix!(
                A_hermite, hermite_data_2d, basis_phs, mon_2d, k_2d
            )
            RBF._build_collocation_matrix!(A_standard, data_2d, basis_phs, mon_2d, k_2d)

            # Should be different from standard
            @test parent(A_hermite) != parent(A_standard)
            @test all(isfinite.(parent(A_hermite)))
            @test issymmetric(A_hermite)
        end

        @testset "Boundary-boundary interactions (2D)" begin
            # Test Robin-Robin mixed derivative code path (both i and j are boundaries)
            is_boundary = [true, true, false, false]
            normals = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

            hermite_robin_robin = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Robin(1.0, 0.5), Robin(0.5, 2.0), Internal(), Internal()],
                normals,
            )
            A_std, A_herm = build_matrix_pair(
                data_2d, hermite_robin_robin, basis_phs, mon_2d, k_2d, n_2d
            )

            # Entry (1,2) involves Robin-Robin interaction with mixed derivatives
            @test parent(A_herm)[1, 2] != parent(A_std)[1, 2]
            @test isfinite(parent(A_herm)[1, 2])
            @test parent(A_herm)[3, 4] ≈ parent(A_std)[3, 4]  # Interior-interior matches

            # Test coefficient sensitivity for boundary-boundary entries
            hermite_different = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Robin(2.0, 0.1), Robin(0.5, 2.0), Internal(), Internal()],
                normals,
            )
            _, A_diff = build_matrix_pair(
                data_2d, hermite_different, basis_phs, mon_2d, k_2d, n_2d
            )
            @test parent(A_diff)[1, 2] != parent(A_herm)[1, 2]
        end

        @testset "Boundary diagonal entries (2D)" begin
            # For PHS: diagonal RBF entries always 0 (φ(x,x)=r³=0, ∇φ(x,x)=0)
            # True for both standard and Hermite, regardless of boundary conditions
            hermite_data = RBF.HermiteStencilData(
                data_2d,
                [false, true, false, true],
                [Internal(), Robin(1.0, 1.0), Internal(), Neumann()],
                [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            )
            A_std, A_herm = build_matrix_pair(
                data_2d, hermite_data, basis_phs, mon_2d, k_2d, n_2d
            )

            # All diagonal entries are zero and match between standard and Hermite
            for i in 1:k_2d
                @test parent(A_herm)[i, i] ≈ parent(A_std)[i, i] ≈ 0.0
                @test isfinite(parent(A_herm)[i, i])
            end
        end

        @testset "Polynomial augmentation with boundaries (2D)" begin
            is_boundary = [false, true, false, false]
            normals_x = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

            # Neumann: applies derivative operator to polynomials
            hermite_neumann = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Internal(), Neumann(), Internal(), Internal()],
                normals_x,
            )
            A_std, A_neu = build_matrix_pair(
                data_2d, hermite_neumann, basis_phs, mon_2d, k_2d, n_2d
            )

            # Interior points: polynomial entries match standard
            for j in 1:nmon_2d
                @test parent(A_neu)[1, k_2d + j] ≈ parent(A_std)[1, k_2d + j]
            end

            # Boundary point at [1,0] with Neumann ∂/∂x: constant term differs
            # Standard: P=1, Neumann: ∂P/∂x=0
            @test parent(A_neu)[2, k_2d + 1] ≈ 0.0  # ∂(1)/∂x
            @test parent(A_std)[2, k_2d + 1] ≈ 1.0  # value of constant
            @test parent(A_neu)[2, k_2d + 1] != parent(A_std)[2, k_2d + 1]

            # Robin: combines value and derivative (α*P + β*∂P/∂n)
            hermite_robin = RBF.HermiteStencilData(
                data_2d,
                is_boundary,
                [Internal(), Robin(0.5, 1.5), Internal(), Internal()],
                normals_x,
            )
            _, A_rob = build_matrix_pair(
                data_2d, hermite_robin, basis_phs, mon_2d, k_2d, n_2d
            )

            # For constant: 0.5*1 + 1.5*0 = 0.5
            @test parent(A_rob)[2, k_2d + 1] ≈ 0.5
            @test parent(A_rob)[2, k_2d + 1] != parent(A_std)[2, k_2d + 1]
            @test parent(A_rob)[2, k_2d + 1] != parent(A_neu)[2, k_2d + 1]
        end

        @testset "All boundary points (2D)" begin
            # Test extreme case where ALL points are boundary points
            is_boundary_all = [true, true, true, true]
            bcs_all = [Dirichlet(), Neumann(), Robin(1.0, 1.0), Robin(0.5, 2.0)]
            normals_all = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]

            hermite_data_all = RBF.HermiteStencilData(
                data_2d, is_boundary_all, bcs_all, normals_all
            )

            A_all_boundary = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            @test_nowarn RBF._build_collocation_matrix!(
                A_all_boundary, hermite_data_all, basis_phs, mon_2d, k_2d
            )

            # Matrix should still be valid
            @test issymmetric(A_all_boundary)
            @test all(isfinite.(parent(A_all_boundary)))
            @test !any(isnan.(parent(A_all_boundary)))

            # Should differ significantly from standard due to all boundary modifications
            A_standard = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            RBF._build_collocation_matrix!(A_standard, data_2d, basis_phs, mon_2d, k_2d)

            # Multiple entries should differ
            diff_count = sum(parent(A_all_boundary) .!= parent(A_standard))
            @test diff_count > 0  # At least some entries should differ
        end
    end

    @testset "3D Smoke Test" begin
        # Basic smoke test to verify 3D functionality

        @testset "Standard 3D matrix" begin
            A_3d = Symmetric(zeros(Float64, n_3d, n_3d), :U)
            @test_nowarn RBF._build_collocation_matrix!(
                A_3d, data_3d, basis_phs, mon_3d, k_3d
            )

            @test issymmetric(A_3d)
            @test all(isfinite.(parent(A_3d)))
            @test size(A_3d) == (n_3d, n_3d)
        end

        @testset "Hermite 3D with boundaries" begin
            # Test 3D Hermite with mixed boundary conditions
            is_boundary_3d = [false, true, false, true]
            bcs_3d = [Internal(), Neumann(), Internal(), Robin(1.0, 0.5)]
            normals_3d = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # x-direction
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],  # z-direction
            ]

            hermite_data_3d = RBF.HermiteStencilData(
                data_3d, is_boundary_3d, bcs_3d, normals_3d
            )

            A_3d_hermite = Symmetric(zeros(Float64, n_3d, n_3d), :U)
            @test_nowarn RBF._build_collocation_matrix!(
                A_3d_hermite, hermite_data_3d, basis_phs, mon_3d, k_3d
            )

            @test issymmetric(A_3d_hermite)
            @test all(isfinite.(parent(A_3d_hermite)))

            # Should differ from standard due to boundary conditions
            A_3d_standard = Symmetric(zeros(Float64, n_3d, n_3d), :U)
            RBF._build_collocation_matrix!(A_3d_standard, data_3d, basis_phs, mon_3d, k_3d)
            @test parent(A_3d_hermite) != parent(A_3d_standard)
        end
    end
end
