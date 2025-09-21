"""
Unit tests for collocation matrix assembly functions.
Tests both standard and Hermite variants of _build_collocation_matrix!
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
    basis_no_poly = PHS(3; poly_deg=0)   # Minimum polynomial augmentation
    mon_1d = MonomialBasis(1, 1)
    mon_2d = MonomialBasis(2, 1)

    # CURRENT LIMITATION: Hermite implementation only supports PHS basis functions
    # This is because Robin-Robin boundary interactions require directional∂²(basis, v1, v2)
    # which is only implemented for PHS1, PHS3, PHS5, PHS7
    # TODO: Implement directional∂² for IMQ and Gaussian to enable full Hermite support
    hermite_compatible_bases = [basis_phs]  # Only PHS for now
    all_bases = [basis_phs, basis_imq, basis_gaussian]  # For standard (non-Hermite) tests

    # 1D test data
    data_1d = [[0.0], [0.5], [1.0]]
    k_1d = 3

    # 2D test data  
    data_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    k_2d = 4

    @testset "Standard Collocation Matrix" begin
        @testset "RBF block assembly" begin
            # Test matrix with minimal polynomial augmentation
            mon_no_poly = MonomialBasis(1, 0)  # Just constant term
            n_no_poly = k_1d + 1  # RBF + constant
            A_no_poly = Symmetric(zeros(Float64, n_no_poly, n_no_poly), :U)

            # Build matrix with minimal polynomials
            RBF._build_collocation_matrix!(
                A_no_poly, data_1d, basis_no_poly, mon_no_poly, k_1d
            )
            AA_no_poly = parent(A_no_poly)

            # Check that RBF block is filled correctly  
            for i in 1:k_1d, j in 1:k_1d
                expected = basis_no_poly(data_1d[i], data_1d[j])
                @test A_no_poly[i, j] ≈ expected  # Use Symmetric wrapper, not parent
            end

            # Check matrix is finite
            @test all(isfinite.(AA_no_poly))
        end

        @testset "Polynomial augmentation" begin
            # Test matrix with polynomial augmentation
            nmon_1d = 2  # 1D linear: [1, x]
            n_1d = k_1d + nmon_1d
            A_poly = Symmetric(zeros(Float64, n_1d, n_1d), :U)

            # Build matrix with polynomial augmentation
            RBF._build_collocation_matrix!(A_poly, data_1d, basis_phs, mon_1d, k_1d)
            AA_poly = parent(A_poly)

            # Check RBF block (upper-left k×k)
            for i in 1:k_1d, j in 1:k_1d
                expected = basis_phs(data_1d[i], data_1d[j])
                if i <= j
                    @test AA_poly[i, j] ≈ expected
                end
            end

            # Check polynomial block (upper-right k×nmon and lower-left nmon×k)
            for i in 1:k_1d
                p_vals = zeros(nmon_1d)
                mon_1d(p_vals, data_1d[i])
                for j in 1:nmon_1d
                    col_idx = k_1d + j
                    @test AA_poly[i, col_idx] ≈ p_vals[j]
                end
            end

            # Check matrix is finite
            @test all(isfinite.(AA_poly))
        end

        @testset "Matrix symmetry" begin
            # Test that built matrix maintains symmetry
            nmon_1d = 2
            n_1d = k_1d + nmon_1d
            A_sym = Symmetric(zeros(Float64, n_1d, n_1d), :U)

            RBF._build_collocation_matrix!(A_sym, data_1d, basis_phs, mon_1d, k_1d)

            # Check that matrix is symmetric
            @test issymmetric(A_sym)

            # Check that only upper triangular part was filled
            AA_sym = parent(A_sym)
            for i in 1:n_1d, j in 1:(i - 1)
                # Lower triangular should be zero in parent matrix (Symmetric handles reflection)
                @test AA_sym[i, j] == 0.0
            end
        end

        @testset "2D standard matrices" begin
            # Test 2D matrix assembly
            nmon_2d = 3  # 2D linear: [1, x, y]
            n_2d = k_2d + nmon_2d
            A_2d = Symmetric(zeros(Float64, n_2d, n_2d), :U)

            RBF._build_collocation_matrix!(A_2d, data_2d, basis_phs, mon_2d, k_2d)
            AA_2d = parent(A_2d)

            # Check some RBF entries
            @test AA_2d[1, 1] ≈ basis_phs(data_2d[1], data_2d[1])
            @test AA_2d[1, 2] ≈ basis_phs(data_2d[1], data_2d[2])

            # Check polynomial entries for first point [0,0]
            p_vals = zeros(nmon_2d)
            mon_2d(p_vals, data_2d[1])  # Should be [1, 0, 0]
            @test AA_2d[1, k_2d + 1] ≈ 1.0  # Constant term
            @test AA_2d[1, k_2d + 2] ≈ 0.0  # x term
            @test AA_2d[1, k_2d + 3] ≈ 0.0  # y term

            # Check polynomial entries for point [1,1]
            mon_2d(p_vals, data_2d[4])  # Should be [1, 1, 1]
            @test AA_2d[4, k_2d + 1] ≈ 1.0  # Constant term
            @test AA_2d[4, k_2d + 2] ≈ 1.0  # x term
            @test AA_2d[4, k_2d + 3] ≈ 1.0  # y term

            @test all(isfinite.(AA_2d))
        end

        @testset "Different basis functions" begin
            # Test that different basis functions produce different matrices (standard case)
            n_1d = k_1d + 2

            # Test all basis functions work for standard (non-Hermite) case
            matrices = []
            for (i, basis) in enumerate(all_bases)
                A = Symmetric(zeros(Float64, n_1d, n_1d), :U)
                @test_nowarn RBF._build_collocation_matrix!(A, data_1d, basis, mon_1d, k_1d)
                push!(matrices, A)
                @test all(isfinite.(parent(A)))
            end

            # RBF blocks should be different between basis functions
            @test parent(matrices[1])[1, 2] != parent(matrices[2])[1, 2]  # PHS vs IMQ
            @test parent(matrices[1])[1, 2] != parent(matrices[3])[1, 2]  # PHS vs Gaussian
            @test parent(matrices[2])[1, 2] != parent(matrices[3])[1, 2]  # IMQ vs Gaussian

            # Polynomial blocks should be the same across all basis functions
            for i in 2:length(matrices)
                @test parent(matrices[1])[1, k_1d + 1] ≈ parent(matrices[i])[1, k_1d + 1]
                @test parent(matrices[1])[1, k_1d + 2] ≈ parent(matrices[i])[1, k_1d + 2]
            end
        end
    end

    @testset "Hermite Collocation Matrix" begin
        # NOTE: Currently only testing PHS basis functions due to directional∂² limitation
        # When directional∂² is implemented for IMQ/Gaussian, expand these tests to hermite_compatible_bases

        @testset "Interior points (no boundary)" begin
            # Test that interior points produce same result as standard
            is_boundary = [false, false, false]
            bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
            normals = [[0.0], [0.0], [0.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            for basis in hermite_compatible_bases
                # Build standard matrix
                n_1d = k_1d + 2
                A_standard = Symmetric(zeros(Float64, n_1d, n_1d), :U)
                RBF._build_collocation_matrix!(A_standard, data_1d, basis, mon_1d, k_1d)

                # Build Hermite matrix with no boundaries
                A_hermite = Symmetric(zeros(Float64, n_1d, n_1d), :U)
                RBF._build_collocation_matrix!(A_hermite, hermite_data, basis, mon_1d, k_1d)

                # Should be identical
                @test parent(A_standard) ≈ parent(A_hermite)
            end
        end

        @testset "Single Dirichlet boundary" begin
            # Test with one Dirichlet boundary point
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Build Hermite matrix
            n_1d = k_1d + 2
            A_hermite = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_hermite, hermite_data, basis_phs, mon_1d, k_1d)

            # Build standard matrix for comparison
            A_standard = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_standard, data_1d, basis_phs, mon_1d, k_1d)

            # Dirichlet boundary should give same result as standard
            @test parent(A_hermite) ≈ parent(A_standard)
        end

        @testset "Single Neumann boundary" begin
            # Test with one Neumann boundary point
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Neumann(), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Build Hermite matrix
            n_1d = k_1d + 2
            A_hermite = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_hermite, hermite_data, basis_phs, mon_1d, k_1d)

            # Build standard matrix for comparison
            A_standard = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_standard, data_1d, basis_phs, mon_1d, k_1d)

            # Should be different due to Neumann boundary
            @test parent(A_hermite) != parent(A_standard)

            # But should still be finite and symmetric
            @test all(isfinite.(parent(A_hermite)))
            @test issymmetric(A_hermite)
        end

        @testset "Single Robin boundary" begin
            # Test with one Robin boundary point
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Robin(0.5, 2.0), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Build Hermite matrix
            n_1d = k_1d + 2
            A_hermite = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_hermite, hermite_data, basis_phs, mon_1d, k_1d)

            # Build standard matrix for comparison
            A_standard = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_standard, data_1d, basis_phs, mon_1d, k_1d)

            # Should be different due to Robin boundary
            @test parent(A_hermite) != parent(A_standard)

            # Test coefficient sensitivity
            bcs_different = [Dirichlet(), Robin(1.0, 0.5), Dirichlet()]
            hermite_data_diff = RBF.HermiteStencilData(
                data_1d, is_boundary, bcs_different, normals
            )
            A_different = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(
                A_different, hermite_data_diff, basis_phs, mon_1d, k_1d
            )

            @test parent(A_hermite) != parent(A_different)
            @test all(isfinite.(parent(A_hermite)))
            @test issymmetric(A_hermite)
        end

        @testset "Multiple boundary points" begin
            # Test with multiple boundary points
            is_boundary = [true, true, false]
            bcs = [Neumann(), Robin(1.0, 1.0), Dirichlet()]
            normals = [[1.0], [-1.0], [0.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Build Hermite matrix
            n_1d = k_1d + 2
            A_hermite = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_hermite, hermite_data, basis_phs, mon_1d, k_1d)

            # Should be finite and symmetric
            @test all(isfinite.(parent(A_hermite)))
            @test issymmetric(A_hermite)

            # Should be different from standard
            A_standard = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_standard, data_1d, basis_phs, mon_1d, k_1d)
            @test parent(A_hermite) != parent(A_standard)
        end

        @testset "2D Hermite matrices" begin
            # Test 2D Hermite matrix assembly
            is_boundary_2d = [false, true, false, true]
            bcs_2d = [Dirichlet(), Neumann(), Dirichlet(), Robin(0.5, 1.0)]
            normals_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]

            hermite_data_2d = RBF.HermiteStencilData(
                data_2d, is_boundary_2d, bcs_2d, normals_2d
            )

            # Build 2D Hermite matrix
            nmon_2d = 3
            n_2d = k_2d + nmon_2d
            A_2d_hermite = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            RBF._build_collocation_matrix!(
                A_2d_hermite, hermite_data_2d, basis_phs, mon_2d, k_2d
            )

            # Should be finite and symmetric
            @test all(isfinite.(parent(A_2d_hermite)))
            @test issymmetric(A_2d_hermite)

            # Should be different from standard 2D matrix
            A_2d_standard = Symmetric(zeros(Float64, n_2d, n_2d), :U)
            RBF._build_collocation_matrix!(A_2d_standard, data_2d, basis_phs, mon_2d, k_2d)
            @test parent(A_2d_hermite) != parent(A_2d_standard)
        end

        @testset "Normal direction sensitivity" begin
            # Test that different normal directions give different results
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Neumann(), Dirichlet()]

            # x-direction normal
            normals_x = [[0.0], [1.0], [0.0]]
            hermite_data_x = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals_x)

            # Different direction (negative x)
            normals_neg_x = [[0.0], [-1.0], [0.0]]
            hermite_data_neg_x = RBF.HermiteStencilData(
                data_1d, is_boundary, bcs, normals_neg_x
            )

            # Build matrices
            n_1d = k_1d + 2
            A_x = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            A_neg_x = Symmetric(zeros(Float64, n_1d, n_1d), :U)

            RBF._build_collocation_matrix!(A_x, hermite_data_x, basis_phs, mon_1d, k_1d)
            RBF._build_collocation_matrix!(
                A_neg_x, hermite_data_neg_x, basis_phs, mon_1d, k_1d
            )

            # Different normals should give different matrices
            @test parent(A_x) != parent(A_neg_x)
        end

        @testset "Matrix structure preservation" begin
            # Test that Hermite modifications preserve important matrix properties
            is_boundary = [true, false, true]
            bcs = [Robin(1.0, 0.5), Dirichlet(), Neumann()]
            normals = [[1.0], [0.0], [-1.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Build matrix
            n_1d = k_1d + 2
            A_hermite = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            RBF._build_collocation_matrix!(A_hermite, hermite_data, basis_phs, mon_1d, k_1d)

            # Check properties
            @test issymmetric(A_hermite)  # Symmetry preserved
            @test all(isfinite.(parent(A_hermite)))  # No infinities
            @test !any(isnan.(parent(A_hermite)))  # No NaNs

            # Check that polynomial constraint block structure is preserved
            AA = parent(A_hermite)
            # Lower-right polynomial block should be zero
            for i in (k_1d + 1):n_1d, j in (k_1d + 1):n_1d
                if i <= j
                    @test AA[i, j] == 0.0
                end
            end
        end
    end

    @testset "Matrix Assembly Integration" begin
        @testset "Function dispatch correctness" begin
            # Verify that correct functions are called for different data types

            # Standard data should work
            n_1d = k_1d + 2
            A_standard = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            @test_nowarn RBF._build_collocation_matrix!(
                A_standard, data_1d, basis_phs, mon_1d, k_1d
            )

            # Hermite data should work
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Neumann(), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]
            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            A_hermite = Symmetric(zeros(Float64, n_1d, n_1d), :U)
            @test_nowarn RBF._build_collocation_matrix!(
                A_hermite, hermite_data, basis_phs, mon_1d, k_1d
            )
        end

        @testset "Basis function compatibility" begin
            # Test Hermite-compatible basis functions with complex boundary conditions
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Robin(1.0, 0.5), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]
            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            n_1d = k_1d + 2

            # Test all Hermite-compatible bases with Robin boundaries
            for basis in hermite_compatible_bases
                A = Symmetric(zeros(Float64, n_1d, n_1d), :U)
                @test_nowarn RBF._build_collocation_matrix!(
                    A, hermite_data, basis, mon_1d, k_1d
                )
                @test all(isfinite.(parent(A)))
            end

            # TODO: When directional∂² is implemented for IMQ/Gaussian, test them here too
            # For now, test non-Hermite-compatible bases only with ALL interior points
            # to completely avoid any boundary interactions requiring directional∂²
            is_interior_only = [false, false, false]  # All interior points
            bcs_interior = [Dirichlet(), Dirichlet(), Dirichlet()]  # Unused but required
            normals_interior = [[0.0], [0.0], [0.0]]  # Unused but required
            hermite_data_interior = RBF.HermiteStencilData(
                data_1d, is_interior_only, bcs_interior, normals_interior
            )

            non_hermite_bases = [basis_imq, basis_gaussian]
            for basis in non_hermite_bases
                A = Symmetric(zeros(Float64, n_1d, n_1d), :U)
                @test_nowarn RBF._build_collocation_matrix!(
                    A, hermite_data_interior, basis, mon_1d, k_1d
                )
                @test all(isfinite.(parent(A)))

                # Should match standard collocation matrix for interior-only case
                A_standard = Symmetric(zeros(Float64, n_1d, n_1d), :U)
                RBF._build_collocation_matrix!(A_standard, data_1d, basis, mon_1d, k_1d)
                @test parent(A) ≈ parent(A_standard)
            end
        end

        @testset "Matrix size consistency" begin
            # Test that matrix dimensions are handled correctly

            # Test different k values
            for k_test in [2, 3, 5]
                data_test = [rand(1) for _ in 1:k_test]
                n_test = k_test + 2  # With linear polynomials

                A_test = Symmetric(zeros(Float64, n_test, n_test), :U)
                @test_nowarn RBF._build_collocation_matrix!(
                    A_test, data_test, basis_phs, mon_1d, k_test
                )
                @test size(A_test) == (n_test, n_test)
            end

            # Test minimal polynomial case (degree 0 - just constant)
            k_no_poly = 4
            data_no_poly = [rand(1) for _ in 1:k_no_poly]
            n_minimal = k_no_poly + 1  # RBF + constant term
            A_minimal = Symmetric(zeros(Float64, n_minimal, n_minimal), :U)

            mon_minimal = MonomialBasis(1, 0)  # Just constant term [1]
            @test_nowarn RBF._build_collocation_matrix!(
                A_minimal, data_no_poly, basis_no_poly, mon_minimal, k_no_poly
            )
            @test size(A_minimal) == (n_minimal, n_minimal)
        end
    end
end
