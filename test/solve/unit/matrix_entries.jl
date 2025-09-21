"""
Unit tests for matrix entry computation functions.
Tests both standard and Hermite variants, plus dispatch verification.
"""

using Test
using LinearAlgebra
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Matrix Entry Functions" begin

    # Test setup - common data for all tests
    basis_phs = PHS(3; poly_deg=1)
    basis_imq = IMQ(1.0)
    mon_1d = MonomialBasis(1, 1)
    mon_2d = MonomialBasis(2, 1)

    # 1D test data
    data_1d = [[0.0], [0.5], [1.0]]

    # 2D test data  
    data_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    @testset "Standard RBF Matrix Entries" begin
        @testset "Standard basis evaluation" begin
            # Test that basis functions work as expected
            x1, x2 = [0.0], [1.0]
            @test basis_phs(x1, x2) ≈ basis_phs([0.0], [1.0])
            @test isfinite(basis_phs(x1, x2))
            @test isfinite(basis_imq(x1, x2))

            # Test symmetry
            @test basis_phs(x1, x2) ≈ basis_phs(x2, x1)
            @test basis_imq(x1, x2) ≈ basis_imq(x2, x1)
        end

        @testset "Distance-based functions" begin
            # Test that RBF depends on distance
            x1, x2, x3 = [0.0], [1.0], [2.0]

            # For PHS(3): r³, larger distance gives larger values
            val_close = basis_phs(x1, [0.1])
            val_far = basis_phs(x1, x2)
            @test val_far > val_close  # PHS3 increases with distance

            # Same distance should give same value
            @test basis_phs(x1, x2) ≈ basis_phs([0.5], [1.5])
        end

        @testset "2D standard entries" begin
            # Test 2D basis functions
            x1, x2 = [0.0, 0.0], [1.0, 1.0]
            @test isfinite(basis_phs(x1, x2))
            @test isfinite(basis_imq(x1, x2))

            # Test distance calculation is correct
            dist = sqrt(2.0)  # Distance from (0,0) to (1,1)
            @test basis_phs(x1, x2) ≈ basis_phs([0.0], [dist])
        end
    end

    @testset "Hermite RBF Matrix Entries" begin
        @testset "Interior-Interior entries" begin
            # Setup interior points (no boundary)
            is_boundary = [false, false, false]
            bcs = [Dirichlet(), Dirichlet(), Dirichlet()]  # Unused for interior
            normals = [[0.0], [0.0], [0.0]]  # Unused for interior

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Interior-Interior should match standard RBF evaluation
            val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
            val_standard = basis_phs(data_1d[1], data_1d[2])

            @test val_hermite ≈ val_standard
            @test isfinite(val_hermite)
        end

        @testset "Interior-Boundary entries" begin
            @testset "Interior-Dirichlet" begin
                # Setup: point 1 interior, point 2 Dirichlet boundary
                is_boundary = [false, true, false]
                bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
                normals = [[0.0], [1.0], [0.0]]  # Normal for point 2

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Interior-Dirichlet should be standard φ(x1, x2)
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                @test val_hermite ≈ val_standard
            end

            @testset "Interior-Neumann" begin
                # Setup: point 1 interior, point 2 Neumann boundary
                is_boundary = [false, true, false]
                bcs = [Dirichlet(), Neumann(), Dirichlet()]
                normals = [[0.0], [1.0], [0.0]]  # Normal for point 2

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Interior-Neumann should be α*φ + β*∂ₙφ = 0*φ + 1*∂ₙφ = ∂ₙφ
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)

                # Should be different from standard evaluation
                val_standard = basis_phs(data_1d[1], data_1d[2])
                @test val_hermite != val_standard
                @test isfinite(val_hermite)
            end

            @testset "Interior-Robin" begin
                # Setup: point 1 interior, point 2 Robin boundary
                is_boundary = [false, true, false]
                bcs = [Dirichlet(), Robin(0.5, 2.0), Dirichlet()]
                normals = [[0.0], [1.0], [0.0]]  # Normal for point 2

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Interior-Robin should be α*φ + β*∂ₙφ = 0.5*φ + 2.0*∂ₙφ
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                # Should be different from standard and finite
                @test val_hermite != val_standard
                @test isfinite(val_hermite)

                # Test coefficient sensitivity
                bcs_different = [Dirichlet(), Robin(1.0, 0.5), Dirichlet()]
                hermite_data_diff = RBF.HermiteStencilData(
                    data_1d, is_boundary, bcs_different, normals
                )
                val_different = RBF._hermite_rbf_entry(1, 2, hermite_data_diff, basis_phs)
                @test val_hermite != val_different
            end
        end

        @testset "Boundary-Interior entries" begin
            @testset "Dirichlet-Interior" begin
                # Setup: point 1 Dirichlet boundary, point 2 interior
                is_boundary = [true, false, false]
                bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
                normals = [[1.0], [0.0], [0.0]]  # Normal for point 1

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Dirichlet-Interior should be standard φ(x1, x2)
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                @test val_hermite ≈ val_standard
            end

            @testset "Neumann-Interior" begin
                # Setup: point 1 Neumann boundary, point 2 interior
                is_boundary = [true, false, false]
                bcs = [Neumann(), Dirichlet(), Dirichlet()]
                normals = [[1.0], [0.0], [0.0]]  # Normal for point 1

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Neumann-Interior should be ∂ₙφ(x1, x2)
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                @test val_hermite != val_standard
                @test isfinite(val_hermite)
            end
        end

        @testset "Boundary-Boundary entries" begin
            @testset "Dirichlet-Dirichlet" begin
                # Setup: both points Dirichlet boundary
                is_boundary = [true, true, false]
                bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
                normals = [[1.0], [-1.0], [0.0]]  # Normals for points 1,2

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Dirichlet-Dirichlet should be standard φ(x1, x2)
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                @test val_hermite ≈ val_standard
            end

            @testset "Dirichlet-Neumann" begin
                # Setup: point 1 Dirichlet, point 2 Neumann
                is_boundary = [true, true, false]
                bcs = [Dirichlet(), Neumann(), Dirichlet()]
                normals = [[1.0], [-1.0], [0.0]]  # Normals for points 1,2

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Should apply Neumann operator to second argument
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                @test val_hermite != val_standard
                @test isfinite(val_hermite)
            end

            @testset "Neumann-Neumann" begin
                # Setup: both points Neumann boundary
                is_boundary = [true, true, false]
                bcs = [Neumann(), Neumann(), Dirichlet()]
                normals = [[1.0], [-1.0], [0.0]]  # Normals for points 1,2

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Should apply boundary operators to both arguments (mixed derivative)
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                @test val_hermite != val_standard
                @test isfinite(val_hermite)
            end

            @testset "Robin-Robin" begin
                # Setup: both points Robin boundary
                is_boundary = [true, true, false]
                bcs = [Robin(0.5, 1.0), Robin(1.0, 0.5), Dirichlet()]
                normals = [[1.0], [-1.0], [0.0]]  # Normals for points 1,2

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Should be most complex case with mixed derivatives
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val_standard = basis_phs(data_1d[1], data_1d[2])

                @test val_hermite != val_standard
                @test isfinite(val_hermite)
            end
        end

        @testset "2D Hermite entries" begin
            @testset "2D Neumann boundaries" begin
                # Setup 2D problem with Neumann boundary
                is_boundary_2d = [false, true, false, false]
                bcs_2d = [Dirichlet(), Neumann(), Dirichlet(), Dirichlet()]
                normals_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # x-direction normal

                hermite_data_2d = RBF.HermiteStencilData(
                    data_2d, is_boundary_2d, bcs_2d, normals_2d
                )

                # Test Interior-Neumann in 2D
                val_hermite = RBF._hermite_rbf_entry(1, 2, hermite_data_2d, basis_phs)
                val_standard = basis_phs(data_2d[1], data_2d[2])

                @test val_hermite != val_standard
                @test isfinite(val_hermite)
            end

            @testset "2D different normal directions" begin
                # Test with y-direction normal
                is_boundary_2d = [false, true, false, false]
                bcs_2d = [Dirichlet(), Neumann(), Dirichlet(), Dirichlet()]
                normals_2d = [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]  # y-direction normal

                hermite_data_2d = RBF.HermiteStencilData(
                    data_2d, is_boundary_2d, bcs_2d, normals_2d
                )

                val_y_normal = RBF._hermite_rbf_entry(1, 2, hermite_data_2d, basis_phs)

                # Change to x-direction normal and compare
                normals_2d_x = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
                hermite_data_2d_x = RBF.HermiteStencilData(
                    data_2d, is_boundary_2d, bcs_2d, normals_2d_x
                )
                val_x_normal = RBF._hermite_rbf_entry(1, 2, hermite_data_2d_x, basis_phs)

                # Different normal directions should give different results
                @test val_y_normal != val_x_normal
                @test isfinite(val_y_normal)
                @test isfinite(val_x_normal)
            end
        end
    end

    @testset "Polynomial Matrix Entries" begin
        @testset "Standard polynomial entries" begin
            # Test standard monomial evaluation
            a = zeros(2)  # For 1D linear: [1, x]
            x = [0.5]

            mon_1d(a, x)
            @test a[1] ≈ 1.0      # Constant term
            @test a[2] ≈ 0.5      # Linear term

            # Test 2D monomial
            a_2d = zeros(3)  # For 2D linear: [1, x, y]
            x_2d = [0.5, 0.3]

            mon_2d(a_2d, x_2d)
            @test a_2d[1] ≈ 1.0   # Constant term
            @test a_2d[2] ≈ 0.5   # x term
            @test a_2d[3] ≈ 0.3   # y term
        end

        @testset "Hermite polynomial entries" begin
            @testset "Interior polynomial entries" begin
                # Setup interior point
                is_boundary = [false, true, false]
                bcs = [Dirichlet(), Neumann(), Dirichlet()]
                normals = [[0.0], [1.0], [0.0]]

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Interior point should use standard polynomial evaluation
                a_interior = zeros(2)
                RBF._hermite_poly_entry!(a_interior, 1, hermite_data, mon_1d)

                # Compare with direct evaluation
                a_standard = zeros(2)
                mon_1d(a_standard, data_1d[1])

                @test a_interior ≈ a_standard
            end

            @testset "Dirichlet polynomial entries" begin
                # Setup Dirichlet boundary point
                is_boundary = [true, false, false]
                bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
                normals = [[1.0], [0.0], [0.0]]

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Dirichlet should use standard polynomial evaluation
                a_dirichlet = zeros(2)
                RBF._hermite_poly_entry!(a_dirichlet, 1, hermite_data, mon_1d)

                # Compare with direct evaluation
                a_standard = zeros(2)
                mon_1d(a_standard, data_1d[1])

                @test a_dirichlet ≈ a_standard
            end

            @testset "Neumann polynomial entries" begin
                # Setup Neumann boundary point
                is_boundary = [true, false, false]
                bcs = [Neumann(), Dirichlet(), Dirichlet()]
                normals = [[1.0], [0.0], [0.0]]

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Neumann should use α*P + β*∂ₙP = 0*P + 1*∂ₙP = ∂ₙP
                a_neumann = zeros(2)
                RBF._hermite_poly_entry!(a_neumann, 1, hermite_data, mon_1d)

                # Should be different from standard evaluation
                a_standard = zeros(2)
                mon_1d(a_standard, data_1d[1])

                @test a_neumann != a_standard
                @test isfinite(a_neumann[1])
                @test isfinite(a_neumann[2])

                # For linear polynomials: ∂P₁/∂x = 0, ∂P₂/∂x = 1
                @test a_neumann[1] ≈ 0.0   # ∂(1)/∂x = 0
                @test a_neumann[2] ≈ 1.0   # ∂(x)/∂x = 1
            end

            @testset "Robin polynomial entries" begin
                # Setup Robin boundary point
                is_boundary = [true, false, false]
                bcs = [Robin(0.5, 2.0), Dirichlet(), Dirichlet()]
                normals = [[1.0], [0.0], [0.0]]

                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                # Robin should use α*P + β*∂ₙP = 0.5*P + 2.0*∂ₙP
                a_robin = zeros(2)
                RBF._hermite_poly_entry!(a_robin, 1, hermite_data, mon_1d)

                # For point at x=0: P₁=1, P₂=0, ∂P₁/∂x=0, ∂P₂/∂x=1
                # So: [0.5*1 + 2.0*0, 0.5*0 + 2.0*1] = [0.5, 2.0]
                @test a_robin[1] ≈ 0.5
                @test a_robin[2] ≈ 2.0
            end

            @testset "2D polynomial entries" begin
                # Setup 2D Neumann boundary
                is_boundary_2d = [true, false, false, false]
                bcs_2d = [Neumann(), Dirichlet(), Dirichlet(), Dirichlet()]
                normals_2d = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # x-direction normal

                hermite_data_2d = RBF.HermiteStencilData(
                    data_2d, is_boundary_2d, bcs_2d, normals_2d
                )

                # 2D linear: [1, x, y], derivatives: [0, 1, 0] in x-direction
                a_2d_neumann = zeros(3)
                RBF._hermite_poly_entry!(a_2d_neumann, 1, hermite_data_2d, mon_2d)

                @test a_2d_neumann[1] ≈ 0.0   # ∂(1)/∂x = 0
                @test a_2d_neumann[2] ≈ 1.0   # ∂(x)/∂x = 1  
                @test a_2d_neumann[3] ≈ 0.0   # ∂(y)/∂x = 0
            end
        end
    end

    @testset "Dispatch Verification" begin
        @testset "Function existence and signatures" begin
            # Verify that both standard and Hermite functions exist
            @test hasmethod(
                RBF._hermite_rbf_entry,
                (Int, Int, RBF.HermiteStencilData, AbstractRadialBasis),
            )
            @test hasmethod(
                RBF._hermite_poly_entry!,
                (AbstractVector, Int, RBF.HermiteStencilData, MonomialBasis),
            )
        end

        @testset "Type dispatch correctness" begin
            # Setup both standard data and Hermite data
            is_boundary = [false, false, false]
            bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
            normals = [[0.0], [0.0], [0.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Test that functions can be called without errors
            @test_nowarn RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
            @test_nowarn RBF._hermite_rbf_entry(1, 2, hermite_data, basis_imq)

            a = zeros(2)
            @test_nowarn RBF._hermite_poly_entry!(a, 1, hermite_data, mon_1d)
        end

        @testset "Boundary condition dispatch" begin
            # Test that different boundary conditions produce different results
            normals = [[1.0], [0.0], [0.0]]

            # Create different boundary condition setups
            is_boundary = [true, false, false]

            bc_dirichlet = [Dirichlet(), Dirichlet(), Dirichlet()]
            bc_neumann = [Neumann(), Dirichlet(), Dirichlet()]
            bc_robin = [Robin(1.0, 1.0), Dirichlet(), Dirichlet()]

            hermite_dirichlet = RBF.HermiteStencilData(
                data_1d, is_boundary, bc_dirichlet, normals
            )
            hermite_neumann = RBF.HermiteStencilData(
                data_1d, is_boundary, bc_neumann, normals
            )
            hermite_robin = RBF.HermiteStencilData(data_1d, is_boundary, bc_robin, normals)

            # Get values for each case
            val_dirichlet = RBF._hermite_rbf_entry(1, 2, hermite_dirichlet, basis_phs)
            val_neumann = RBF._hermite_rbf_entry(1, 2, hermite_neumann, basis_phs)
            val_robin = RBF._hermite_rbf_entry(1, 2, hermite_robin, basis_phs)

            # Different boundary conditions should give different results (except Dirichlet acts like interior)
            standard_val = basis_phs(data_1d[1], data_1d[2])
            @test val_dirichlet ≈ standard_val  # Dirichlet should match standard
            @test val_neumann != standard_val   # Neumann should be different
            @test val_robin != standard_val     # Robin should be different
            @test val_neumann != val_robin      # Neumann and Robin should differ
        end

        @testset "Basis function compatibility" begin
            # Test that Hermite functions work with different basis types
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Neumann(), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Test different basis functions
            @test_nowarn RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
            @test_nowarn RBF._hermite_rbf_entry(1, 2, hermite_data, basis_imq)

            # All should produce finite results
            val_phs = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
            val_imq = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_imq)

            @test isfinite(val_phs)
            @test isfinite(val_imq)
        end
    end
end
