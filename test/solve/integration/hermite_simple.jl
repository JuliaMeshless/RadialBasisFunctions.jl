"""
Simple integration tests for solve_hermite.jl functionality.
Tests basic Hermite stencil building and boundary condition dispatch.
Focuses on what actually works rather than complex internal APIs.

CURRENT LIMITATION: Hermite interpolation is only available for PHS basis functions.
When IMQ/Gaussian get the required operators, expand hermite_compatible_bases below.
"""

using Test
using LinearAlgebra
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "Hermite Simple Integration Tests" begin

    # Test setup - basis function configuration
    basis_phs = PHS(3; poly_deg=1)
    basis_imq = IMQ(1.0)
    basis_gaussian = Gaussian(1.0)

    # IMPORTANT: Hermite functionality is currently PHS-only
    hermite_compatible_bases = [basis_phs]  # TODO: Add basis_imq, basis_gaussian when operators are implemented
    all_bases = [basis_phs, basis_imq, basis_gaussian]  # For standard tests

    # Simple test data
    data_1d = [[0.0], [0.5], [1.0]]
    eval_point_1d = [0.25]

    @testset "Basic Hermite Data Structure" begin
        @testset "HermiteStencilData Creation" begin
            # Test that we can create HermiteStencilData structures
            is_boundary = [false, true, false]
            bcs = [RBF.Dirichlet(), RBF.Neumann(), RBF.Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]

            @test_nowarn hermite_data = RBF.HermiteStencilData(
                data_1d, is_boundary, bcs, normals
            )

            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Test basic properties
            @test hermite_data isa RBF.HermiteStencilData
            @test length(hermite_data.data) == 3
            @test length(hermite_data.is_boundary) == 3
            @test length(hermite_data.boundary_conditions) == 3
            @test length(hermite_data.normals) == 3
            @test hermite_data.is_boundary == is_boundary
        end

        @testset "Boundary Condition Types" begin
            # Test boundary condition type detection
            bc_dirichlet = RBF.Dirichlet()
            bc_neumann = RBF.Neumann()
            bc_robin = RBF.Robin(1.0, 2.0)

            @test RBF.is_dirichlet(bc_dirichlet)
            @test !RBF.is_neumann(bc_dirichlet)
            @test !RBF.is_robin(bc_dirichlet)

            @test !RBF.is_dirichlet(bc_neumann)
            @test RBF.is_neumann(bc_neumann)
            @test !RBF.is_robin(bc_neumann)

            @test !RBF.is_dirichlet(bc_robin)
            @test !RBF.is_neumann(bc_robin)
            @test RBF.is_robin(bc_robin)

            # Test Robin coefficients
            @test RBF.α(bc_robin) == 1.0
            @test RBF.β(bc_robin) == 2.0
        end
    end

    @testset "Matrix Building Functions" begin
        @testset "Collocation Matrix Building" begin
            # Test basic collocation matrix building for compatible bases
            k = 3
            mon_1d = MonomialBasis(1, 1)
            nmon = 2
            n = k + nmon

            for basis in hermite_compatible_bases  # Currently only PHS
                # Standard case for comparison
                A_std = Symmetric(zeros(Float64, n, n), :U)
                @test_nowarn RBF._build_collocation_matrix!(
                    A_std, data_1d, basis, mon_1d, k
                )

                # Hermite case with all interior points (should be similar)
                is_boundary = [false, false, false]
                bcs = [RBF.Internal(), RBF.Internal(), RBF.Internal()]  # Interior sentinel values
                normals = [[0.0], [0.0], [0.0]]
                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                A_herm = Symmetric(zeros(Float64, n, n), :U)
                @test_nowarn RBF._build_collocation_matrix!(
                    A_herm, hermite_data, basis, mon_1d, k
                )

                # Both should be valid matrices
                @test size(A_std) == (n, n)
                @test size(A_herm) == (n, n)
                @test all(isfinite.(A_std))
                @test all(isfinite.(A_herm))
            end
        end

        @testset "Matrix Entry Computation" begin
            # Test individual matrix entry computation
            data_test = [[0.0], [1.0]]

            for basis in hermite_compatible_bases  # Currently only PHS
                # Interior-Interior case
                is_boundary = [false, false]
                bcs = [RBF.Dirichlet(), RBF.Dirichlet()]
                normals = [[0.0], [0.0]]
                hermite_data = RBF.HermiteStencilData(data_test, is_boundary, bcs, normals)

                @test_nowarn entry = RBF._hermite_rbf_entry(1, 2, hermite_data, basis)
                entry = RBF._hermite_rbf_entry(1, 2, hermite_data, basis)
                @test isfinite(entry)

                # Should match standard basis evaluation for interior points
                expected = basis(data_test[1], data_test[2])
                @test entry ≈ expected
            end
        end

        @testset "Polynomial Entry Computation" begin
            # Test polynomial entry computation
            mon = MonomialBasis(1, 1)
            data_test = [[0.5]]
            nmon = 2

            # Interior point case
            is_boundary = [false]
            bcs = [RBF.Dirichlet()]
            normals = [[0.0]]
            hermite_data = RBF.HermiteStencilData(data_test, is_boundary, bcs, normals)

            a = zeros(Float64, nmon)
            @test_nowarn RBF._hermite_poly_entry!(a, 1, hermite_data, mon)
            @test all(isfinite.(a))

            # Should match standard monomial evaluation for interior points
            expected = zeros(Float64, nmon)
            mon(expected, data_test[1])
            @test a ≈ expected
        end
    end

    @testset "Operator Limitations" begin
        @testset "Required Operator Availability" begin
            # Document which operators are available for each basis

            # All bases should have basic operators
            for basis in all_bases
                @test hasmethod(RBF.∇, (typeof(basis),))
                @test hasmethod(RBF.∂, (typeof(basis), Int))
                @test hasmethod(RBF.∂², (typeof(basis), Int))
            end

            # Only PHS has advanced operators needed for Hermite
            @test hasmethod(
                RBF.directional∂², (typeof(basis_phs), AbstractVector, AbstractVector)
            )
            @test !hasmethod(
                RBF.directional∂², (typeof(basis_imq), AbstractVector, AbstractVector)
            )
            @test !hasmethod(
                RBF.directional∂², (typeof(basis_gaussian), AbstractVector, AbstractVector)
            )
        end

        @testset "Robin Boundary Limitation" begin
            # Demonstrate the specific limitation for Robin boundaries
            data_test = [[0.0], [1.0]]
            is_boundary = [true, true]
            bcs = [RBF.Robin(1.0, 1.0), RBF.Robin(1.0, 1.0)]
            normals = [[-1.0], [1.0]]

            # PHS: works
            hermite_data_phs = RBF.HermiteStencilData(data_test, is_boundary, bcs, normals)
            @test_nowarn RBF._hermite_rbf_entry(1, 2, hermite_data_phs, basis_phs)

            # IMQ/Gaussian: would fail
            hermite_data_imq = RBF.HermiteStencilData(data_test, is_boundary, bcs, normals)
            @test_throws MethodError RBF._hermite_rbf_entry(
                1, 2, hermite_data_imq, basis_imq
            )
        end
    end

    @testset "Integration with Standard Solve" begin
        @testset "Dispatch to Correct Implementation" begin
            # Test that the system correctly dispatches to Hermite vs standard implementations
            k = 3
            mon_1d = MonomialBasis(1, 1)
            nmon = 2
            n = k + nmon

            # Simple identity operators (no boundary complications)
            identity_rbf = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))
            identity_mon = RBF.Custom(mon -> (arr, x) -> mon(arr, x))

            for basis in hermite_compatible_bases  # Currently only PHS
                ℒrbf = identity_rbf(basis)
                ℒmon = identity_mon(mon_1d)

                # Standard data (Vector{Vector}) should use standard path
                A_std = Symmetric(zeros(Float64, n, n), :U)
                b_std = zeros(Float64, n, 1)
                @test_nowarn weights_std = RBF._build_stencil!(
                    A_std, b_std, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, mon_1d, k
                )

                # Hermite data should use Hermite path
                is_boundary = [false, false, false]  # All interior for simplicity
                bcs = [RBF.Internal(), RBF.Internal(), RBF.Internal()]  # Interior sentinel values
                normals = [[0.0], [0.0], [0.0]]
                hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

                A_herm = Symmetric(zeros(Float64, n, n), :U)
                b_herm = zeros(Float64, n, 1)
                @test_nowarn weights_herm = RBF._build_stencil!(
                    A_herm,
                    b_herm,
                    ℒrbf,
                    ℒmon,
                    hermite_data,
                    eval_point_1d,
                    basis,
                    mon_1d,
                    k,
                )

                # Both should work and give similar results for interior-only case
                weights_std = RBF._build_stencil!(
                    A_std, b_std, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, mon_1d, k
                )
                weights_herm = RBF._build_stencil!(
                    A_herm,
                    b_herm,
                    ℒrbf,
                    ℒmon,
                    hermite_data,
                    eval_point_1d,
                    basis,
                    mon_1d,
                    k,
                )

                @test size(weights_std) == size(weights_herm)
                @test all(isfinite.(weights_std))
                @test all(isfinite.(weights_herm))
                # Results should be close for interior-only case
                @test weights_std ≈ weights_herm rtol = 1e-10
            end
        end
    end

    @testset "Future Expansion Framework" begin
        @testset "Expandability Check" begin
            # Test framework that will automatically work when IMQ/Gaussian support is added

            function supports_hermite_boundaries(basis)
                return hasmethod(
                    RBF.directional∂², (typeof(basis), AbstractVector, AbstractVector)
                )
            end

            # Current state
            @test supports_hermite_boundaries(basis_phs) == true
            @test supports_hermite_boundaries(basis_imq) == false
            @test supports_hermite_boundaries(basis_gaussian) == false

            # Count how many bases support Hermite
            supported_count = sum(
                supports_hermite_boundaries.([basis_phs, basis_imq, basis_gaussian])
            )
            @test supported_count == 1  # Currently only PHS

            # When operators are implemented, this should increase:
            # supported_count should become 3
            # and hermite_compatible_bases should be expanded to [basis_phs, basis_imq, basis_gaussian]
        end

        @testset "Boundary Condition Complexity Assessment" begin
            # Test which boundary condition combinations work with current implementation
            simple_bcs = [RBF.Dirichlet(), RBF.Neumann(), RBF.Dirichlet()]
            complex_bcs = [RBF.Robin(1.0, 1.0), RBF.Robin(1.0, 1.0), RBF.Dirichlet()]

            data_test = [[0.0], [0.5], [1.0]]
            is_boundary = [true, true, false]
            normals = [[-1.0], [1.0], [0.0]]

            # Simple boundary conditions should work with PHS
            hermite_data_simple = RBF.HermiteStencilData(
                data_test, is_boundary, simple_bcs, normals
            )
            k = 3
            mon_1d = MonomialBasis(1, 1)
            n = k + 2
            A = Symmetric(zeros(Float64, n, n), :U)

            @test_nowarn RBF._build_collocation_matrix!(
                A, hermite_data_simple, basis_phs, mon_1d, k
            )

            # Complex boundary conditions (multiple Robin) need directional∂²
            hermite_data_complex = RBF.HermiteStencilData(
                data_test, is_boundary, complex_bcs, normals
            )
            A_complex = Symmetric(zeros(Float64, n, n), :U)

            # This should work for PHS but would fail for IMQ/Gaussian
            @test_nowarn RBF._build_collocation_matrix!(
                A_complex, hermite_data_complex, basis_phs, mon_1d, k
            )
        end
    end
end
