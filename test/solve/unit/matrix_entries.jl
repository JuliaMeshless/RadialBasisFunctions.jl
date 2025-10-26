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

    # Basis functions
    basis_phs = PHS(3; poly_deg=1)
    basis_imq = IMQ(1.0)
    mon_1d = MonomialBasis(1, 1)
    mon_2d = MonomialBasis(2, 1)

    # Point data
    data_1d = [[0.0], [0.5], [1.0]]
    data_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    """Create Hermite data with specified boundary conditions at given indices."""
    function make_hermite_data(data, boundary_indices, boundary_conditions, normals)
        n = length(data)
        is_boundary = fill(false, n)
        T = eltype(data[1])
        bcs = BoundaryCondition{T}[Internal() for _ in 1:n]
        all_normals = [zeros(T, length(data[1])) for _ in 1:n]

        for (idx, bc, normal) in zip(boundary_indices, boundary_conditions, normals)
            is_boundary[idx] = true
            bcs[idx] = bc
            all_normals[idx] = normal
        end

        return RBF.HermiteStencilData(data, is_boundary, bcs, all_normals)
    end

    """Test that Hermite entry matches standard RBF evaluation."""
    function test_matches_standard(i, j, hermite_data, basis, data)
        val_hermite = RBF._hermite_rbf_entry(i, j, hermite_data, basis)
        val_standard = basis(data[i], data[j])
        @test val_hermite ≈ val_standard
    end

    """Test that Hermite entry differs from standard RBF evaluation."""
    function test_differs_from_standard(i, j, hermite_data, basis, data)
        val_hermite = RBF._hermite_rbf_entry(i, j, hermite_data, basis)
        val_standard = basis(data[i], data[j])
        @test val_hermite != val_standard
        @test isfinite(val_hermite)
    end

    """Test Hermite entry against manually computed expected value."""
    function test_hermite_value(hermite_data, expected_value; rtol=1e-10)
        phs3 = PHS3(; poly_deg=-1)
        actual = RBF._hermite_rbf_entry(1, 2, hermite_data, phs3)
        @test actual ≈ expected_value rtol = rtol
    end

    @testset "Hermite RBF Matrix Entries" begin
        @testset "Interior-Interior entries" begin
            # All interior points - should match standard RBF evaluation
            hermite_data = make_hermite_data(
                data_1d, Int[], BoundaryCondition[], Vector{Float64}[]
            )
            test_matches_standard(1, 2, hermite_data, basis_phs, data_1d)
        end

        @testset "Interior-Boundary entries" begin
            @testset "Interior-Dirichlet" begin
                # Dirichlet BC at point 2: should match standard φ(x₁, x₂)
                hermite_data = make_hermite_data(data_1d, [2], [Dirichlet()], [[1.0]])
                test_matches_standard(1, 2, hermite_data, basis_phs, data_1d)
            end

            @testset "Interior-Neumann" begin
                # Neumann BC at point 2: α*φ + β*∂ₙφ = 0*φ + 1*∂ₙφ = ∂ₙφ
                hermite_data = make_hermite_data(data_1d, [2], [Neumann()], [[1.0]])
                test_differs_from_standard(1, 2, hermite_data, basis_phs, data_1d)
            end

            @testset "Interior-Robin" begin
                # Robin BC at point 2: α*φ + β*∂ₙφ with α=0.5, β=2.0
                hermite_data = make_hermite_data(data_1d, [2], [Robin(0.5, 2.0)], [[1.0]])
                test_differs_from_standard(1, 2, hermite_data, basis_phs, data_1d)

                # Test coefficient sensitivity - different α, β should give different results
                hermite_data_diff = make_hermite_data(
                    data_1d, [2], [Robin(1.0, 0.5)], [[1.0]]
                )
                val1 = RBF._hermite_rbf_entry(1, 2, hermite_data, basis_phs)
                val2 = RBF._hermite_rbf_entry(1, 2, hermite_data_diff, basis_phs)
                @test val1 != val2
            end
        end

        @testset "Boundary-Interior entries" begin
            @testset "Dirichlet-Interior" begin
                # Dirichlet BC at point 1: should match standard φ(x₁, x₂)
                hermite_data = make_hermite_data(data_1d, [1], [Dirichlet()], [[1.0]])
                test_matches_standard(1, 2, hermite_data, basis_phs, data_1d)
            end

            @testset "Neumann-Interior" begin
                # Neumann BC at point 1: should be ∂ₙφ(x₁, x₂)
                hermite_data = make_hermite_data(data_1d, [1], [Neumann()], [[1.0]])
                test_differs_from_standard(1, 2, hermite_data, basis_phs, data_1d)
            end
        end

        @testset "Boundary-Boundary entries (2D)" begin
            @testset "Dirichlet-Dirichlet" begin
                # Both points Dirichlet: should match standard φ(x₁, x₂)
                hermite_data = make_hermite_data(
                    data_2d, [1, 2], [Dirichlet(), Dirichlet()], [[1.0, 0.0], [0.0, 1.0]]
                )
                test_matches_standard(1, 2, hermite_data, basis_phs, data_2d)
            end

            @testset "Dirichlet-Neumann" begin
                # Point 1 Dirichlet, point 2 Neumann with y-direction normal
                hermite_data = make_hermite_data(
                    data_2d, [1, 2], [Dirichlet(), Neumann()], [[1.0, 0.0], [0.0, 1.0]]
                )
                test_differs_from_standard(1, 2, hermite_data, basis_phs, data_2d)
            end

            @testset "Dirichlet-Neumann (value verification)" begin
                # Formula simplifies to: ⟨n₂, -∇φ⟩
                x₁, x₂ = [0.0, 0.0], [0.5, 0.3]
                n₁, n₂ = [1.0, 0.0], [0.0, 1.0]

                hermite_data = make_hermite_data(
                    [x₁, x₂], [1, 2], [Dirichlet(), Neumann()], [n₁, n₂]
                )

                phs3 = PHS3(; poly_deg=-1)
                expected = dot(n₂, -RBF.∇(phs3)(x₁, x₂))
                test_hermite_value(hermite_data, expected)
            end

            @testset "Neumann-Dirichlet (value verification)" begin
                # Formula simplifies to: ⟨n₁, ∇φ⟩
                x₁, x₂ = [0.0, 0.0], [0.5, 0.3]
                n₁, n₂ = [1.0, 0.0], [0.0, 1.0]

                hermite_data = make_hermite_data(
                    [x₁, x₂], [1, 2], [Neumann(), Dirichlet()], [n₁, n₂]
                )

                phs3 = PHS3(; poly_deg=-1)
                expected = dot(n₁, RBF.∇(phs3)(x₁, x₂))
                test_hermite_value(hermite_data, expected)
            end

            @testset "Neumann-Neumann (value verification)" begin
                # Formula simplifies to: directional∂²(φ, n₁, n₂)
                x₁, x₂ = [0.0, 0.0], [0.5, 0.3]
                n₁, n₂ = [1.0, 0.0], [0.0, 1.0]

                hermite_data = make_hermite_data(
                    [x₁, x₂], [1, 2], [Neumann(), Neumann()], [n₁, n₂]
                )

                phs3 = PHS3(; poly_deg=-1)
                expected = RBF.directional∂²(phs3, n₁, n₂)(x₁, x₂)
                test_hermite_value(hermite_data, expected)
            end

            @testset "Robin-Robin (value verification)" begin
                # Full 4-term formula: α₁α₂φ + α₁β₂⟨n₂,-∇φ⟩ + β₁α₂⟨n₁,∇φ⟩ + β₁β₂·∂ₙ₁∂ₙ₂φ
                x₁, x₂ = [0.0, 0.0], [0.5, 0.3]
                n₁, n₂ = [1.0, 0.0], [0.0, 1.0]
                α₁, β₁ = 0.5, 1.0
                α₂, β₂ = 1.0, 0.5

                hermite_data = make_hermite_data(
                    [x₁, x₂], [1, 2], [Robin(α₁, β₁), Robin(α₂, β₂)], [n₁, n₂]
                )

                # Compute each term of the formula
                phs3 = PHS3(; poly_deg=-1)
                grad_val = RBF.∇(phs3)(x₁, x₂)

                expected = (
                    α₁ * α₂ * phs3(x₁, x₂) +                        # Term 1: α₁α₂φ
                    α₁ * β₂ * dot(n₂, -grad_val) +                  # Term 2: α₁β₂⟨n₂,-∇φ⟩
                    β₁ * α₂ * dot(n₁, grad_val) +                   # Term 3: β₁α₂⟨n₁,∇φ⟩
                    β₁ * β₂ * RBF.directional∂²(phs3, n₁, n₂)(x₁, x₂)  # Term 4: β₁β₂·∂ₙ₁∂ₙ₂φ
                )

                test_hermite_value(hermite_data, expected)
            end
        end
    end

    @testset "Hermite Polynomial Entries" begin
        @testset "Interior and Dirichlet" begin
            # Both interior and Dirichlet points use standard polynomial evaluation
            @testset "Interior polynomial entries" begin
                # Test a truly interior point (no boundary conditions)
                hermite_data = make_hermite_data(
                    data_1d, Int[], BoundaryCondition{Float64}[], Vector{Float64}[]
                )

                a_hermite = zeros(2)
                RBF._hermite_poly_entry!(a_hermite, 1, hermite_data, mon_1d)

                a_standard = zeros(2)
                mon_1d(a_standard, data_1d[1])

                @test a_hermite ≈ a_standard
            end

            @testset "Dirichlet polynomial entries" begin
                # Dirichlet BC should use standard polynomial evaluation
                hermite_data = make_hermite_data(data_1d, [1], [Dirichlet()], [[1.0]])

                a_hermite = zeros(2)
                RBF._hermite_poly_entry!(a_hermite, 1, hermite_data, mon_1d)

                a_standard = zeros(2)
                mon_1d(a_standard, data_1d[1])

                @test a_hermite ≈ a_standard
            end
        end

        @testset "Neumann polynomial entries (1D)" begin
            # Neumann BC: ∂ₙP. For 1D linear polynomials [1, x]: derivatives are [0, 1]
            hermite_data = make_hermite_data(data_1d, [1], [Neumann()], [[1.0]])

            a = zeros(2)
            RBF._hermite_poly_entry!(a, 1, hermite_data, mon_1d)

            @test a[1] ≈ 0.0   # ∂(1)/∂x = 0
            @test a[2] ≈ 1.0   # ∂(x)/∂x = 1
        end

        @testset "Robin polynomial entries (1D)" begin
            # Robin BC: α*P + β*∂ₙP. At x=0: P=[1,0], ∂P=[0,1]
            hermite_data = make_hermite_data(data_1d, [1], [Robin(0.5, 2.0)], [[1.0]])

            a = zeros(2)
            RBF._hermite_poly_entry!(a, 1, hermite_data, mon_1d)

            @test a[1] ≈ 0.5   # 0.5*1 + 2.0*0
            @test a[2] ≈ 2.0   # 0.5*0 + 2.0*1
        end

        @testset "Neumann polynomial entries (2D)" begin
            # 2D linear polynomials [1, x, y] with x-direction normal: ∂P/∂x = [0, 1, 0]
            hermite_data = make_hermite_data(data_2d, [1], [Neumann()], [[1.0, 0.0]])

            a = zeros(3)
            RBF._hermite_poly_entry!(a, 1, hermite_data, mon_2d)

            @test a[1] ≈ 0.0   # ∂(1)/∂x = 0
            @test a[2] ≈ 1.0   # ∂(x)/∂x = 1
            @test a[3] ≈ 0.0   # ∂(y)/∂x = 0
        end
    end
end
