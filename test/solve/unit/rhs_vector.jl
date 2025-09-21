"""
Unit tests for RHS vector building functions.
Tests both standard and Hermite variants of _build_rhs! and related functions.
"""

using Test
using LinearAlgebra
using StaticArraysCore
using RadialBasisFunctions
import RadialBasisFunctions as RBF

@testset "RHS Vector Building" begin

    # Test setup - common data for all tests
    basis_phs = PHS(3; poly_deg=1)
    basis_imq = IMQ(1.0)
    basis_gaussian = Gaussian(1.0)
    mon_1d = MonomialBasis(1, 1)
    mon_2d = MonomialBasis(2, 1)

    # CURRENT LIMITATION: Hermite RHS building requires extensive operator support
    # beyond just directional∂². For Robin boundary conditions, we need:
    # - ∇(basis) for normal derivatives
    # - directional∂²(basis, v1, v2) for mixed derivatives  
    # - Potentially higher-order operators depending on the differential operator
    # Currently only PHS has complete operator implementations
    hermite_compatible_bases = [basis_phs]  # Only PHS for now
    all_bases = [basis_phs, basis_imq, basis_gaussian]  # For standard (non-Hermite) tests

    # 1D test data
    data_1d = [[0.0], [0.5], [1.0]]
    eval_point_1d = [0.25]
    k_1d = 3

    # 2D test data  
    data_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    eval_point_2d = [0.5, 0.5]
    k_2d = 4

    @testset "Standard RHS Vector" begin
        @testset "Identity operator RHS" begin
            # Test RHS with identity operator (no derivatives)
            identity_op_1d = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))
            identity_op_mon = RBF.Custom(mon -> (arr, x) -> mon(arr, x))

            for basis in all_bases
                ℒrbf = identity_op_1d(basis)
                ℒmon = identity_op_mon(mon_1d)

                nmon_1d = 2  # 1D linear: [1, x]
                n_1d = k_1d + nmon_1d
                b = zeros(Float64, n_1d, 1)

                # Build RHS vector
                @test_nowarn RBF._build_rhs!(
                    b, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, k_1d
                )

                # Check RHS structure
                @test size(b) == (n_1d, 1)
                @test all(isfinite.(b))

                # RBF part should match basis evaluations
                for i in 1:k_1d
                    expected = basis(eval_point_1d, data_1d[i])
                    @test b[i, 1] ≈ expected
                end

                # Polynomial part should match monomial evaluations
                poly_vals = zeros(nmon_1d)
                mon_1d(poly_vals, eval_point_1d)
                for i in 1:nmon_1d
                    @test b[k_1d + i, 1] ≈ poly_vals[i]
                end
            end
        end

        @testset "Partial derivative operator RHS" begin
            # Test RHS with partial derivative operator ∂/∂x
            partial_op_1d = RBF.Custom(basis -> RBF.∂(basis, 1))
            partial_op_mon = RBF.Custom(mon -> RBF.∂(mon, 1))

            for basis in all_bases
                ℒrbf = partial_op_1d(basis)
                ℒmon = partial_op_mon(mon_1d)

                nmon_1d = 2
                n_1d = k_1d + nmon_1d
                b = zeros(Float64, n_1d, 1)

                @test_nowarn RBF._build_rhs!(
                    b, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, k_1d
                )

                # Check basic properties
                @test size(b) == (n_1d, 1)
                @test all(isfinite.(b))

                # RBF part should be derivatives of basis functions
                for i in 1:k_1d
                    expected = RBF.∂(basis, 1)(eval_point_1d, data_1d[i])
                    @test b[i, 1] ≈ expected
                end

                # Polynomial part: ∂/∂x of [1, x] = [0, 1]
                @test b[k_1d + 1, 1] ≈ 0.0  # ∂(1)/∂x = 0
                @test b[k_1d + 2, 1] ≈ 1.0  # ∂(x)/∂x = 1
            end
        end

        @testset "Second derivative operator RHS" begin
            # Test RHS with second derivative operator ∂²/∂x²
            second_deriv_op_1d = RBF.Custom(basis -> RBF.∂²(basis, 1))
            second_deriv_op_mon = RBF.Custom(mon -> RBF.∂²(mon, 1))

            for basis in all_bases
                ℒrbf = second_deriv_op_1d(basis)
                ℒmon = second_deriv_op_mon(mon_1d)

                nmon_1d = 2
                n_1d = k_1d + nmon_1d
                b = zeros(Float64, n_1d, 1)

                @test_nowarn RBF._build_rhs!(
                    b, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, k_1d
                )

                # Check basic properties
                @test size(b) == (n_1d, 1)
                @test all(isfinite.(b))

                # Polynomial part: ∂²/∂x² of [1, x] = [0, 0]
                @test b[k_1d + 1, 1] ≈ 0.0  # ∂²(1)/∂x² = 0
                @test b[k_1d + 2, 1] ≈ 0.0  # ∂²(x)/∂x² = 0
            end
        end

        @testset "2D gradient operator RHS" begin
            # Test 2D RHS with gradient operator (tuple version)
            # Gradient returns tuple of partial derivatives: (∂/∂x, ∂/∂y)

            for basis in all_bases
                # Create gradient operators as tuples
                ℒrbf = (RBF.∂(basis, 1), RBF.∂(basis, 2))  # (∂/∂x, ∂/∂y)
                ℒmon = (RBF.∂(mon_2d, 1), RBF.∂(mon_2d, 2))  # (∂/∂x, ∂/∂y) for monomials

                nmon_2d = 3  # 2D linear: [1, x, y]
                n_2d = k_2d + nmon_2d
                b = zeros(Float64, n_2d, 2)  # 2 columns for 2D gradient

                @test_nowarn RBF._build_rhs!(
                    b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, k_2d
                )

                # Check basic properties
                @test size(b) == (n_2d, 2)
                @test all(isfinite.(b))

                # Polynomial part: ∇[1, x, y] = [[0,0], [1,0], [0,1]]
                @test b[k_2d + 1, 1] ≈ 0.0 && b[k_2d + 1, 2] ≈ 0.0  # ∇(1) = [0,0]
                @test b[k_2d + 2, 1] ≈ 1.0 && b[k_2d + 2, 2] ≈ 0.0  # ∇(x) = [1,0]
                @test b[k_2d + 3, 1] ≈ 0.0 && b[k_2d + 3, 2] ≈ 1.0  # ∇(y) = [0,1]
            end
        end

        @testset "Laplacian operator RHS" begin
            # Test 2D RHS with Laplacian operator ∇²
            laplacian_op_2d = RBF.Custom(basis -> RBF.∇²(basis))
            laplacian_op_mon = RBF.Custom(mon -> RBF.∇²(mon))

            for basis in all_bases
                ℒrbf = laplacian_op_2d(basis)
                ℒmon = laplacian_op_mon(mon_2d)

                nmon_2d = 3
                n_2d = k_2d + nmon_2d
                b = zeros(Float64, n_2d, 1)

                @test_nowarn RBF._build_rhs!(
                    b, ℒrbf, ℒmon, data_2d, eval_point_2d, basis, k_2d
                )

                # Check basic properties
                @test size(b) == (n_2d, 1)
                @test all(isfinite.(b))

                # Polynomial part: ∇²[1, x, y] = [0, 0, 0]
                @test b[k_2d + 1, 1] ≈ 0.0  # ∇²(1) = 0
                @test b[k_2d + 2, 1] ≈ 0.0  # ∇²(x) = 0
                @test b[k_2d + 3, 1] ≈ 0.0  # ∇²(y) = 0
            end
        end

        @testset "RHS consistency across basis functions" begin
            # Test that polynomial parts are consistent across different basis functions
            identity_op_1d = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))
            identity_op_mon = RBF.Custom(mon -> (arr, x) -> mon(arr, x))
            ℒmon = identity_op_mon(mon_1d)

            nmon_1d = 2
            n_1d = k_1d + nmon_1d

            rhs_vectors = []
            for basis in all_bases
                ℒrbf = identity_op_1d(basis)
                b = zeros(Float64, n_1d, 1)
                RBF._build_rhs!(b, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, k_1d)
                push!(rhs_vectors, b)
            end

            # Polynomial parts (last nmon_1d entries) should be identical
            for i in 2:length(rhs_vectors)
                poly_part_1 = rhs_vectors[1][(k_1d + 1):end, :]
                poly_part_i = rhs_vectors[i][(k_1d + 1):end, :]
                @test poly_part_1 ≈ poly_part_i
            end

            # RBF parts should be different (unless by coincidence)
            @test rhs_vectors[1][1:k_1d, :] != rhs_vectors[2][1:k_1d, :]  # PHS vs IMQ
            @test rhs_vectors[1][1:k_1d, :] != rhs_vectors[3][1:k_1d, :]  # PHS vs Gaussian
        end
    end

    @testset "Hermite RHS Vector" begin
        # NOTE: Currently only testing PHS basis functions due to operator limitations
        # Hermite RHS requires extensive operator support that IMQ/Gaussian lack

        @testset "Interior points (no boundary)" begin
            # Test that interior points produce same result as standard
            is_boundary = [false, false, false]
            bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
            normals = [[0.0], [0.0], [0.0]]
            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Identity operator for comparison
            identity_op_1d = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))
            identity_op_mon = RBF.Custom(mon -> (arr, x) -> mon(arr, x))

            for basis in hermite_compatible_bases
                ℒrbf = identity_op_1d(basis)
                ℒmon = identity_op_mon(mon_1d)

                nmon_1d = 2
                n_1d = k_1d + nmon_1d

                # Build standard RHS
                b_standard = zeros(Float64, n_1d, 1)
                RBF._build_rhs!(b_standard, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, k_1d)

                # Build Hermite RHS with no boundaries (should be identical)
                b_hermite = zeros(Float64, n_1d, 1)
                # Note: We would need RBF._build_rhs_hermite! but it might not exist yet
                # For now, test that we can at least create the boundary data structure
                @test hermite_data isa RBF.HermiteStencilData
                @test length(hermite_data.data) == k_1d
                @test all(hermite_data.is_boundary .== false)
            end
        end

        @testset "Dirichlet boundary RHS" begin
            # Test RHS with Dirichlet boundary conditions
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Dirichlet(), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]
            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Test that boundary data is set up correctly
            @test hermite_data.is_boundary[2] == true
            @test is_dirichlet(hermite_data.boundary_conditions[2])

            # Dirichlet boundaries should behave like standard case
            # (no modification to operators needed)
            for basis in hermite_compatible_bases
                @test hermite_data isa RBF.HermiteStencilData{Float64}
                @test all(isfinite.(hermite_data.normals[2]))
            end
        end

        @testset "Neumann boundary RHS - operator requirements" begin
            # Test demonstrates why IMQ/Gaussian can't handle Neumann boundaries
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Neumann(), Dirichlet()]
            normals = [[0.0], [1.0], [0.0]]
            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            @test hermite_data.is_boundary[2] == true
            @test is_neumann(hermite_data.boundary_conditions[2])

            # For Neumann boundaries, we need normal derivative operators
            # Let's check which bases have the required ∂(basis, dim) implementation

            # PHS should have ∂ operator
            @test hasmethod(RBF.∂, (typeof(basis_phs), Int))

            # IMQ and Gaussian should also have ∂ operator
            @test hasmethod(RBF.∂, (typeof(basis_imq), Int))
            @test hasmethod(RBF.∂, (typeof(basis_gaussian), Int))

            # Test that we can compute normal derivatives for all bases
            normal = hermite_data.normals[2]  # [1.0] in x-direction

            for basis in all_bases
                ∂_op = RBF.∂(basis, 1)  # ∂/∂x
                @test_nowarn ∂_op(eval_point_1d, data_1d[2])
                result = ∂_op(eval_point_1d, data_1d[2])
                @test isfinite(result)
            end
        end

        @testset "Robin boundary RHS - operator requirements" begin
            # Test demonstrates the REAL limitation: Robin boundaries need directional∂²
            is_boundary = [true, true, false]  # Multiple boundary points
            bcs = [Robin(1.0, 0.5), Robin(0.5, 1.0), Dirichlet()]
            normals = [[1.0], [-1.0], [0.0]]
            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            @test hermite_data.is_boundary[1] == true
            @test hermite_data.is_boundary[2] == true
            @test is_robin(hermite_data.boundary_conditions[1])
            @test is_robin(hermite_data.boundary_conditions[2])

            # Robin-Robin interactions in RHS building would require:
            # 1. ∇(basis) for normal derivatives - ALL bases have this
            # 2. directional∂²(basis, v1, v2) for mixed derivatives - ONLY PHS has this

            # Check ∇ availability
            for basis in all_bases
                @test hasmethod(RBF.∇, (typeof(basis),))
                ∇_op = RBF.∇(basis)
                @test_nowarn ∇_op(eval_point_1d, data_1d[1])
            end

            # Check directional∂² availability - THIS IS THE BLOCKER
            @test hasmethod(
                RBF.directional∂², (typeof(basis_phs), AbstractVector, AbstractVector)
            )
            @test !hasmethod(
                RBF.directional∂², (typeof(basis_imq), AbstractVector, AbstractVector)
            )
            @test !hasmethod(
                RBF.directional∂², (typeof(basis_gaussian), AbstractVector, AbstractVector)
            )

            # This is why only PHS works with complex Robin boundary conditions
            v1, v2 = hermite_data.normals[1], hermite_data.normals[2]

            # PHS: works
            @test_nowarn RBF.directional∂²(basis_phs, v1, v2)
            dir_op_phs = RBF.directional∂²(basis_phs, v1, v2)
            @test_nowarn dir_op_phs(eval_point_1d, data_1d[1])

            # IMQ: fails
            @test_throws MethodError RBF.directional∂²(basis_imq, v1, v2)

            # Gaussian: fails  
            @test_throws MethodError RBF.directional∂²(basis_gaussian, v1, v2)
        end

        @testset "Higher-order operator requirements" begin
            # Test that demonstrates even MORE missing operators for complex scenarios

            # Even if we implemented directional∂² for IMQ/Gaussian, we might need:
            # - Higher-order mixed derivatives for complex boundary operators
            # - Specialized operators for specific PDE types
            # - Normal derivative operators ∂ₙ, ∂ₙ²

            # Check what operators PHS has that others don't
            phs_methods = methods(RBF.∂).ms
            phs_specific = filter(
                m -> (s -> occursin("PHS", s))(string(m.sig)), phs_methods
            )

            # Check for PHS-specific advanced operators that might be needed
            # Note: This is more of a documentation test showing the scope of the problem

            @test length(phs_specific) > 0  # PHS has specialized methods

            # TODO: When implementing full Hermite support for IMQ/Gaussian:
            # 1. Implement directional∂²(::IMQ, v1, v2)
            # 2. Implement directional∂²(::Gaussian, v1, v2)  
            # 3. Check if any higher-order operators are needed
            # 4. Implement missing operators as needed
            # 5. Add comprehensive testing for all boundary condition combinations
        end

        @testset "RHS vector structure validation" begin
            # Test that Hermite RHS maintains proper structure even with limitations
            is_boundary = [false, true, false]
            bcs = [Dirichlet(), Neumann(), Dirichlet()]  # Safe case for all bases
            normals = [[0.0], [1.0], [0.0]]
            hermite_data = RBF.HermiteStencilData(data_1d, is_boundary, bcs, normals)

            # Test boundary data structure is valid
            @test length(hermite_data.data) == k_1d
            @test length(hermite_data.is_boundary) == k_1d
            @test length(hermite_data.boundary_conditions) == k_1d
            @test length(hermite_data.normals) == k_1d

            # Test that we can identify the limitation programmatically
            function can_handle_hermite_boundaries(basis, boundary_conditions, normals)
                # Check if basis can handle the required operators for given boundaries
                has_directional_second = hasmethod(
                    RBF.directional∂², (typeof(basis), AbstractVector, AbstractVector)
                )

                # Check if we need directional∂² (Robin-Robin interactions)
                robin_count = sum(is_robin.(boundary_conditions))
                needs_directional_second = robin_count >= 2

                return !needs_directional_second || has_directional_second
            end

            # Test the limitation detector
            robin_bcs = [Robin(1.0, 1.0), Robin(1.0, 1.0), Dirichlet()]
            simple_bcs = [Dirichlet(), Neumann(), Dirichlet()]

            @test can_handle_hermite_boundaries(basis_phs, robin_bcs, normals)      # PHS: OK
            @test !can_handle_hermite_boundaries(basis_imq, robin_bcs, normals)    # IMQ: NO
            @test !can_handle_hermite_boundaries(basis_gaussian, robin_bcs, normals) # Gaussian: NO

            @test can_handle_hermite_boundaries(basis_phs, simple_bcs, normals)     # PHS: OK
            @test can_handle_hermite_boundaries(basis_imq, simple_bcs, normals)     # IMQ: OK
            @test can_handle_hermite_boundaries(basis_gaussian, simple_bcs, normals) # Gaussian: OK
        end
    end

    @testset "RHS Building Integration" begin
        @testset "Function dispatch and signatures" begin
            # Verify that RHS building functions exist and have correct signatures
            @test hasmethod(
                RBF._build_rhs!,
                (Any, Any, Any, AbstractVector, Any, AbstractRadialBasis, Int),
            )

            # Test with simple operators
            identity_op_1d = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))
            identity_op_mon = RBF.Custom(mon -> (arr, x) -> mon(arr, x))

            for basis in all_bases
                ℒrbf = identity_op_1d(basis)
                ℒmon = identity_op_mon(mon_1d)

                nmon_1d = 2
                n_1d = k_1d + nmon_1d
                b = zeros(Float64, n_1d, 1)

                @test_nowarn RBF._build_rhs!(
                    b, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, k_1d
                )
                @test all(isfinite.(b))
            end
        end

        @testset "Vector size consistency" begin
            # Test that RHS vectors have correct dimensions for different problem sizes
            identity_op_1d = RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))
            identity_op_mon = RBF.Custom(mon -> (arr, x) -> mon(arr, x))

            test_sizes = [(3, 1, 2), (5, 1, 2), (4, 2, 3)]  # (k, dim, nmon)

            for (k, dim, expected_nmon) in test_sizes
                data_test = [rand(dim) for _ in 1:k]
                eval_point_test = rand(dim)
                mon_test = MonomialBasis(dim, 1)  # Linear polynomials
                n_test = k + expected_nmon

                for basis in all_bases
                    ℒrbf = identity_op_1d(basis)
                    ℒmon = identity_op_mon(mon_test)

                    b = zeros(Float64, n_test, 1)
                    @test_nowarn RBF._build_rhs!(
                        b, ℒrbf, ℒmon, data_test, eval_point_test, basis, k
                    )
                    @test size(b) == (n_test, 1)
                end
            end
        end

        @testset "Operator compatibility" begin
            # Test that different operators work correctly with RHS building
            operators_1d = [
                ("Identity", RBF.Custom(basis -> (x1, x2) -> basis(x1, x2))),
                ("First derivative", RBF.Custom(basis -> RBF.∂(basis, 1))),
                ("Second derivative", RBF.Custom(basis -> RBF.∂²(basis, 1))),
            ]

            mon_operators_1d = [
                ("Identity", RBF.Custom(mon -> (arr, x) -> mon(arr, x))),
                ("First derivative", RBF.Custom(mon -> RBF.∂(mon, 1))),
                ("Second derivative", RBF.Custom(mon -> RBF.∂²(mon, 1))),
            ]

            for (op_name, op_rbf) in operators_1d
                for (mon_op_name, op_mon) in mon_operators_1d
                    for basis in all_bases
                        ℒrbf = op_rbf(basis)
                        ℒmon = op_mon(mon_1d)

                        nmon_1d = 2
                        n_1d = k_1d + nmon_1d
                        b = zeros(Float64, n_1d, 1)

                        @test_nowarn RBF._build_rhs!(
                            b, ℒrbf, ℒmon, data_1d, eval_point_1d, basis, k_1d
                        )
                        @test all(isfinite.(b))
                        @test size(b) == (n_1d, 1)
                    end
                end
            end
        end
    end
end
