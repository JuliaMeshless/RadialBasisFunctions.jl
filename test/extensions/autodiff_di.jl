using RadialBasisFunctions
using StaticArraysCore
using FiniteDifferences
using LinearAlgebra
using Test
import DifferentiationInterface as DI
using Enzyme: Enzyme
using Mooncake: Mooncake

const FD = FiniteDifferences

# Version compatibility - Enzyme.jl has known issues with Julia 1.12+
# See: https://github.com/EnzymeAD/Enzyme.jl/issues/2699
const ENZYME_SUPPORTED = VERSION < v"1.12"

# Backend configuration
const ENZYME_BACKEND = DI.AutoEnzyme(; function_annotation = Enzyme.Const)
const MOONCAKE_BACKEND = DI.AutoMooncake(; config = nothing)

# Build backend registry (only include supported backends)
const AD_BACKENDS = Pair{String, Any}[]
ENZYME_SUPPORTED && push!(AD_BACKENDS, "Enzyme" => ENZYME_BACKEND)
push!(AD_BACKENDS, "Mooncake" => MOONCAKE_BACKEND)

"""
    test_gradient_vs_fd(f, x, backend; rtol=1e-4, name="")

Test that DI.gradient matches finite differences for function f at point x.
"""
function test_gradient_vs_fd(f, x, backend; rtol = 1.0e-4, name = "")
    di_grad = DI.gradient(f, backend, x)
    fd_grad = FD.grad(FD.central_fdm(5, 1), f, x)[1]
    @test !all(iszero, di_grad)
    return @test isapprox(di_grad, fd_grad; rtol = rtol)
end

@testset "Autodiff via DifferentiationInterface" begin
    @testset "Operator Differentiation" begin
        N = 50
        points = [
            SVector{2}(0.1 + 0.8 * i / N, 0.1 + 0.8 * j / N) for i in 1:isqrt(N)
                for j in 1:isqrt(N)
        ]
        N = length(points)
        values = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))

        @testset "Laplacian Operator" begin
            lap = laplacian(points)
            loss_lap(v) = sum(lap(v) .^ 2)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(loss_lap, values, backend; rtol = 1.0e-4)
                end
            end
        end

        @testset "Gradient Operator" begin
            grad_op = RadialBasisFunctions.gradient(points)
            loss_grad(v) = sum(grad_op(v) .^ 2)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    # Vector-valued operators have known issues on Julia 1.12+ with Enzyme
                    if name == "Enzyme" && VERSION >= v"1.12"
                        @test_skip "Vector-valued operators have known issues on Julia 1.12+"
                    else
                        test_gradient_vs_fd(loss_grad, values, backend; rtol = 1.0e-4)
                    end
                end
            end
        end

        @testset "Partial Derivative Operator" begin
            partial_x = partial(points, 1, 1)
            loss_partial(v) = sum(partial_x(v) .^ 2)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(loss_partial, values, backend; rtol = 1.0e-4)
                end
            end
        end
    end

    @testset "Interpolator Differentiation" begin
        N = 30
        points = [SVector{2}(rand(), rand()) for _ in 1:N]
        values = sin.(getindex.(points, 1))
        eval_points = [SVector{2}(rand(), rand()) for _ in 1:10]

        @testset "Construction w.r.t. values (PHS default)" begin
            function loss_interp(v)
                interp_local = Interpolator(points, v)
                return sum(interp_local(eval_points) .^ 2)
            end

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    if name == "Enzyme"
                        @test_skip "Enzyme cannot differentiate through Interpolator constructor (factorize Union types)"
                    else
                        test_gradient_vs_fd(loss_interp, values, backend; rtol = 1.0e-3)
                    end
                end
            end
        end

        @testset "Construction with IMQ basis" begin
            function loss_interp_imq(v)
                interp_local = Interpolator(points, v, IMQ(1.0))
                return sum(interp_local(eval_points) .^ 2)
            end

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    if name == "Enzyme"
                        @test_skip "Enzyme cannot differentiate through Interpolator constructor (factorize Union types)"
                    else
                        test_gradient_vs_fd(loss_interp_imq, values, backend; rtol = 1.0e-3)
                    end
                end
            end
        end

        @testset "Construction with Gaussian basis" begin
            function loss_interp_gauss(v)
                interp_local = Interpolator(points, v, Gaussian(1.0))
                return sum(interp_local(eval_points) .^ 2)
            end

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    if name == "Enzyme"
                        @test_skip "Enzyme cannot differentiate through Interpolator constructor (factorize Union types)"
                    else
                        test_gradient_vs_fd(loss_interp_gauss, values, backend; rtol = 1.0e-3)
                    end
                end
            end
        end

        @testset "Single point evaluation" begin
            single_eval_point = SVector{2}(0.5, 0.5)

            function loss_interp_single(v)
                interp_local = Interpolator(points, v)
                return interp_local(single_eval_point)^2
            end

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    if name == "Enzyme"
                        @test_skip "Enzyme cannot differentiate through Interpolator constructor (factorize Union types)"
                    else
                        test_gradient_vs_fd(loss_interp_single, values, backend; rtol = 1.0e-3)
                    end
                end
            end
        end
    end

    @testset "Basis Function Differentiation" begin
        x = [0.5, 0.5]
        xi = [0.3, 0.4]

        @testset "PHS Basis Functions" begin
            for (order, phs_type) in [(1, PHS(1)), (3, PHS(3)), (5, PHS(5)), (7, PHS(7))]
                loss_phs(xv) = phs_type(xv, xi)^2

                for (name, backend) in AD_BACKENDS
                    @testset "PHS($order) - $name" begin
                        test_gradient_vs_fd(loss_phs, x, backend; rtol = 1.0e-4)
                    end
                end
            end
        end

        @testset "IMQ Basis Function" begin
            imq = IMQ(1.0)
            loss_imq(xv) = imq(xv, xi)^2

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(loss_imq, x, backend; rtol = 1.0e-4)
                end
            end
        end

        @testset "Gaussian Basis Function" begin
            gauss = Gaussian(1.0)
            loss_gauss(xv) = gauss(xv, xi)^2

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(loss_gauss, x, backend; rtol = 1.0e-4)
                end
            end
        end
    end

    @testset "_build_weights Differentiation" begin
        N = 25
        points = [SVector{2}(0.1 + 0.8 * i / 5, 0.1 + 0.8 * j / 5) for i in 1:5 for j in 1:5]
        adjl = RadialBasisFunctions.find_neighbors(points, 10)

        @testset "Partial operator with PHS3" begin
            basis = PHS(3; poly_deg = 2)
            ℒ = Partial(1, 1)

            function loss_partial_weights(pts)
                pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            pts_flat = vcat([collect(p) for p in points]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(loss_partial_weights, pts_flat, backend; rtol = 1.0e-3)
                end
            end
        end

        @testset "Laplacian operator with PHS3" begin
            basis = PHS(3; poly_deg = 2)
            ℒ = Laplacian()

            function loss_laplacian_weights(pts)
                pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            pts_flat = vcat([collect(p) for p in points]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(
                        loss_laplacian_weights, pts_flat, backend; rtol = 1.0e-3
                    )
                end
            end
        end

        @testset "1D Partial operator with PHS3" begin
            N_1d = 10
            points_1d = [SVector{1}(0.1 + 0.8 * i / N_1d) for i in 1:N_1d]
            adjl_1d = RadialBasisFunctions.find_neighbors(points_1d, 5)
            basis_1d = PHS(3; poly_deg = 2)
            ℒ_1d = Partial(1, 1)

            function loss_partial_weights_1d(pts)
                pts_vec = [SVector{1}(pts[i]) for i in 1:N_1d]
                W = RadialBasisFunctions._build_weights(
                    ℒ_1d, pts_vec, pts_vec, adjl_1d, basis_1d
                )
                return sum(W.nzval .^ 2)
            end

            pts_flat_1d = vcat([collect(p) for p in points_1d]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(
                        loss_partial_weights_1d, pts_flat_1d, backend; rtol = 1.0e-3
                    )
                end
            end
        end

        @testset "3D Partial operator with PHS3" begin
            # Use Halton-like sequence to avoid singular stencils on regular grids
            N_3d = 64
            points_3d = [
                SVector{3}(
                    0.1 + 0.8 * ((i * 7 + 3) % N_3d) / N_3d,
                    0.1 + 0.8 * ((i * 11 + 5) % N_3d) / N_3d,
                    0.1 + 0.8 * ((i * 13 + 7) % N_3d) / N_3d,
                ) for i in 1:N_3d
            ]
            adjl_3d = RadialBasisFunctions.find_neighbors(points_3d, 20)
            basis_3d = PHS(3; poly_deg = 2)
            ℒ_3d = Partial(1, 1)

            function loss_partial_weights_3d(pts)
                pts_vec = [
                    SVector{3}(pts[3 * i - 2], pts[3 * i - 1], pts[3 * i]) for i in 1:N_3d
                ]
                W = RadialBasisFunctions._build_weights(
                    ℒ_3d, pts_vec, pts_vec, adjl_3d, basis_3d
                )
                return sum(W.nzval .^ 2)
            end

            pts_flat_3d = vcat([collect(p) for p in points_3d]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(
                        loss_partial_weights_3d, pts_flat_3d, backend; rtol = 1.0e-3
                    )
                end
            end
        end

        @testset "2D Partial(1,2) operator with PHS3" begin
            basis = PHS(3; poly_deg = 2)
            ℒ_y = Partial(1, 2)

            function loss_partial_y_weights(pts)
                pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                W = RadialBasisFunctions._build_weights(ℒ_y, pts_vec, pts_vec, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            pts_flat = vcat([collect(p) for p in points]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(
                        loss_partial_y_weights, pts_flat, backend; rtol = 1.0e-3
                    )
                end
            end
        end

        @testset "Different PHS orders" begin
            for n in [1, 3, 5, 7]
                basis = PHS(n; poly_deg = 1)
                ℒ = Partial(1, 1)

                function loss_phs_order(pts)
                    pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                    W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                    return sum(W.nzval .^ 2)
                end

                pts_flat = vcat([collect(p) for p in points]...)

                for (name, backend) in AD_BACKENDS
                    @testset "PHS($n) - $name" begin
                        di_grad = DI.gradient(loss_phs_order, backend, pts_flat)
                        fd_grad = FD.grad(FD.central_fdm(5, 1), loss_phs_order, pts_flat)[1]
                        # PHS1 may have zero gradient for some configurations
                        @test !all(iszero, di_grad) || n == 1
                        @test isapprox(di_grad, fd_grad; rtol = 1.0e-2)
                    end
                end
            end
        end

        @testset "IMQ basis with Partial operator" begin
            basis = IMQ(1.0; poly_deg = 2)
            ℒ = Partial(1, 1)

            function loss_imq_partial(pts)
                pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            pts_flat = vcat([collect(p) for p in points]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(loss_imq_partial, pts_flat, backend; rtol = 1.0e-3)
                end
            end
        end

        @testset "IMQ basis with Laplacian operator" begin
            basis = IMQ(1.0; poly_deg = 2)
            ℒ = Laplacian()

            function loss_imq_laplacian(pts)
                pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            pts_flat = vcat([collect(p) for p in points]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(loss_imq_laplacian, pts_flat, backend; rtol = 1.0e-3)
                end
            end
        end

        @testset "Gaussian basis with Partial operator" begin
            basis = Gaussian(1.0; poly_deg = 2)
            ℒ = Partial(1, 1)

            function loss_gaussian_partial(pts)
                pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            pts_flat = vcat([collect(p) for p in points]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(
                        loss_gaussian_partial, pts_flat, backend; rtol = 1.0e-3
                    )
                end
            end
        end

        @testset "Gaussian basis with Laplacian operator" begin
            basis = Gaussian(1.0; poly_deg = 2)
            ℒ = Laplacian()

            function loss_gaussian_laplacian(pts)
                pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            pts_flat = vcat([collect(p) for p in points]...)

            for (name, backend) in AD_BACKENDS
                @testset "$name" begin
                    test_gradient_vs_fd(
                        loss_gaussian_laplacian, pts_flat, backend; rtol = 1.0e-3
                    )
                end
            end
        end

        @testset "Different shape parameters" begin
            for ε in [0.5, 1.0, 2.0]
                @testset "IMQ with ε=$ε" begin
                    basis = IMQ(ε; poly_deg = 2)
                    ℒ = Partial(1, 1)

                    function loss_imq_shape(pts)
                        pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                        W = RadialBasisFunctions._build_weights(
                            ℒ, pts_vec, pts_vec, adjl, basis
                        )
                        return sum(W.nzval .^ 2)
                    end

                    pts_flat = vcat([collect(p) for p in points]...)

                    for (name, backend) in AD_BACKENDS
                        @testset "$name" begin
                            test_gradient_vs_fd(
                                loss_imq_shape, pts_flat, backend; rtol = 1.0e-2
                            )
                        end
                    end
                end

                @testset "Gaussian with ε=$ε" begin
                    basis = Gaussian(ε; poly_deg = 2)
                    ℒ = Partial(1, 1)

                    function loss_gaussian_shape(pts)
                        pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                        W = RadialBasisFunctions._build_weights(
                            ℒ, pts_vec, pts_vec, adjl, basis
                        )
                        return sum(W.nzval .^ 2)
                    end

                    pts_flat = vcat([collect(p) for p in points]...)

                    for (name, backend) in AD_BACKENDS
                        @testset "$name" begin
                            test_gradient_vs_fd(
                                loss_gaussian_shape, pts_flat, backend; rtol = 1.0e-2
                            )
                        end
                    end
                end
            end
        end
    end

    # Test that extensions load correctly
    @testset "Extension Loading" begin
        @test Base.find_package("Enzyme") !== nothing
        @test Base.find_package("Mooncake") !== nothing
        @test Base.find_package("DifferentiationInterface") !== nothing
    end
end
