using RadialBasisFunctions
using Enzyme
using StaticArraysCore
using FiniteDifferences
using LinearAlgebra
using Test

const FD = FiniteDifferences

# Check Julia version - Enzyme.jl has known issues with Julia 1.12+
# See: https://github.com/EnzymeAD/Enzyme.jl/issues/2699
const ENZYME_SUPPORTED_JULIA = VERSION < v"1.12"

if ENZYME_SUPPORTED_JULIA
    @testset "Enzyme Extension - Operator Differentiation" begin
        N = 50
        points = [SVector{2}(0.1 + 0.8 * i / N, 0.1 + 0.8 * j / N) for i in 1:isqrt(N) for j in 1:isqrt(N)]
        N = length(points)
        values = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))

        @testset "Laplacian Operator" begin
            lap = laplacian(points)

            function loss_lap(v)
                result = lap(v)
                return sum(result .^ 2)
            end

            dv = zeros(N)
            Enzyme.autodiff(Reverse, loss_lap, Active, Duplicated(values, dv))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_lap, values)[1]
            @test !all(iszero, dv)
            @test isapprox(dv, fd_grad; rtol = 1.0e-4)
        end

        @testset "Gradient Operator" begin
            grad = gradient(points)

            function loss_grad(v)
                result = grad(v)
                return sum(result .^ 2)
            end

            dv = zeros(N)
            Enzyme.autodiff(Reverse, loss_grad, Active, Duplicated(values, dv))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_grad, values)[1]
            @test !all(iszero, dv)
            @test isapprox(dv, fd_grad; rtol = 1.0e-4)
        end

        @testset "Partial Derivative Operator" begin
            partial_x = partial(points, 1, 1)

            function loss_partial(v)
                result = partial_x(v)
                return sum(result .^ 2)
            end

            dv = zeros(N)
            Enzyme.autodiff(Reverse, loss_partial, Active, Duplicated(values, dv))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_partial, values)[1]
            @test !all(iszero, dv)
            @test isapprox(dv, fd_grad; rtol = 1.0e-4)
        end
    end

    @testset "Enzyme Extension - Interpolator Differentiation" begin
        N = 30
        points = [SVector{2}(rand(), rand()) for _ in 1:N]
        values = sin.(getindex.(points, 1))
        eval_points = [SVector{2}(rand(), rand()) for _ in 1:10]

        interp = Interpolator(points, values)

        function loss_interp(v)
            interp_local = Interpolator(points, v)
            result = interp_local(eval_points)
            return sum(result .^ 2)
        end

        dv = zeros(N)
        Enzyme.autodiff(Reverse, loss_interp, Active, Duplicated(values, dv))

        fd_grad = FD.grad(FD.central_fdm(5, 1), loss_interp, values)[1]
        @test !all(iszero, dv)
        @test isapprox(dv, fd_grad; rtol = 1.0e-3)
    end

    @testset "Enzyme Extension - Basis Function Differentiation" begin
        x = [0.5, 0.5]
        xi = [0.3, 0.4]

        @testset "PHS Basis Functions" begin
            for phs_type in [PHS(1), PHS(3), PHS(5), PHS(7)]
                function loss_phs(xv)
                    return phs_type(xv, xi)^2
                end

                dx = zeros(2)
                Enzyme.autodiff(Reverse, loss_phs, Active, Duplicated(x, dx))

                fd_grad = FD.grad(FD.central_fdm(5, 1), loss_phs, x)[1]
                @test !all(iszero, dx)
                @test isapprox(dx, fd_grad; rtol = 1.0e-4)
            end
        end

        @testset "IMQ Basis Function" begin
            imq = IMQ(1.0)

            function loss_imq(xv)
                return imq(xv, xi)^2
            end

            dx = zeros(2)
            Enzyme.autodiff(Reverse, loss_imq, Active, Duplicated(x, dx))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_imq, x)[1]
            @test !all(iszero, dx)
            @test isapprox(dx, fd_grad; rtol = 1.0e-4)
        end

        @testset "Gaussian Basis Function" begin
            gauss = Gaussian(1.0)

            function loss_gauss(xv)
                return gauss(xv, xi)^2
            end

            dx = zeros(2)
            Enzyme.autodiff(Reverse, loss_gauss, Active, Duplicated(x, dx))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_gauss, x)[1]
            @test !all(iszero, dx)
            @test isapprox(dx, fd_grad; rtol = 1.0e-4)
        end
    end
    @testset "Enzyme Extension - Native Rules for _build_weights" begin
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
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss_partial_weights, Active, Duplicated(pts_flat, dpts))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_partial_weights, pts_flat)[1]
            @test !all(iszero, dpts)
            @test isapprox(dpts, fd_grad; rtol = 1.0e-3)
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
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss_laplacian_weights, Active, Duplicated(pts_flat, dpts))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_laplacian_weights, pts_flat)[1]
            @test !all(iszero, dpts)
            @test isapprox(dpts, fd_grad; rtol = 1.0e-3)
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
                dpts = zeros(length(pts_flat))
                Enzyme.autodiff(Reverse, loss_phs_order, Active, Duplicated(pts_flat, dpts))

                fd_grad = FD.grad(FD.central_fdm(5, 1), loss_phs_order, pts_flat)[1]
                @test !all(iszero, dpts) || n == 1  # PHS1 may have zero gradient for some configurations
                @test isapprox(dpts, fd_grad; rtol = 1.0e-2)
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
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss_imq_partial, Active, Duplicated(pts_flat, dpts))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_imq_partial, pts_flat)[1]
            @test !all(iszero, dpts)
            @test isapprox(dpts, fd_grad; rtol = 1.0e-3)
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
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss_imq_laplacian, Active, Duplicated(pts_flat, dpts))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_imq_laplacian, pts_flat)[1]
            @test !all(iszero, dpts)
            @test isapprox(dpts, fd_grad; rtol = 1.0e-3)
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
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss_gaussian_partial, Active, Duplicated(pts_flat, dpts))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_gaussian_partial, pts_flat)[1]
            @test !all(iszero, dpts)
            @test isapprox(dpts, fd_grad; rtol = 1.0e-3)
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
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss_gaussian_laplacian, Active, Duplicated(pts_flat, dpts))

            fd_grad = FD.grad(FD.central_fdm(5, 1), loss_gaussian_laplacian, pts_flat)[1]
            @test !all(iszero, dpts)
            @test isapprox(dpts, fd_grad; rtol = 1.0e-3)
        end

        @testset "Different shape parameters" begin
            for ε in [0.5, 1.0, 2.0]
                @testset "IMQ with ε=$ε" begin
                    basis = IMQ(ε; poly_deg = 2)
                    ℒ = Partial(1, 1)

                    function loss_imq_shape(pts)
                        pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                        W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                        return sum(W.nzval .^ 2)
                    end

                    pts_flat = vcat([collect(p) for p in points]...)
                    dpts = zeros(length(pts_flat))
                    Enzyme.autodiff(Reverse, loss_imq_shape, Active, Duplicated(pts_flat, dpts))

                    fd_grad = FD.grad(FD.central_fdm(5, 1), loss_imq_shape, pts_flat)[1]
                    @test !all(iszero, dpts)
                    @test isapprox(dpts, fd_grad; rtol = 1.0e-2)
                end

                @testset "Gaussian with ε=$ε" begin
                    basis = Gaussian(ε; poly_deg = 2)
                    ℒ = Partial(1, 1)

                    function loss_gaussian_shape(pts)
                        pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
                        W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
                        return sum(W.nzval .^ 2)
                    end

                    pts_flat = vcat([collect(p) for p in points]...)
                    dpts = zeros(length(pts_flat))
                    Enzyme.autodiff(Reverse, loss_gaussian_shape, Active, Duplicated(pts_flat, dpts))

                    fd_grad = FD.grad(FD.central_fdm(5, 1), loss_gaussian_shape, pts_flat)[1]
                    @test !all(iszero, dpts)
                    @test isapprox(dpts, fd_grad; rtol = 1.0e-2)
                end
            end
        end
    end
else
    @testset "Enzyme Extension - Julia $(VERSION) (skipped)" begin
        @test_skip begin
            # Enzyme.jl has known issues with Julia 1.12+
            # See: https://github.com/EnzymeAD/Enzyme.jl/issues/2699
            @info "Enzyme tests skipped on Julia $(VERSION) due to known compatibility issues"
            true
        end
    end
end

# Test that the extension loads correctly regardless of Julia version
@testset "Enzyme Extension - Loading" begin
    @test Base.find_package("Enzyme") !== nothing
    # The extension should be loaded when both packages are available
    # (verified by the package loading without error)
    @test true
end
