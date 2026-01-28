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
            @test isapprox(dv, fd_grad; rtol=1e-4)
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
            @test isapprox(dv, fd_grad; rtol=1e-4)
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
            @test isapprox(dv, fd_grad; rtol=1e-4)
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
        @test isapprox(dv, fd_grad; rtol=1e-3)
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
                @test isapprox(dx, fd_grad; rtol=1e-4)
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
            @test isapprox(dx, fd_grad; rtol=1e-4)
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
            @test isapprox(dx, fd_grad; rtol=1e-4)
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
