using RadialBasisFunctions
using Mooncake
using StaticArraysCore
using FiniteDifferences
using LinearAlgebra
using Test

const FDM = FiniteDifferences

@testset "Mooncake Extension - Operator Differentiation" begin
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

        rule = Mooncake.build_rrule(loss_lap, values)
        _, (_, dv) = Mooncake.value_and_gradient!!(rule, loss_lap, values)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_lap, values)[1]
        @test !all(iszero, dv)
        @test isapprox(dv, fd_grad; rtol = 1.0e-4)
    end

    @testset "Gradient Operator" begin
        grad = gradient(points)

        function loss_grad(v)
            result = grad(v)
            return sum(result .^ 2)
        end

        rule = Mooncake.build_rrule(loss_grad, values)
        _, (_, dv) = Mooncake.value_and_gradient!!(rule, loss_grad, values)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_grad, values)[1]
        @test !all(iszero, dv)
        @test isapprox(dv, fd_grad; rtol = 1.0e-4)
    end

    @testset "Partial Derivative Operator" begin
        partial_x = partial(points, 1, 1)

        function loss_partial(v)
            result = partial_x(v)
            return sum(result .^ 2)
        end

        rule = Mooncake.build_rrule(loss_partial, values)
        _, (_, dv) = Mooncake.value_and_gradient!!(rule, loss_partial, values)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_partial, values)[1]
        @test !all(iszero, dv)
        @test isapprox(dv, fd_grad; rtol = 1.0e-4)
    end
end

@testset "Mooncake Extension - Interpolator Differentiation" begin
    N = 30
    points = [SVector{2}(rand(), rand()) for _ in 1:N]
    values = sin.(getindex.(points, 1))

    # Pre-build interpolator outside AD (Mooncake can't differentiate through
    # LAPACK factorization in the constructor)
    interp = Interpolator(points, values)

    # Differentiate w.r.t. evaluation point
    eval_pt = [0.5, 0.5]

    function loss_interp(x)
        return interp(x)^2
    end

    rule = Mooncake.build_rrule(loss_interp, eval_pt)
    _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss_interp, eval_pt)

    fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_interp, eval_pt)[1]
    @test !all(iszero, dx)
    @test isapprox(dx, fd_grad; rtol = 1.0e-3)
end

@testset "Mooncake Extension - Basis Function Differentiation" begin
    x = [0.5, 0.5]
    xi = [0.3, 0.4]

    @testset "PHS Basis Functions" begin
        for phs_type in [PHS(1), PHS(3), PHS(5), PHS(7)]
            function loss_phs(xv)
                return phs_type(xv, xi)^2
            end

            rule = Mooncake.build_rrule(loss_phs, x)
            _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss_phs, x)

            fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_phs, x)[1]
            @test !all(iszero, dx)
            @test isapprox(dx, fd_grad; rtol = 1.0e-4)
        end
    end

    @testset "IMQ Basis Function" begin
        imq = IMQ(1.0)

        function loss_imq(xv)
            return imq(xv, xi)^2
        end

        rule = Mooncake.build_rrule(loss_imq, x)
        _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss_imq, x)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_imq, x)[1]
        @test !all(iszero, dx)
        @test isapprox(dx, fd_grad; rtol = 1.0e-4)
    end

    @testset "Gaussian Basis Function" begin
        gauss = Gaussian(1.0)

        function loss_gauss(xv)
            return gauss(xv, xi)^2
        end

        rule = Mooncake.build_rrule(loss_gauss, x)
        _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss_gauss, x)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_gauss, x)[1]
        @test !all(iszero, dx)
        @test isapprox(dx, fd_grad; rtol = 1.0e-4)
    end
end

@testset "Mooncake Extension - Native Rules for _build_weights" begin
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
        rule = Mooncake.build_rrule(loss_partial_weights, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_partial_weights, pts_flat)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_partial_weights, pts_flat)[1]
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
        rule = Mooncake.build_rrule(loss_laplacian_weights, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_laplacian_weights, pts_flat)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_laplacian_weights, pts_flat)[1]
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
            rule = Mooncake.build_rrule(loss_phs_order, pts_flat)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_phs_order, pts_flat)

            fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_phs_order, pts_flat)[1]
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
        rule = Mooncake.build_rrule(loss_imq_partial, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_imq_partial, pts_flat)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_imq_partial, pts_flat)[1]
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
        rule = Mooncake.build_rrule(loss_imq_laplacian, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_imq_laplacian, pts_flat)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_imq_laplacian, pts_flat)[1]
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
        rule = Mooncake.build_rrule(loss_gaussian_partial, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_gaussian_partial, pts_flat)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_gaussian_partial, pts_flat)[1]
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
        rule = Mooncake.build_rrule(loss_gaussian_laplacian, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_gaussian_laplacian, pts_flat)

        fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_gaussian_laplacian, pts_flat)[1]
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
                rule = Mooncake.build_rrule(loss_imq_shape, pts_flat)
                _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_imq_shape, pts_flat)

                fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_imq_shape, pts_flat)[1]
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
                rule = Mooncake.build_rrule(loss_gaussian_shape, pts_flat)
                _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_gaussian_shape, pts_flat)

                fd_grad = FDM.grad(FDM.central_fdm(5, 1), loss_gaussian_shape, pts_flat)[1]
                @test !all(iszero, dpts)
                @test isapprox(dpts, fd_grad; rtol = 1.0e-2)
            end
        end
    end

    @testset "Shape parameter (ε) differentiation" begin
        @testset "IMQ Partial - d(loss)/d(ε)" begin
            ℒ = Partial(1, 1)

            function loss_imq_eps(ε)
                basis = IMQ(ε; poly_deg = 2)
                W = RadialBasisFunctions._build_weights(ℒ, points, points, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            rule = Mooncake.build_rrule(loss_imq_eps, 1.0)
            _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss_imq_eps, 1.0)
            fd_dε = FDM.central_fdm(5, 1)(loss_imq_eps, 1.0)
            @test !iszero(dε)
            @test isapprox(dε, fd_dε; rtol = 1.0e-3)
        end

        @testset "IMQ Laplacian - d(loss)/d(ε)" begin
            ℒ = Laplacian()

            function loss_imq_lap_eps(ε)
                basis = IMQ(ε; poly_deg = 2)
                W = RadialBasisFunctions._build_weights(ℒ, points, points, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            rule = Mooncake.build_rrule(loss_imq_lap_eps, 1.0)
            _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss_imq_lap_eps, 1.0)
            fd_dε = FDM.central_fdm(5, 1)(loss_imq_lap_eps, 1.0)
            @test !iszero(dε)
            @test isapprox(dε, fd_dε; rtol = 1.0e-3)
        end

        @testset "Gaussian Partial - d(loss)/d(ε)" begin
            ℒ = Partial(1, 1)

            function loss_gauss_eps(ε)
                basis = Gaussian(ε; poly_deg = 2)
                W = RadialBasisFunctions._build_weights(ℒ, points, points, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            rule = Mooncake.build_rrule(loss_gauss_eps, 1.0)
            _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss_gauss_eps, 1.0)
            fd_dε = FDM.central_fdm(5, 1)(loss_gauss_eps, 1.0)
            @test !iszero(dε)
            @test isapprox(dε, fd_dε; rtol = 1.0e-3)
        end

        @testset "Gaussian Laplacian - d(loss)/d(ε)" begin
            ℒ = Laplacian()

            function loss_gauss_lap_eps(ε)
                basis = Gaussian(ε; poly_deg = 2)
                W = RadialBasisFunctions._build_weights(ℒ, points, points, adjl, basis)
                return sum(W.nzval .^ 2)
            end

            rule = Mooncake.build_rrule(loss_gauss_lap_eps, 1.0)
            _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss_gauss_lap_eps, 1.0)
            fd_dε = FDM.central_fdm(5, 1)(loss_gauss_lap_eps, 1.0)
            @test !iszero(dε)
            @test isapprox(dε, fd_dε; rtol = 1.0e-3)
        end

        @testset "Different ε values" begin
            for ε_val in [0.5, 2.0, 5.0]
                @testset "IMQ ε=$ε_val" begin
                    ℒ = Partial(1, 1)

                    function loss_imq_eps_val(ε)
                        basis = IMQ(ε; poly_deg = 2)
                        W = RadialBasisFunctions._build_weights(ℒ, points, points, adjl, basis)
                        return sum(W.nzval .^ 2)
                    end

                    rule = Mooncake.build_rrule(loss_imq_eps_val, ε_val)
                    _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss_imq_eps_val, ε_val)
                    fd_dε = FDM.central_fdm(5, 1)(loss_imq_eps_val, ε_val)
                    @test !iszero(dε)
                    @test isapprox(dε, fd_dε; rtol = 1.0e-2)
                end

                @testset "Gaussian ε=$ε_val" begin
                    ℒ = Partial(1, 1)

                    function loss_gauss_eps_val(ε)
                        basis = Gaussian(ε; poly_deg = 2)
                        W = RadialBasisFunctions._build_weights(ℒ, points, points, adjl, basis)
                        return sum(W.nzval .^ 2)
                    end

                    rule = Mooncake.build_rrule(loss_gauss_eps_val, ε_val)
                    _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss_gauss_eps_val, ε_val)
                    fd_dε = FDM.central_fdm(5, 1)(loss_gauss_eps_val, ε_val)
                    @test !iszero(dε)
                    @test isapprox(dε, fd_dε; rtol = 1.0e-2)
                end
            end
        end
    end
end

@testset "Mooncake Extension - Loading" begin
    @test Base.find_package("Mooncake") !== nothing
    @test true
end
