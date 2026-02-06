using RadialBasisFunctions
using ChainRulesCore
using StaticArraysCore
using FiniteDifferences
using Test

const FD = FiniteDifferences

@testset "ChainRulesCore - Interpolator Construction Rules" begin
    N = 30
    points = [SVector{2}(0.1 + 0.8 * i / 6, 0.1 + 0.8 * j / 6) for i in 1:6 for j in 1:5]
    values = sin.(getindex.(points, 1))
    eval_points = [SVector{2}(0.2 + 0.6 * i / 3, 0.2 + 0.6 * j / 3) for i in 1:3 for j in 1:3]

    @testset "End-to-end gradient test via rrules" begin
        # Forward pass: construct and evaluate
        interp, construction_pb = ChainRulesCore.rrule(Interpolator, points, values)
        ys, eval_pb = ChainRulesCore.rrule(interp, eval_points)

        # Backward pass: start with loss gradient
        Δys = 2 .* ys  # gradient of sum(y^2) w.r.t. y
        Δinterp, Δxs = eval_pb(Δys)

        # Propagate through construction
        result = construction_pb(Δinterp)
        Δv = result[3]

        # Compare with finite differences
        function loss(v)
            interp_local = Interpolator(points, v)
            return sum(interp_local(eval_points) .^ 2)
        end

        fd_grad = FD.grad(FD.central_fdm(5, 1), loss, values)[1]

        @test !all(iszero, Δv)
        @test isapprox(Δv, fd_grad; rtol = 1.0e-4)
    end

    @testset "Different bases" begin
        # Use slightly relaxed tolerance for Gaussian basis which can be more sensitive
        for (name, basis, tol) in [("PHS", PHS(), 1.0e-3), ("IMQ", IMQ(1.0), 1.0e-3), ("Gaussian", Gaussian(1.0), 5.0e-2)]
            @testset "$name basis" begin
                interp, construction_pb = ChainRulesCore.rrule(Interpolator, points, values, basis)
                ys, eval_pb = ChainRulesCore.rrule(interp, eval_points)

                Δys = 2 .* ys
                Δinterp, _ = eval_pb(Δys)
                result = construction_pb(Δinterp)
                Δv = result[3]

                function loss_basis(v)
                    interp_local = Interpolator(points, v, basis)
                    return sum(interp_local(eval_points) .^ 2)
                end

                fd_grad = FD.grad(FD.central_fdm(5, 1), loss_basis, values)[1]

                @test !all(iszero, Δv)
                @test isapprox(Δv, fd_grad; rtol = tol)
            end
        end
    end

    @testset "Single point evaluation" begin
        single_point = SVector{2}(0.5, 0.5)

        interp, construction_pb = ChainRulesCore.rrule(Interpolator, points, values)
        y, eval_pb = ChainRulesCore.rrule(interp, single_point)

        Δy = 2 * y  # gradient of y^2
        Δinterp, _ = eval_pb(Δy)
        result = construction_pb(Δinterp)
        Δv = result[3]

        function loss_single(v)
            interp_local = Interpolator(points, v)
            return interp_local(single_point)^2
        end

        fd_grad = FD.grad(FD.central_fdm(5, 1), loss_single, values)[1]

        @test !all(iszero, Δv)
        @test isapprox(Δv, fd_grad; rtol = 1.0e-4)
    end
end

@testset "ChainRulesCore - Evaluation Rules" begin
    N = 30
    points = [SVector{2}(0.1 + 0.8 * i / 6, 0.1 + 0.8 * j / 6) for i in 1:6 for j in 1:5]
    values = sin.(getindex.(points, 1))
    interp = Interpolator(points, values)

    @testset "Single point evaluation gradient w.r.t. x" begin
        x = SVector{2}(0.5, 0.5)

        y, eval_pb = ChainRulesCore.rrule(interp, x)
        Δy = 1.0
        Δinterp, Δx = eval_pb(Δy)

        # Compare with finite differences
        function eval_loss(xv)
            return interp(SVector{2}(xv[1], xv[2]))
        end

        fd_grad = FD.grad(FD.central_fdm(5, 1), eval_loss, collect(x))[1]

        @test isapprox(collect(Δx), fd_grad; rtol = 1.0e-4)
    end

    @testset "Batch evaluation gradient w.r.t. x" begin
        xs = [SVector{2}(0.3, 0.3), SVector{2}(0.5, 0.5), SVector{2}(0.7, 0.7)]

        ys, eval_pb = ChainRulesCore.rrule(interp, xs)
        Δys = ones(length(xs))
        Δinterp, Δxs = eval_pb(Δys)

        # Compare each point's gradient with finite differences
        for (i, x) in enumerate(xs)
            function eval_loss_i(xv)
                xs_mod = copy(xs)
                xs_mod[i] = SVector{2}(xv[1], xv[2])
                return sum(interp(xs_mod))
            end

            fd_grad = FD.grad(FD.central_fdm(5, 1), eval_loss_i, collect(x))[1]
            @test isapprox(collect(Δxs[i]), fd_grad; rtol = 1.0e-4)
        end
    end

    @testset "Weight gradients from evaluation" begin
        x = SVector{2}(0.5, 0.5)

        y, eval_pb = ChainRulesCore.rrule(interp, x)
        Δy = 1.0
        Δinterp, _ = eval_pb(Δy)

        # Check that weight gradients are returned
        @test Δinterp isa Tangent{Interpolator}
        @test hasproperty(Δinterp, :rbf_weights)
        @test hasproperty(Δinterp, :monomial_weights)
        @test !all(iszero, Δinterp.rbf_weights)
    end
end
