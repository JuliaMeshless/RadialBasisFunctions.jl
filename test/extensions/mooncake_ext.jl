include("ad_test_utils.jl")
using Mooncake

@testset "Mooncake Extension - Operator Differentiation" begin
    points, N, values = make_operator_test_data()

    @testset "Laplacian Operator" begin
        lap = laplacian(points)
        loss(v) = sum(lap(v) .^ 2)

        rule = Mooncake.build_rrule(loss, values)
        _, (_, dv) = Mooncake.value_and_gradient!!(rule, loss, values)
        validate_gradient(dv, loss, values)
    end

    @testset "Gradient Operator" begin
        grad = gradient(points)
        loss(v) = sum(grad(v) .^ 2)

        rule = Mooncake.build_rrule(loss, values)
        _, (_, dv) = Mooncake.value_and_gradient!!(rule, loss, values)
        validate_gradient(dv, loss, values)
    end

    @testset "Partial Derivative Operator" begin
        partial_x = partial(points, 1, 1)
        loss(v) = sum(partial_x(v) .^ 2)

        rule = Mooncake.build_rrule(loss, values)
        _, (_, dv) = Mooncake.value_and_gradient!!(rule, loss, values)
        validate_gradient(dv, loss, values)
    end
end

@testset "Mooncake Extension - Interpolator Differentiation" begin
    N = 30
    points = [SVector{2}(rand(), rand()) for _ in 1:N]
    values = sin.(getindex.(points, 1))

    interp = Interpolator(points, values)
    eval_pt = [0.5, 0.5]
    loss(x) = interp(x)^2

    rule = Mooncake.build_rrule(loss, eval_pt)
    _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss, eval_pt)
    validate_gradient(dx, loss, eval_pt; rtol=1e-3)
end

@testset "Mooncake Extension - Basis Function Differentiation" begin
    x = [0.5, 0.5]
    xi = [0.3, 0.4]

    @testset "PHS Basis Functions" begin
        for phs_type in [PHS(1), PHS(3), PHS(5), PHS(7)]
            loss(xv) = phs_type(xv, xi)^2

            rule = Mooncake.build_rrule(loss, x)
            _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss, x)
            validate_gradient(dx, loss, x)
        end
    end

    @testset "IMQ Basis Function" begin
        imq = IMQ(1.0)
        loss(xv) = imq(xv, xi)^2

        rule = Mooncake.build_rrule(loss, x)
        _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss, x)
        validate_gradient(dx, loss, x)
    end

    @testset "Gaussian Basis Function" begin
        gauss = Gaussian(1.0)
        loss(xv) = gauss(xv, xi)^2

        rule = Mooncake.build_rrule(loss, x)
        _, (_, dx) = Mooncake.value_and_gradient!!(rule, loss, x)
        validate_gradient(dx, loss, x)
    end
end

@testset "Mooncake Extension - Native Rules for _build_weights" begin
    points, N, adjl, pts_flat = make_build_weights_test_data()

    @testset "Partial operator with PHS3" begin
        loss = make_build_weights_loss(Partial(1, 1), adjl, PHS(3; poly_deg=2), N)
        rule = Mooncake.build_rrule(loss, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
        validate_gradient(dpts, loss, pts_flat; rtol=1e-3)
    end

    @testset "Laplacian operator with PHS3" begin
        loss = make_build_weights_loss(Laplacian(), adjl, PHS(3; poly_deg=2), N)
        rule = Mooncake.build_rrule(loss, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
        validate_gradient(dpts, loss, pts_flat; rtol=1e-3)
    end

    @testset "Different PHS orders" begin
        for n in [1, 3, 5, 7]
            loss = make_build_weights_loss(Partial(1, 1), adjl, PHS(n; poly_deg=1), N)
            rule = Mooncake.build_rrule(loss, pts_flat)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
            validate_gradient(dpts, loss, pts_flat; rtol=1e-2, check_nonzero=(n != 1))
        end
    end

    @testset "$BasisName basis with $OpName operator" for
            (BasisName, basis) in [("IMQ", IMQ(1.0; poly_deg=2)), ("Gaussian", Gaussian(1.0; poly_deg=2))],
            (OpName, op) in [("Partial", Partial(1, 1)), ("Laplacian", Laplacian())]
        loss = make_build_weights_loss(op, adjl, basis, N)
        rule = Mooncake.build_rrule(loss, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
        validate_gradient(dpts, loss, pts_flat; rtol=1e-3)
    end

    @testset "Different shape parameters" begin
        for ε in [0.5, 1.0, 2.0]
            @testset "$BasisName with ε=$ε" for (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)]
                loss = make_build_weights_loss(Partial(1, 1), adjl, BT(ε; poly_deg=2), N)
                rule = Mooncake.build_rrule(loss, pts_flat)
                _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
                validate_gradient(dpts, loss, pts_flat; rtol=1e-2)
            end
        end
    end

    @testset "Shape parameter (ε) differentiation" begin
        @testset "$BasisName $OpName - d(loss)/d(ε)" for
                (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)],
                (OpName, op) in [("Partial", Partial(1, 1)), ("Laplacian", Laplacian())]
            loss = make_eps_loss(op, points, adjl, BT)
            rule = Mooncake.build_rrule(loss, 1.0)
            _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss, 1.0)
            validate_scalar_gradient(dε, loss, 1.0)
        end

        @testset "Different ε values" begin
            for ε_val in [0.5, 2.0, 5.0]
                @testset "$BasisName ε=$ε_val" for (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)]
                    loss = make_eps_loss(Partial(1, 1), points, adjl, BT)
                    rule = Mooncake.build_rrule(loss, ε_val)
                    _, (_, dε) = Mooncake.value_and_gradient!!(rule, loss, ε_val)
                    validate_scalar_gradient(dε, loss, ε_val; rtol=1e-2)
                end
            end
        end
    end
end

@testset "Mooncake Extension - Loading" begin
    @test Base.find_package("Mooncake") !== nothing
    @test true
end
