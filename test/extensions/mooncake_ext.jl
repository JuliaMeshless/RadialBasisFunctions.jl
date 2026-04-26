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
    validate_gradient(dx, loss, eval_pt; rtol=1.0e-3)
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
        validate_gradient(dpts, loss, pts_flat; rtol=1.0e-3)
    end

    @testset "Laplacian operator with PHS3" begin
        loss = make_build_weights_loss(Laplacian(), adjl, PHS(3; poly_deg=2), N)
        rule = Mooncake.build_rrule(loss, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
        validate_gradient(dpts, loss, pts_flat; rtol=1.0e-3)
    end

    @testset "Different PHS orders" begin
        for n in [1, 3, 5, 7]
            loss = make_build_weights_loss(Partial(1, 1), adjl, PHS(n; poly_deg=1), N)
            rule = Mooncake.build_rrule(loss, pts_flat)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
            validate_gradient(dpts, loss, pts_flat; rtol=1.0e-2, check_nonzero=(n != 1))
        end
    end

    @testset "$BasisName basis with $OpName operator" for (BasisName, basis) in [("IMQ", IMQ(1.0; poly_deg=2)), ("Gaussian", Gaussian(1.0; poly_deg=2))],
        (OpName, op) in [("Partial", Partial(1, 1)), ("Laplacian", Laplacian())]

        loss = make_build_weights_loss(op, adjl, basis, N)
        rule = Mooncake.build_rrule(loss, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
        validate_gradient(dpts, loss, pts_flat; rtol=1.0e-3)
    end

    @testset "Different shape parameters" begin
        for ε in [0.5, 1.0, 2.0]
            @testset "$BasisName with ε=$ε" for (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)]
                loss = make_build_weights_loss(Partial(1, 1), adjl, BT(ε; poly_deg=2), N)
                rule = Mooncake.build_rrule(loss, pts_flat)
                _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
                validate_gradient(dpts, loss, pts_flat; rtol=1.0e-2)
            end
        end
    end

    @testset "Shape parameter (ε) differentiation" begin
        @testset "$BasisName $OpName - d(loss)/d(ε)" for (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)],
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
                    validate_scalar_gradient(dε, loss, ε_val; rtol=1.0e-2)
                end
            end
        end
    end
end

@testset "Mooncake Extension - MixedPartial _build_weights" begin
    points, N, adjl, pts_flat = make_build_weights_test_data()

    @testset "MixedPartial(1,2) with PHS3" begin
        loss = make_build_weights_loss(MixedPartial(1, 2), adjl, PHS(3; poly_deg=2), N)
        rule = Mooncake.build_rrule(loss, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
        validate_gradient(dpts, loss, pts_flat; rtol=1.0e-3)
    end

    @testset "MixedPartial(1,2) with PHS1/5/7" begin
        for n in [1, 5, 7]
            loss = make_build_weights_loss(MixedPartial(1, 2), adjl, PHS(n; poly_deg=2), N)
            rule = Mooncake.build_rrule(loss, pts_flat)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
            validate_gradient(dpts, loss, pts_flat; rtol=1.0e-2)
        end
    end

    @testset "MixedPartial(1,2) with IMQ and Gaussian" begin
        for (name, basis) in [("IMQ", IMQ(1.0; poly_deg=2)), ("Gaussian", Gaussian(1.0; poly_deg=2))]
            loss = make_build_weights_loss(MixedPartial(1, 2), adjl, basis, N)
            rule = Mooncake.build_rrule(loss, pts_flat)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss, pts_flat)
            validate_gradient(dpts, loss, pts_flat; rtol=1.0e-3)
        end
    end

    @testset "MixedPartial in 3D: (1,2), (1,3), (2,3)" begin
        N3 = 64
        pts3 = [SVector{3}(0.1 + 0.8 * ((i * 7 + 3) % N3) / N3,
            0.1 + 0.8 * ((i * 11 + 5) % N3) / N3,
            0.1 + 0.8 * ((i * 13 + 7) % N3) / N3) for i in 1:N3]
        adj3 = RadialBasisFunctions.find_neighbors(pts3, 20)
        flat3 = vcat([collect(p) for p in pts3]...)
        for (d1, d2) in [(1, 2), (1, 3), (2, 3)]
            basis3d = PHS(3; poly_deg=2)
            op3d = MixedPartial(d1, d2)
            loss3d = function (pts)
                pts_vec = [SVector{3}(pts[3i-2], pts[3i-1], pts[3i]) for i in 1:N3]
                W = RadialBasisFunctions._build_weights(op3d, pts_vec, pts_vec, adj3, basis3d)
                return sum(W.nzval .^ 2)
            end
            rule = Mooncake.build_rrule(loss3d, flat3)
            _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss3d, flat3)
            validate_gradient(dpts, loss3d, flat3; rtol=1.0e-3)
        end
    end
end

@testset "Mooncake Extension - Directional _build_weights" begin
    points, N, adjl, pts_flat, n_const, n_varying = make_directional_test_data()

    @testset "Constant direction - gradient w.r.t. pts" begin
        function loss_dir_const(pts)
            pts_vec = [SVector{2}(pts[2i-1], pts[2i]) for i in 1:N]
            W = RadialBasisFunctions._build_weights(
                RadialBasisFunctions.Directional{2}(n_const),
                pts_vec, pts_vec, adjl, PHS(3; poly_deg=2))
            return sum(W.nzval .^ 2)
        end
        rule = Mooncake.build_rrule(loss_dir_const, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_dir_const, pts_flat)
        validate_gradient(dpts, loss_dir_const, pts_flat; rtol=1.0e-3)
    end

    @testset "Spatially varying direction - gradient w.r.t. pts" begin
        function loss_dir_varying(pts)
            pts_vec = [SVector{2}(pts[2i-1], pts[2i]) for i in 1:N]
            W = RadialBasisFunctions._build_weights(
                RadialBasisFunctions.Directional{2}(n_varying),
                pts_vec, pts_vec, adjl, PHS(3; poly_deg=2))
            return sum(W.nzval .^ 2)
        end
        rule = Mooncake.build_rrule(loss_dir_varying, pts_flat)
        _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_dir_varying, pts_flat)
        validate_gradient(dpts, loss_dir_varying, pts_flat; rtol=1.0e-3)
    end

    @testset "Constant direction - gradient flows into direction vector" begin
        pts_fixed = [SVector{2}(0.1 + 0.8 * i / 5, 0.1 + 0.8 * j / 5) for i in 1:5 for j in 1:5]
        adj_fixed = RadialBasisFunctions.find_neighbors(pts_fixed, 10)
        function loss_wrt_direction(n_flat)
            n = [n_flat[1], n_flat[2]]
            W = RadialBasisFunctions._build_weights(
                RadialBasisFunctions.Directional{2}(n),
                pts_fixed, pts_fixed, adj_fixed, PHS(3; poly_deg=2))
            return sum(W.nzval .^ 2)
        end
        n0 = [1.0 / sqrt(2), 1.0 / sqrt(2)]
        rule = Mooncake.build_rrule(loss_wrt_direction, n0)
        _, (_, dn) = Mooncake.value_and_gradient!!(rule, loss_wrt_direction, n0)
        validate_gradient(dn, loss_wrt_direction, n0; rtol=1.0e-3)
    end
end

@testset "Mooncake Extension - Elasticity assembly gradient (all 5 operators)" begin
    N = 25
    points = [SVector{2}(0.1 + 0.8 * i / 5, 0.1 + 0.8 * j / 5) for i in 1:5 for j in 1:5]
    adjl = RadialBasisFunctions.find_neighbors(points, 14)
    basis = PHS(3; poly_deg=2)
    pts_flat = vcat([collect(p) for p in points]...)

    function loss_elasticity_assembly(pts)
        p = [SVector{2}(pts[2i-1], pts[2i]) for i in 1:N]
        W_d2x = RadialBasisFunctions._build_weights(Partial(2, 1), p, p, adjl, basis)
        W_d2y = RadialBasisFunctions._build_weights(Partial(2, 2), p, p, adjl, basis)
        W_d2xy = RadialBasisFunctions._build_weights(MixedPartial(1, 2), p, p, adjl, basis)
        W_dx = RadialBasisFunctions._build_weights(Partial(1, 1), p, p, adjl, basis)
        W_dy = RadialBasisFunctions._build_weights(Partial(1, 2), p, p, adjl, basis)
        return sum(W_d2x.nzval .^ 2) + sum(W_d2y.nzval .^ 2) + sum(W_d2xy.nzval .^ 2) +
               sum(W_dx.nzval .^ 2) + sum(W_dy.nzval .^ 2)
    end

    rule = Mooncake.build_rrule(loss_elasticity_assembly, pts_flat)
    _, (_, dpts) = Mooncake.value_and_gradient!!(rule, loss_elasticity_assembly, pts_flat)
    validate_gradient(dpts, loss_elasticity_assembly, pts_flat; rtol=1.0e-3)
end

@testset "Mooncake Extension - Loading" begin
    @test Base.find_package("Mooncake") !== nothing
    @test true
end
