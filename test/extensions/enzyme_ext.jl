include("ad_test_utils.jl")
using Enzyme

# Check Julia version - Enzyme.jl has known issues with Julia 1.12+
# See: https://github.com/EnzymeAD/Enzyme.jl/issues/2699
const ENZYME_SUPPORTED_JULIA = VERSION < v"1.12"

if ENZYME_SUPPORTED_JULIA
    @testset "Enzyme Extension - Operator Differentiation" begin
        points, N, values = make_operator_test_data()

        @testset "Laplacian Operator" begin
            lap = laplacian(points)
            loss(v) = sum(lap(v) .^ 2)

            dv = zeros(N)
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(values, dv))
            validate_gradient(dv, loss, values)
        end

        @testset "Gradient Operator" begin
            grad = gradient(points)
            loss(v) = sum(grad(v) .^ 2)

            dv = zeros(N)
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(values, dv))
            validate_gradient(dv, loss, values)
        end

        @testset "Partial Derivative Operator" begin
            partial_x = partial(points, 1, 1)
            loss(v) = sum(partial_x(v) .^ 2)

            dv = zeros(N)
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(values, dv))
            validate_gradient(dv, loss, values)
        end
    end

    @testset "Enzyme Extension - Interpolator Differentiation" begin
        N = 30
        points = [SVector{2}(rand(), rand()) for _ in 1:N]
        values = sin.(getindex.(points, 1))
        eval_points = [SVector{2}(rand(), rand()) for _ in 1:10]

        function loss_interp(v)
            interp_local = Interpolator(points, v)
            result = interp_local(eval_points)
            return sum(result .^ 2)
        end

        dv = zeros(N)
        Enzyme.autodiff(Reverse, loss_interp, Active, Duplicated(values, dv))
        validate_gradient(dv, loss_interp, values; rtol = 1.0e-3)
    end

    @testset "Enzyme Extension - Basis Function Differentiation" begin
        x = [0.5, 0.5]
        xi = [0.3, 0.4]

        @testset "PHS Basis Functions" begin
            for phs_type in [PHS(1), PHS(3), PHS(5), PHS(7)]
                loss(xv) = phs_type(xv, xi)^2

                dx = zeros(2)
                Enzyme.autodiff(Reverse, loss, Active, Duplicated(x, dx))
                validate_gradient(dx, loss, x)
            end
        end

        @testset "IMQ Basis Function" begin
            imq = IMQ(1.0)
            loss(xv) = imq(xv, xi)^2

            dx = zeros(2)
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(x, dx))
            validate_gradient(dx, loss, x)
        end

        @testset "Gaussian Basis Function" begin
            gauss = Gaussian(1.0)
            loss(xv) = gauss(xv, xi)^2

            dx = zeros(2)
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(x, dx))
            validate_gradient(dx, loss, x)
        end
    end

    @testset "Enzyme Extension - Native Rules for _build_weights" begin
        points, N, adjl, pts_flat = make_build_weights_test_data()

        @testset "Partial operator with PHS3" begin
            loss = make_build_weights_loss(Partial(1, 1), adjl, PHS(3; poly_deg = 2), N)
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(pts_flat, dpts))
            validate_gradient(dpts, loss, pts_flat; rtol = 1.0e-3)
        end

        @testset "Laplacian operator with PHS3" begin
            loss = make_build_weights_loss(Laplacian(), adjl, PHS(3; poly_deg = 2), N)
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(pts_flat, dpts))
            validate_gradient(dpts, loss, pts_flat; rtol = 1.0e-3)
        end

        @testset "Different PHS orders" begin
            for n in [1, 3, 5, 7]
                loss = make_build_weights_loss(Partial(1, 1), adjl, PHS(n; poly_deg = 1), N)
                dpts = zeros(length(pts_flat))
                Enzyme.autodiff(Reverse, loss, Active, Duplicated(pts_flat, dpts))
                validate_gradient(dpts, loss, pts_flat; rtol = 1.0e-2, check_nonzero = (n != 1))
            end
        end

        @testset "$BasisName basis with $OpName operator" for
            (BasisName, basis) in [("IMQ", IMQ(1.0; poly_deg = 2)), ("Gaussian", Gaussian(1.0; poly_deg = 2))],
                (OpName, op) in [("Partial", Partial(1, 1)), ("Laplacian", Laplacian())]
            loss = make_build_weights_loss(op, adjl, basis, N)
            dpts = zeros(length(pts_flat))
            Enzyme.autodiff(Reverse, loss, Active, Duplicated(pts_flat, dpts))
            validate_gradient(dpts, loss, pts_flat; rtol = 1.0e-3)
        end

        @testset "Different shape parameters" begin
            for ε in [0.5, 1.0, 2.0]
                @testset "$BasisName with ε=$ε" for (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)]
                    loss = make_build_weights_loss(Partial(1, 1), adjl, BT(ε; poly_deg = 2), N)
                    dpts = zeros(length(pts_flat))
                    Enzyme.autodiff(Reverse, loss, Active, Duplicated(pts_flat, dpts))
                    validate_gradient(dpts, loss, pts_flat; rtol = 1.0e-2)
                end
            end
        end

        @testset "Shape parameter (ε) differentiation via Active basis" begin
            @testset "$BasisName $OpName - d(loss)/d(ε)" for
                (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)],
                    (OpName, op) in [("Partial", Partial(1, 1)), ("Laplacian", Laplacian())]
                loss = make_eps_loss(op, points, adjl, BT)
                dε = Enzyme.autodiff(Reverse, loss, Active, Active(1.0))[1][1]
                validate_scalar_gradient(dε, loss, 1.0)
            end

            @testset "Different ε values" begin
                for ε_val in [0.5, 2.0, 5.0]
                    @testset "$BasisName ε=$ε_val" for (BasisName, BT) in [("IMQ", IMQ), ("Gaussian", Gaussian)]
                        loss = make_eps_loss(Partial(1, 1), points, adjl, BT)
                        dε = Enzyme.autodiff(Reverse, loss, Active, Active(ε_val))[1][1]
                        validate_scalar_gradient(dε, loss, ε_val; rtol = 1.0e-2)
                    end
                end
            end
        end
    end
else
    @testset "Enzyme Extension - Julia $(VERSION) (skipped)" begin
        @test_skip begin
            @info "Enzyme tests skipped on Julia $(VERSION) due to known compatibility issues"
            true
        end
    end
end

@testset "Enzyme Extension - Loading" begin
    @test Base.find_package("Enzyme") !== nothing
    @test true
end
