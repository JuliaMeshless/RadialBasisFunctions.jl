using RadialBasisFunctions
using ChainRulesCore
using StaticArraysCore
using FiniteDifferences
using LinearAlgebra: Symmetric, dot
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

@testset "Direct backward stencil functions" begin
    # Test backward_stencil_partial! and backward_stencil_laplacian! directly.
    # These are the non-ε variants used by the Enzyme extension, which is skipped
    # on Julia 1.12+ so they need direct coverage.

    N = 25
    points = [SVector{2}(0.1 + 0.8 * i / 5, 0.1 + 0.8 * j / 5) for i in 1:5 for j in 1:5]
    adjl = RadialBasisFunctions.find_neighbors(points, 10)
    k = 10

    # Helper: compute stencil weights from flat local_data + eval_point vector
    function _compute_stencil_weights(flat_data, dim_space, k, basis, mon, ℒrbf, ℒmon)
        nmon = binomial(dim_space + basis.poly_deg, basis.poly_deg)
        n = k + nmon
        ld = [SVector{dim_space}(flat_data[(dim_space * (i - 1) + 1):(dim_space * i)]...) for i in 1:k]
        ep_start = dim_space * k + 1
        ep = SVector{dim_space}(flat_data[ep_start:(ep_start + dim_space - 1)]...)
        A_full = zeros(Float64, n, n)
        A = Symmetric(A_full, :U)
        RadialBasisFunctions._build_collocation_matrix!(A, ld, basis, mon, k)
        b = zeros(Float64, n)
        RadialBasisFunctions._build_rhs!(b, ℒrbf, ℒmon, ld, ep, basis, mon, k)
        λ = Symmetric(A_full, :U) \ b
        return λ[1:k]
    end

    @testset "backward_stencil_partial!" begin
        basis = PHS(3; poly_deg = 2)
        ℒ = Partial(1, 1)
        mon = MonomialBasis(2, basis.poly_deg)
        ℒmon = ℒ(mon)
        ℒrbf = ℒ(basis)

        W, cache = RadialBasisFunctions._forward_with_cache(
            points, points, adjl, basis, ℒrbf, ℒmon, mon, Partial
        )

        eval_idx = 13
        neighbors = adjl[eval_idx]
        eval_point = points[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = randn(k, 1)

        local_data = [points[i] for i in neighbors]
        Δlocal_data = [zeros(Float64, 2) for _ in 1:k]
        Δeval_pt = zeros(Float64, 2)

        grad_Lφ_x = RadialBasisFunctions.grad_applied_partial_wrt_x(basis, ℒ.dim)
        grad_Lφ_xi = RadialBasisFunctions.grad_applied_partial_wrt_xi(basis, ℒ.dim)

        RadialBasisFunctions.backward_stencil_partial!(
            Δlocal_data, Δeval_pt, Δw, stencil_cache, collect(1:k),
            eval_point, local_data, basis, mon, k, ℒ.dim, grad_Lφ_x, grad_Lφ_xi
        )

        # Compare with with-ε version (should give identical results for PHS)
        Δlocal_data_ε = [zeros(Float64, 2) for _ in 1:k]
        Δeval_pt_ε = zeros(Float64, 2)
        Δε_acc = Ref(0.0)

        RadialBasisFunctions.backward_stencil_partial_with_ε!(
            Δlocal_data_ε, Δeval_pt_ε, Δε_acc, Δw, stencil_cache, collect(1:k),
            eval_point, local_data, basis, mon, k, ℒ.dim, grad_Lφ_x, grad_Lφ_xi
        )

        for i in 1:k
            @test isapprox(Δlocal_data[i], Δlocal_data_ε[i]; atol = 1.0e-12)
        end
        @test isapprox(Δeval_pt, Δeval_pt_ε; atol = 1.0e-12)

        # Verify against finite differences
        flat_data = vcat([collect(d) for d in local_data]..., collect(eval_point))
        loss_stencil(x) = dot(
            Δw[:, 1], _compute_stencil_weights(x, 2, k, basis, mon, ℒrbf, ℒmon)
        )
        fd_grad = FD.grad(FD.central_fdm(5, 1), loss_stencil, flat_data)[1]

        backward_grad = zeros(2 * k + 2)
        for i in 1:k
            backward_grad[2 * i - 1] = Δlocal_data[i][1]
            backward_grad[2 * i] = Δlocal_data[i][2]
        end
        backward_grad[2 * k + 1] = Δeval_pt[1]
        backward_grad[2 * k + 2] = Δeval_pt[2]

        @test !all(iszero, backward_grad)
        @test isapprox(backward_grad, fd_grad; rtol = 1.0e-4)
    end

    @testset "backward_stencil_laplacian!" begin
        basis = PHS(3; poly_deg = 2)
        ℒ = Laplacian()
        mon = MonomialBasis(2, basis.poly_deg)
        ℒmon = ℒ(mon)
        ℒrbf = ℒ(basis)

        W, cache = RadialBasisFunctions._forward_with_cache(
            points, points, adjl, basis, ℒrbf, ℒmon, mon, Laplacian
        )

        eval_idx = 13
        neighbors = adjl[eval_idx]
        eval_point = points[eval_idx]
        stencil_cache = cache.stencil_caches[eval_idx]

        Δw = randn(k, 1)

        local_data = [points[i] for i in neighbors]
        Δlocal_data = [zeros(Float64, 2) for _ in 1:k]
        Δeval_pt = zeros(Float64, 2)

        grad_Lφ_x = RadialBasisFunctions.grad_applied_laplacian_wrt_x(basis)
        grad_Lφ_xi = RadialBasisFunctions.grad_applied_laplacian_wrt_xi(basis)

        RadialBasisFunctions.backward_stencil_laplacian!(
            Δlocal_data, Δeval_pt, Δw, stencil_cache, collect(1:k),
            eval_point, local_data, basis, mon, k, grad_Lφ_x, grad_Lφ_xi
        )

        # Compare with with-ε version (should give identical results for PHS)
        Δlocal_data_ε = [zeros(Float64, 2) for _ in 1:k]
        Δeval_pt_ε = zeros(Float64, 2)
        Δε_acc = Ref(0.0)

        RadialBasisFunctions.backward_stencil_laplacian_with_ε!(
            Δlocal_data_ε, Δeval_pt_ε, Δε_acc, Δw, stencil_cache, collect(1:k),
            eval_point, local_data, basis, mon, k, grad_Lφ_x, grad_Lφ_xi
        )

        for i in 1:k
            @test isapprox(Δlocal_data[i], Δlocal_data_ε[i]; atol = 1.0e-12)
        end
        @test isapprox(Δeval_pt, Δeval_pt_ε; atol = 1.0e-12)

        # Verify against finite differences
        flat_data = vcat([collect(d) for d in local_data]..., collect(eval_point))
        loss_stencil(x) = dot(
            Δw[:, 1], _compute_stencil_weights(x, 2, k, basis, mon, ℒrbf, ℒmon)
        )
        fd_grad = FD.grad(FD.central_fdm(5, 1), loss_stencil, flat_data)[1]

        backward_grad = zeros(2 * k + 2)
        for i in 1:k
            backward_grad[2 * i - 1] = Δlocal_data[i][1]
            backward_grad[2 * i] = Δlocal_data[i][2]
        end
        backward_grad[2 * k + 1] = Δeval_pt[1]
        backward_grad[2 * k + 2] = Δeval_pt[2]

        @test !all(iszero, backward_grad)
        @test isapprox(backward_grad, fd_grad; rtol = 1.0e-4)
    end
end
