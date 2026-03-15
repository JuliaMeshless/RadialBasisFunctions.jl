using RadialBasisFunctions
using LuxCore
using Random
using Test

const LuxExt = Base.get_extension(RadialBasisFunctions, :RadialBasisFunctionsLuxCoreExt)
using .LuxExt: RBFLayer, _rbf_activate, _softplus, _inverse_softplus, _pairwise_sq_euclidean

const rng = Random.MersenneTwister(42)

@testset "Construction" begin
    @testset "Valid basis types" begin
        for basis_type in (Gaussian, IMQ, PHS1, PHS3, PHS5, PHS7)
            l = RBFLayer(2, 10, 1; basis_type = basis_type)
            @test l.in_dims == 2
            @test l.num_centers == 10
            @test l.out_dims == 1
            @test l.basis_type === basis_type
        end
    end

    @testset "Pair syntax" begin
        l = RBFLayer((3 => 20) => 5)
        @test l.in_dims == 3
        @test l.num_centers == 20
        @test l.out_dims == 5
        @test l.basis_type === Gaussian  # default
    end

    @testset "Invalid inputs" begin
        @test_throws ArgumentError RBFLayer(2, 10, 1; init_shape = -1.0)
        @test_throws ArgumentError RBFLayer(2, 10, 1; init_shape = 0.0)
        @test_throws ArgumentError RBFLayer(0, 10, 1)
        @test_throws ArgumentError RBFLayer(2, 0, 1)
        @test_throws ArgumentError RBFLayer(2, 10, 0)
    end

    @testset "Default options" begin
        l = RBFLayer(2, 10, 1)
        @test l.use_bias == true
        @test l.learn_centers == true
        @test l.learn_shape == true
        @test l.init_shape == 1.0
    end
end

@testset "Parameter initialization" begin
    @testset "All learnable (Gaussian)" begin
        l = RBFLayer(3, 15, 4; basis_type = Gaussian)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)

        @test size(ps.weight) == (4, 15)
        @test size(ps.centers) == (3, 15)
        @test size(ps.log_shape) == (15,)
        @test size(ps.bias) == (4,)

        @test !haskey(st, :centers)
        @test !haskey(st, :log_shape)
    end

    @testset "Fixed centers and shape" begin
        l = RBFLayer(2, 10, 1; basis_type = IMQ, learn_centers = false, learn_shape = false)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)

        @test haskey(ps, :weight)
        @test !haskey(ps, :centers)
        @test !haskey(ps, :log_shape)

        @test haskey(st, :centers)
        @test haskey(st, :log_shape)
        @test size(st.centers) == (2, 10)
        @test size(st.log_shape) == (10,)
    end

    @testset "No bias" begin
        l = RBFLayer(2, 10, 1; use_bias = false)
        ps = LuxCore.initialparameters(rng, l)
        @test !haskey(ps, :bias)
    end

    @testset "PHS has no shape params" begin
        for basis_type in (PHS1, PHS3, PHS5, PHS7)
            l = RBFLayer(2, 10, 1; basis_type = basis_type)
            ps = LuxCore.initialparameters(rng, l)
            st = LuxCore.initialstates(rng, l)
            @test !haskey(ps, :log_shape)
            @test !haskey(st, :log_shape)
        end
    end
end

@testset "Forward pass" begin
    in_dims, num_centers, out_dims, batch_size = 3, 20, 4, 16

    @testset "All basis types produce correct output shape" begin
        for basis_type in (Gaussian, IMQ, PHS1, PHS3, PHS5, PHS7)
            l = RBFLayer(in_dims, num_centers, out_dims; basis_type = basis_type)
            ps = LuxCore.initialparameters(rng, l)
            st = LuxCore.initialstates(rng, l)
            x = randn(rng, Float32, in_dims, batch_size)

            y, st_out = l(x, ps, st)
            @test size(y) == (out_dims, batch_size)
            @test all(isfinite, y)
            @test st_out === st
        end
    end

    @testset "Single sample input (vector)" begin
        l = RBFLayer(2, 10, 3; basis_type = Gaussian)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        x = randn(rng, Float32, 2)

        y, _ = l(x, ps, st)
        @test size(y) == (3,)
        @test all(isfinite, y)
    end

    @testset "With and without bias" begin
        l_bias = RBFLayer(2, 10, 3; basis_type = Gaussian, use_bias = true)
        l_nobias = RBFLayer(2, 10, 3; basis_type = Gaussian, use_bias = false)
        ps_bias = LuxCore.initialparameters(rng, l_bias)
        ps_nobias = LuxCore.initialparameters(rng, l_nobias)
        st_bias = LuxCore.initialstates(rng, l_bias)
        st_nobias = LuxCore.initialstates(rng, l_nobias)

        x = randn(rng, Float32, 2, 8)

        y_bias, _ = l_bias(x, ps_bias, st_bias)
        y_nobias, _ = l_nobias(x, ps_nobias, st_nobias)

        @test size(y_bias) == (3, 8)
        @test size(y_nobias) == (3, 8)
    end

    @testset "Fixed centers produce correct output" begin
        l = RBFLayer(2, 10, 1; basis_type = Gaussian, learn_centers = false)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        x = randn(rng, Float32, 2, 5)

        y, _ = l(x, ps, st)
        @test size(y) == (1, 5)
        @test all(isfinite, y)
    end

    @testset "Fixed shape produces correct output" begin
        l = RBFLayer(2, 10, 1; basis_type = IMQ, learn_shape = false)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        x = randn(rng, Float32, 2, 5)

        y, _ = l(x, ps, st)
        @test size(y) == (1, 5)
        @test all(isfinite, y)
    end
end

@testset "Output dimensions" begin
    configs = [(1, 5, 1), (2, 10, 3), (5, 50, 10), (3, 100, 1)]
    for (d, n, o) in configs
        l = RBFLayer(d, n, o; basis_type = Gaussian)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        x = randn(rng, Float32, d, 8)
        y, _ = l(x, ps, st)
        @test size(y) == (o, 8)
    end
end

@testset "Activation math" begin
    D2 = Float32[0.0 1.0 4.0; 0.25 2.0 9.0]
    log_shape = Float32[0.5, 0.5]

    @testset "Gaussian bounded in (0, 1]" begin
        Phi = LuxExt._rbf_activate(Gaussian, D2, log_shape)
        @test all(0 .< Phi .<= 1)
        @test Phi[1, 1] ≈ 1.0 atol = 1.0e-6
    end

    @testset "IMQ bounded in (0, 1]" begin
        Phi = LuxExt._rbf_activate(IMQ, D2, log_shape)
        @test all(0 .< Phi .<= 1)
        @test Phi[1, 1] ≈ 1.0 atol = 1.0e-6
    end

    @testset "PHS monotonically increasing" begin
        D2_sorted = Float32[0.01 0.25 1.0 4.0]

        for basis_type in (PHS1, PHS3, PHS5, PHS7)
            Phi = LuxExt._rbf_activate(basis_type, D2_sorted, nothing)
            for j in 1:(size(Phi, 2) - 1)
                @test Phi[1, j] < Phi[1, j + 1]
            end
        end
    end
end

@testset "Softplus constraint" begin
    @testset "Negative log_shape still produces valid output" begin
        l = RBFLayer(2, 10, 1; basis_type = Gaussian)
        ps = LuxCore.initialparameters(rng, l)
        st = LuxCore.initialstates(rng, l)
        ps = (; ps..., log_shape = fill(Float32(-5.0), 10))
        x = randn(rng, Float32, 2, 8)
        y, _ = l(x, ps, st)
        @test all(isfinite, y)
    end

    @testset "Softplus invertibility" begin
        for val in [0.01f0, 0.5f0, 1.0f0, 5.0f0, 25.0f0]
            inv = LuxExt._inverse_softplus(val)
            recovered = LuxExt._softplus(inv)
            @test recovered ≈ val atol = 1.0e-5
        end
    end
end

@testset "Pairwise squared Euclidean" begin
    C = Float32[1.0 0.0; 0.0 1.0; 0.0 0.0]
    X = Float32[1.0 0.0; 0.0 1.0; 0.0 0.0]

    D2 = LuxExt._pairwise_sq_euclidean(C, X)
    @test size(D2) == (2, 2)
    @test D2[1, 1] ≈ 0.0 atol = 1.0e-6
    @test D2[2, 2] ≈ 0.0 atol = 1.0e-6
    @test D2[1, 2] ≈ 2.0 atol = 1.0e-6
    @test D2[2, 1] ≈ 2.0 atol = 1.0e-6

    @testset "Non-negative distances" begin
        @test all(D2 .>= 0)
    end
end

@testset "Pretty printing" begin
    l = RBFLayer(2, 10, 1; basis_type = Gaussian)
    str = sprint(show, l)
    @test contains(str, "RBFLayer")
    @test contains(str, "2 => 10 => 1")
    @test contains(str, "Gaussian")

    l2 = RBFLayer(3, 20, 5; basis_type = PHS3, use_bias = false)
    str2 = sprint(show, l2)
    @test contains(str2, "use_bias=false")
    @test contains(str2, "PHS3")

    l3 = RBFLayer(2, 10, 1; basis_type = IMQ, learn_shape = false)
    str3 = sprint(show, l3)
    @test contains(str3, "learn_shape=false")
end

if Base.find_package("Mooncake") !== nothing
    @testset "Gradient correctness (Mooncake)" begin
        using DifferentiationInterface
        using Mooncake

        backend = AutoMooncake(; config = nothing)

        @testset "Gaussian - all learnable" begin
            l = RBFLayer(2, 10, 1; basis_type = Gaussian)
            ps = LuxCore.initialparameters(rng, l)
            st = LuxCore.initialstates(rng, l)
            x = randn(rng, Float32, 2, 8)

            loss(ps) = sum(first(l(x, ps, st)))
            gs = DifferentiationInterface.gradient(loss, backend, ps)

            @test all(isfinite, gs.weight)
            @test all(isfinite, gs.centers)
            @test all(isfinite, gs.log_shape)
            @test all(isfinite, gs.bias)
        end

        @testset "IMQ - all learnable" begin
            l = RBFLayer(2, 10, 1; basis_type = IMQ)
            ps = LuxCore.initialparameters(rng, l)
            st = LuxCore.initialstates(rng, l)
            x = randn(rng, Float32, 2, 8)

            loss(ps) = sum(first(l(x, ps, st)))
            gs = DifferentiationInterface.gradient(loss, backend, ps)

            @test all(isfinite, gs.weight)
            @test all(isfinite, gs.centers)
            @test all(isfinite, gs.log_shape)
            @test all(isfinite, gs.bias)
        end

        @testset "PHS basis types" begin
            for basis_type in (PHS1, PHS3, PHS5, PHS7)
                l = RBFLayer(2, 10, 1; basis_type = basis_type)
                ps = LuxCore.initialparameters(rng, l)
                st = LuxCore.initialstates(rng, l)
                x = randn(rng, Float32, 2, 8)

                loss(ps) = sum(first(l(x, ps, st)))
                gs = DifferentiationInterface.gradient(loss, backend, ps)

                @test all(isfinite, gs.weight)
                @test all(isfinite, gs.bias)
            end
        end

        @testset "Fixed centers" begin
            l = RBFLayer(2, 10, 1; basis_type = Gaussian, learn_centers = false)
            ps = LuxCore.initialparameters(rng, l)
            st = LuxCore.initialstates(rng, l)
            x = randn(rng, Float32, 2, 8)

            loss(ps) = sum(first(l(x, ps, st)))
            gs = DifferentiationInterface.gradient(loss, backend, ps)

            @test all(isfinite, gs.weight)
            @test all(isfinite, gs.log_shape)
            @test all(isfinite, gs.bias)
        end

        @testset "Fixed shape" begin
            l = RBFLayer(2, 10, 1; basis_type = IMQ, learn_shape = false)
            ps = LuxCore.initialparameters(rng, l)
            st = LuxCore.initialstates(rng, l)
            x = randn(rng, Float32, 2, 8)

            loss(ps) = sum(first(l(x, ps, st)))
            gs = DifferentiationInterface.gradient(loss, backend, ps)

            @test all(isfinite, gs.weight)
            @test all(isfinite, gs.centers)
            @test all(isfinite, gs.bias)
        end

        @testset "No bias" begin
            l = RBFLayer(2, 10, 1; basis_type = Gaussian, use_bias = false)
            ps = LuxCore.initialparameters(rng, l)
            st = LuxCore.initialstates(rng, l)
            x = randn(rng, Float32, 2, 8)

            loss(ps) = sum(first(l(x, ps, st)))
            gs = DifferentiationInterface.gradient(loss, backend, ps)

            @test all(isfinite, gs.weight)
            @test all(isfinite, gs.centers)
            @test all(isfinite, gs.log_shape)
        end
    end
end
