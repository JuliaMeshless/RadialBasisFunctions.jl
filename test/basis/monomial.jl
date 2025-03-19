using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore

@testset "Construction and Printing" begin
    @test_throws ArgumentError MonomialBasis(1, -1)
    m = MonomialBasis(1, 0)
    @test repr(m) == "MonomialBasis of degree 0 in 1 dimensions"
end

@testset "dim=1, deg=0" begin
    inputs = (SVector(2.0), 2.0)
    foreach(inputs) do x
        m = MonomialBasis(1, 0)
        @test m isa MonomialBasis
        @test typeof(m) <: MonomialBasis{1,0}

        # standard evaluation
        @test all(isapprox.(m(x), [1]))

        # derivatives
        @test all(isapprox.(RBF.∂(m, 1)(x), [0]))
        @test all(isapprox.(RBF.∂²(m, 1)(x), [0]))
        @test all(isapprox.(RBF.∇²(m)(x), [0]))
    end
end

@testset "dim=1, deg=1" begin
    inputs = (SVector(2.0), 2.0)

    foreach(inputs) do x
        m = MonomialBasis(1, 1)
        @test m isa MonomialBasis
        @test typeof(m) <: MonomialBasis{1,1}

        # standard evaluation
        @test all(isapprox.(m(x), [1, 2]))

        # in-place evaluation
        b = ones(_get_underlying_type(x), 2)
        m(b, x)
        @test all(isapprox.(b, [1, 2]))

        # derivatives
        @test all(isapprox.(RBF.∂(m, 1)(x), [0, 1]))
        @test all(isapprox.(RBF.∂²(m, 1)(x), [0, 0]))
        @test all(isapprox.(RBF.∇²(m)(x), [0, 0]))
    end
end

@testset "dim=1, deg=2" begin
    inputs = (SVector(2.0), 2.0)

    foreach(inputs) do x
        m = MonomialBasis(1, 2)
        @test m isa MonomialBasis
        @test typeof(m) <: MonomialBasis{1,2}

        # standard evaluation
        @test all(isapprox.(m(x), [1, 2, 4]))

        # derivatives
        @test all(isapprox.(RBF.∂(m, 1)(x), [0, 1, 4]))
        @test all(isapprox.(RBF.∂²(m, 1)(x), [0, 0, 2]))
        @test all(isapprox.(RBF.∇²(m)(x), [0, 0, 2]))
    end
end

@testset "dim=2, deg=0" begin
    x = SVector(2.0, 3.0)

    m = MonomialBasis(2, 0)
    @test m isa MonomialBasis
    @test typeof(m) <: MonomialBasis{2,0}

    # standard evaluation
    @test all(isapprox.(m(x), [1]))

    # derivatives
    @test all(isapprox.(RBF.∂(m, 1)(x), [0]))
    @test all(isapprox.(RBF.∂(m, 2)(x), [0]))
    @test all(isapprox.(RBF.∂²(m, 1)(x), [0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0]))
    @test all(isapprox.(RBF.∇²(m)(x), [0]))
end

@testset "dim=2, deg=1" begin
    x = SVector(2.0, 3.0)

    m = MonomialBasis(2, 1)
    @test m isa MonomialBasis
    @test typeof(m) <: MonomialBasis{2,1}

    # standard evaluation
    @test all(isapprox.(m(x), [1, 2, 3]))

    # derivatives
    @test all(isapprox.(RBF.∂(m, 1)(x), [0, 1, 0]))
    @test all(isapprox.(RBF.∂(m, 2)(x), [0, 0, 1]))
    @test all(isapprox.(RBF.∂²(m, 1)(x), [0, 0, 0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0, 0, 0]))
    @test all(isapprox.(RBF.∇²(m)(x), [0, 0, 0]))
end

@testset "dim=2, deg=2" begin
    x = SVector(2.0, 3.0)

    m = MonomialBasis(2, 2)
    @test m isa MonomialBasis
    @test typeof(m) <: MonomialBasis{2,2}

    # standard evaluation
    @test all(isapprox.(m(x), [1, 2, 3, 6, 4, 9]))

    # derivatives
    @test all(isapprox.(RBF.∂(m, 1)(x), [0, 1, 0, 3, 4, 0]))
    @test all(isapprox.(RBF.∂(m, 2)(x), [0, 0, 1, 2, 0, 6]))
    @test all(isapprox.(RBF.∂²(m, 1)(x), [0, 0, 0, 0, 2, 0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0, 0, 0, 0, 0, 2]))
    @test all(isapprox.(RBF.∇²(m)(x), [0, 0, 0, 0, 2, 2]))
end

@testset "dim=2, deg=3 - fallback for higher orders" begin
    x = SVector(2.0, 3.0)

    m = MonomialBasis(2, 3)
    @test m isa MonomialBasis
    @test typeof(m) <: MonomialBasis{2,3}

    # standard evaluation
    @test all(isapprox.(m(x), [8, 12, 4, 18, 6, 2, 27, 9, 3, 1]))

    # derivatives
    @test all(isapprox.(RBF.∂(m, 1)(x), [12, 12, 4, 9, 3, 1, 0, 0, 0, 0]))
    @test all(isapprox.(RBF.∂(m, 2)(x), [0, 4, 0, 12, 2, 0, 27, 6, 1, 0]))
    @test all(isapprox.(RBF.∂²(m, 1)(x), [12, 6, 2, 0, 0, 0, 0, 0, 0, 0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0, 0, 0, 4, 0, 0, 18, 2, 0, 0]))
    @test all(isapprox.(RBF.∇²(m)(x), [12, 6, 2, 4, 0, 0, 18, 2, 0, 0]))
end

@testset "dim=3, deg=0" begin
    x = SVector(2.0, 3.0, 4.0)

    m = MonomialBasis(3, 0)
    @test m isa MonomialBasis
    @test typeof(m) <: MonomialBasis{3,0}

    # standard evaluation
    @test all(isapprox.(m(x), [1]))

    # derivatives
    @test all(isapprox.(RBF.∂(m, 1)(x), [0]))
    @test all(isapprox.(RBF.∂(m, 2)(x), [0]))
    @test all(isapprox.(RBF.∂(m, 3)(x), [0]))
    @test all(isapprox.(RBF.∂²(m, 1)(x), [0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0]))
    @test all(isapprox.(RBF.∇²(m)(x), [0]))
end

@testset "dim=3, deg=1" begin
    x = SVector(2.0, 3.0, 4.0)

    m = MonomialBasis(3, 1)
    @test m isa MonomialBasis
    @test typeof(m) <: MonomialBasis{3,1}

    # standard evaluation
    @test all(isapprox.(m(x), [1, 2, 3, 4]))

    # derivatives
    @test all(isapprox.(RBF.∂(m, 1)(x), [0, 1, 0, 0]))
    @test all(isapprox.(RBF.∂(m, 2)(x), [0, 0, 1, 0]))
    @test all(isapprox.(RBF.∂(m, 3)(x), [0, 0, 0, 1]))
    @test all(isapprox.(RBF.∂²(m, 1)(x), [0, 0, 0, 0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0, 0, 0, 0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0, 0, 0, 0]))
    @test all(isapprox.(RBF.∇²(m)(x), [0, 0, 0, 0]))
end

@testset "dim=3, deg=2" begin
    x = SVector(2.0, 3.0, 4.0)

    m = MonomialBasis(3, 2)
    @test m isa MonomialBasis
    @test typeof(m) <: MonomialBasis{3,2}

    # standard evaluation
    @test all(isapprox.(m(x), [1, 2, 3, 4, 6, 8, 12, 4, 9, 16]))

    # derivatives
    @test all(isapprox.(RBF.∂(m, 1)(x), [0, 1, 0, 0, 3, 4, 0, 4, 0, 0]))
    @test all(isapprox.(RBF.∂(m, 2)(x), [0, 0, 1, 0, 2, 0, 4, 0, 6, 0]))
    @test all(isapprox.(RBF.∂(m, 3)(x), [0, 0, 0, 1, 0, 2, 3, 0, 0, 8]))
    @test all(isapprox.(RBF.∂²(m, 1)(x), [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]))
    @test all(isapprox.(RBF.∂²(m, 2)(x), [0, 0, 0, 0, 0, 0, 0, 0, 2, 0]))
    @test all(isapprox.(RBF.∂²(m, 3)(x), [0, 0, 0, 0, 0, 0, 0, 0, 0, 2]))
    @test all(isapprox.(RBF.∇²(m)(x), [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]))
end
