using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using Statistics
using HaltonSequences

include("../test_utils.jl")

# Setup
N = 1000
x = SVector{2}.(HaltonPoint(2)[1:N])

@testset "Custom struct" begin
    c = Custom(identity)
    @test c(1) == 1
end

@testset "Printing" begin
    c = Custom(identity)
    @test RadialBasisFunctions.print_op(c) == "Custom Operator"
end

@testset "custom() Keyword Constructor" begin
    # Test custom.jl lines 38-40: primary keyword constructor
    op = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ))
    @test op isa RadialBasisOperator

    # Test with explicit keyword arguments
    op2 = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ); basis=PHS(5; poly_deg=3))
    @test op2 isa RadialBasisOperator
end

@testset "custom() Positional Basis Constructor" begin
    # Test custom.jl lines 43-45: backward compatible positional basis
    op = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ), PHS(3; poly_deg=2))
    @test op isa RadialBasisOperator

    # Test with different basis types
    op_imq = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ), IMQ(1; poly_deg=2))
    @test op_imq isa RadialBasisOperator
end

@testset "custom() Different Eval Points" begin
    # Test custom.jl lines 47-55: separate evaluation points
    x2 = SVector{2}.(HaltonPoint(2)[(N + 1):(N + 100)])
    op = custom(x, x2, basis -> (x, xᵢ) -> basis(x, xᵢ))
    @test op isa RadialBasisOperator
    @test length(op.eval_points) == 100

    # With explicit basis
    op2 = custom(x, x2, basis -> (x, xᵢ) -> basis(x, xᵢ), PHS(5; poly_deg=3))
    @test op2 isa RadialBasisOperator
end

@testset "custom() Hermite Boundary Conditions" begin
    # Test custom.jl lines 58-72: Hermite interpolation with boundary conditions
    # Create a simple 1D domain for testing
    spacing = 0.1
    domain = [SVector{1}(x) for x in 0.0:spacing:1.0]
    N_domain = length(domain)

    # Identify boundary points (first and last)
    is_boundary = zeros(Bool, N_domain)
    is_boundary[1] = true
    is_boundary[end] = true

    # Set up boundary conditions (Dirichlet at both ends)
    boundary_conditions = [RBF.Dirichlet(), RBF.Dirichlet()]

    # Normals for 1D (outward pointing)
    normals = [SVector(-1.0), SVector(1.0)]

    # Test that the constructor works
    op = custom(
        domain,
        domain,
        basis -> (x, xᵢ) -> basis(x, xᵢ),
        PHS(3; poly_deg=2),
        is_boundary,
        boundary_conditions,
        normals,
    )
    @test op isa RadialBasisOperator
end
