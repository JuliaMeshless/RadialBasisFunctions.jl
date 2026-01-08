using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra
using Statistics
using HaltonSequences

N = 100
x = SVector{2}.(HaltonPoint(2)[1:N])

@testset "Base Methods" begin
    ∂ = partial(x, 1, 1)
    @test is_cache_valid(∂)
    RBF.invalidate_cache!(∂)
    @test !is_cache_valid(∂)
end

@testset "Operator Evaluation" begin
    ∂ = partial(x, 1, 1)
    y = rand(N)
    z = rand(N)
    ∂(y, z)
    @test y ≈ ∂.weights * z

    ∇ = gradient(x, PHS(3; poly_deg=2))
    y_mat = Matrix{Float64}(undef, N, 2)
    ∇(y_mat, z)
    @test y_mat[:, 1] ≈ ∇.weights[1] * z
    @test y_mat[:, 2] ≈ ∇.weights[2] * z

    @test ∇ ⋅ z ≈ (∇.weights[1] * z) .+ (∇.weights[2] * z)
end

@testset "Operator Update" begin
    ∂ = partial(x, 1, 1)
    correct_weights = copy(∂.weights)
    ∂.weights .= rand(size(∂.weights))
    update_weights!(∂)
    @test ∂.weights ≈ correct_weights
    @test is_cache_valid(∂)

    ∇ = gradient(x, PHS(3; poly_deg=2))
    correct_weights = copy.(∇.weights)
    ∇.weights[1] .= rand(size(∇.weights[1]))
    ∇.weights[2] .= rand(size(∇.weights[2]))
    update_weights!(∇)
    @test ∇.weights[1] ≈ correct_weights[1]
    @test ∇.weights[2] ≈ correct_weights[2]
    @test is_cache_valid(∇)
end

@testset "RadialBasisOperator Constructors" begin
    # Test positional eval_points + basis constructor (operators.jl lines 114-125)
    x2 = SVector{2}.(HaltonPoint(2)[101:150])
    op = RadialBasisOperator(Partial(1, 1), x, x2, PHS(3; poly_deg=2))
    @test op isa RadialBasisOperator
    @test length(op.eval_points) == 50
end

@testset "dim()" begin
    # Test dim() function (operators.jl line 145)
    ∂ = partial(x, 1, 1)
    @test RBF.dim(∂) == 2

    # 3D test
    x3d = [SVector{3}(rand(3)) for _ in 1:50]
    ∂3d = partial(x3d, 1, 1)
    @test RBF.dim(∂3d) == 3
end

@testset "Printing" begin
    ∂ = partial(x, 1, 1)
    @test repr(∂) == """
RadialBasisOperator
├─Operator: ∂ⁿf/∂xᵢ (n = 1, i = 1)
├─Data type: StaticArraysCore.SVector{2, Float64}
├─Number of points: 100
├─Stencil size: 12
└─Basis: Polyharmonic spline (r³) with degree 2 polynomial augmentation
"""

    @test RBF.print_op(∂.ℒ) == "∂ⁿf/∂xᵢ (n = 1, i = 1)"
end
