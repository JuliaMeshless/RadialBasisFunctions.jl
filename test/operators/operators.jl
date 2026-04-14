using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using LinearAlgebra
using Statistics
using HaltonSequences
using Random: MersenneTwister

rng = MersenneTwister(123)

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
    y = rand(rng, N)
    z = rand(rng, N)
    ∂(y, z)
    @test y ≈ ∂.weights * z

    ∇ = gradient(x, PHS(3; poly_deg = 2))
    y_mat = Matrix{Float64}(undef, N, 2)
    ∇(y_mat, z)
    @test y_mat[:, 1] ≈ ∇.weights[1] * z
    @test y_mat[:, 2] ≈ ∇.weights[2] * z

    @test ∇ ⋅ z ≈ (∇.weights[1] * z) .+ (∇.weights[2] * z)
end

@testset "Operator Update" begin
    ∂ = partial(x, 1, 1)
    correct_weights = copy(∂.weights)
    ∂.weights .= rand(rng, size(∂.weights))
    update_weights!(∂)
    @test ∂.weights ≈ correct_weights
    @test is_cache_valid(∂)

    ∇ = gradient(x, PHS(3; poly_deg = 2))
    correct_weights = copy.(∇.weights)
    ∇.weights[1] .= rand(rng, size(∇.weights[1]))
    ∇.weights[2] .= rand(rng, size(∇.weights[2]))
    update_weights!(∇)
    @test ∇.weights[1] ≈ correct_weights[1]
    @test ∇.weights[2] ≈ correct_weights[2]
    @test is_cache_valid(∇)
end

@testset "RadialBasisOperator Constructors" begin
    # Test positional eval_points + basis constructor (operators.jl lines 114-125)
    x2 = SVector{2}.(HaltonPoint(2)[101:150])
    op = RadialBasisOperator(Partial(1, 1), x, x2, PHS(3; poly_deg = 2))
    @test op isa RadialBasisOperator
    @test length(op.eval_points) == 50
end

@testset "dim()" begin
    # Test dim() function (operators.jl line 145)
    ∂ = partial(x, 1, 1)
    @test RBF.dim(∂) == 2

    # 3D test
    x3d = [SVector{3}(rand(rng, 3)) for _ in 1:50]
    ∂3d = partial(x3d, 1, 1)
    @test RBF.dim(∂3d) == 3
end

@testset "LinearAlgebra.mul! interface" begin
    # Rank-0: mul!(y, op, x)
    ∂ = partial(x, 1, 1)
    z = rand(rng, N)
    y1 = similar(z)
    y2 = similar(z)
    ∂(y1, z)
    mul!(y2, ∂, z)
    @test y1 ≈ y2

    # Rank-0: 5-arg mul!(y, op, x, α, β)
    α, β = 2.5, -0.3
    y3 = rand(rng, N)
    y_ref = copy(y3)
    mul!(y3, ∂, z, α, β)
    @test y3 ≈ α * (∂.weights * z) + β * y_ref

    # Laplacian (also rank-0)
    ∇² = laplacian(x)
    y4 = similar(z)
    y5 = similar(z)
    ∇²(y4, z)
    mul!(y5, ∇², z)
    @test y4 ≈ y5

    # size and eltype
    @test size(∂) == size(∂.weights)
    @test eltype(∂) == eltype(∂.weights)
end

@testset "Printing" begin
    ∂ = partial(x, 1, 1)
    r = repr(∂)
    @test contains(r, "RadialBasisOperator")
    @test contains(r, "├─Operator: ∂ⁿf/∂xᵢ (n = 1, i = 1)")
    @test contains(r, "SVector{2, Float64}")
    @test contains(r, "├─Number of points: 100")
    @test contains(r, "├─Stencil size: 12")
    @test contains(r, "└─Basis: Polyharmonic spline (r³) with degree 2 polynomial augmentation")

    @test RBF.print_op(∂.ℒ) == "∂ⁿf/∂xᵢ (n = 1, i = 1)"
end
