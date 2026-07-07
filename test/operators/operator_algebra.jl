using RadialBasisFunctions
using StaticArraysCore
using LinearAlgebra
using Statistics
using HaltonSequences

include("../test_utils.jl")

f(x) = 2 * x[1] + 3 * x[2]
df_dx(x) = 2
df_dy(x) = 3

N = 1000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

dx = partial(x, 1, 1)
dy = partial(x, 1, 2)

dxdy = dx + dy
@test mean_percent_error(dxdy(y), df_dx.(x) .+ df_dy.(x)) < 1.0e-6

dxdy = dx - dy
@test mean_percent_error(dxdy(y), df_dx.(x) .- df_dy.(x)) < 1.0e-6

# test compatibility with other operators
dy = partial(x[1:500], 1, 2)
@test_throws ArgumentError dx + dy

adjl = copy(dx.adjl)
adjl[1] = dx.adjl[2]
adjl[2] = dx.adjl[1]
dy = partial(x, 1, 2; adjl = adjl)
@test_throws ArgumentError dx + dy

# test rank-mismatched algebra gives clear error
@testset "Rank mismatch error" begin
    @test_throws ArgumentError Laplacian() + Jacobian{2}()
    @test_throws ArgumentError Partial(1, 1) - Hessian{2}()
end

# ScaledOperator algebra
@testset "ScaledOperator" begin
    p = Partial(1, 1)
    # right-multiply: op * α
    s = p * 3.0
    @test s isa RadialBasisFunctions.ScaledOperator
    @test s.α == 3.0
    # unary negation
    n = -p
    @test n isa RadialBasisFunctions.ScaledOperator
    @test n.α == -1
    # division by scalar
    d = p / 4.0
    @test d isa RadialBasisFunctions.ScaledOperator
    @test d.α ≈ 0.25
    # print_op
    @test RadialBasisFunctions.print_op(s) == "3.0 × ∂ⁿf/∂xᵢ (n = 1, i = 1)"
    @test RadialBasisFunctions.print_op(Identity()) == "Identity (f)"
end

# ============================================================================
# Structure-preserving algebra (SumOperator / typed kernels)
# ============================================================================

pts = SVector{2}.(HaltonPoint(2)[1:200])
vfx(p) = 2 * p[1] + 3 * p[2]
vfy(p) = p[1] - p[2]
quadratic(p) = p[1]^2 + p[2]^2

@testset "SumOperator structure" begin
    s = Partial(1, 1) + Partial(1, 2)
    @test s isa SumOperator
    @test length(s.ops) == 2
    # nested sums flatten into n-ary term tuples
    s3 = s + Laplacian()
    @test s3 isa SumOperator
    @test length(s3.ops) == 3
    # subtraction stores -1 × op
    sub = Partial(1, 1) - Partial(1, 2)
    @test sub isa SumOperator
    @test sub.ops[2] isa RadialBasisFunctions.ScaledOperator
    @test sub.ops[2].α == -1
end

@testset "Gradient-family sum: Divergence + Divergence" begin
    v = hcat(vfx.(pts), vfy.(pts))
    dsum = Divergence{2}() + Divergence{2}()
    @test dsum isa SumOperator
    op = dsum(pts)
    single = divergence(pts)
    @test isapprox(op(v), 2 .* single(v); rtol = 1.0e-10)
    # ∇⋅(2x + 3y, x − y) = 1, doubled = 2
    @test isapprox(op(v), fill(2.0, length(pts)); rtol = 1.0e-6)
end

@testset "Scaled gradient-family: 2.0 * Jacobian" begin
    scaled = 2.0 * Jacobian{2}()
    @test scaled isa RadialBasisFunctions.ScaledOperator
    op = scaled(pts)
    single = jacobian(pts)
    q = quadratic.(pts)
    @test isapprox(op(q), 2.0 .* single(q); rtol = 1.0e-10)
end

@testset "Gradient-family subtraction cancels" begin
    v = hcat(vfx.(pts), vfy.(pts))
    dzero = (Divergence{2}() - Divergence{2}())(pts)
    @test all(abs.(dzero(v)) .< 1.0e-8)
end

@testset "Scalar sums match weight-combining path" begin
    lap_sum = (Laplacian() + Laplacian())(pts)
    combined = laplacian(pts) + laplacian(pts)
    @test isapprox(Matrix(lap_sum.weights), Matrix(combined.weights); rtol = 1.0e-10)
    q = quadratic.(pts)
    # ∇²(x² + y²) = 4, doubled = 8
    @test isapprox(lap_sum(q), fill(8.0, length(pts)); rtol = 1.0e-6)
end

@testset "Composed trait propagation" begin
    @test derivative_order(Partial(1, 1) + Partial(2, 1)) == 2
    @test derivative_order(2.0 * (Partial(1, 1) + Partial(2, 2))) == 2
    # Custom's derivative_order is missing; it must propagate through composition
    @test ismissing(derivative_order(Laplacian() + Custom{0}(b -> (x, xᵢ) -> b(x, xᵢ))))
    @test ismissing(derivative_order(Partial(1, 1) + 2.0 * Custom{0}(b -> (x, xᵢ) -> b(x, xᵢ))))
    @test is_self_adjoint(Laplacian() + Identity())
    @test is_self_adjoint(2.0 * Laplacian())
    @test !is_self_adjoint(Laplacian() + Partial(1, 1))
end

@testset "@operator trees are structure-preserving" begin
    k2 = 1.7
    helm = @operator ∇² + k2 * f
    @test helm isa SumOperator
    @test length(helm.ops) == 2
    @test helm.ops[2] isa RadialBasisFunctions.ScaledOperator
    aniso = @operator 2.0 * ∂²(1) + 0.5 * ∂²(2)
    @test aniso isa SumOperator
    op = helm(pts)
    hand = Matrix(laplacian(pts).weights) .+ k2 .* Matrix(Identity()(pts).weights)
    @test isapprox(Matrix(op.weights), hand; rtol = 1.0e-10)
end

@testset "Composed rank mismatch and mixed-action errors" begin
    @test_throws ArgumentError (Laplacian() + Laplacian()) + Jacobian{2}()
    @test_throws ArgumentError 2.0 * Laplacian() + Jacobian{2}()
    # scalar action + per-dimension tuple action cannot combine at weight build
    @test_throws ArgumentError (Laplacian() + Divergence{2}())(pts)
end
