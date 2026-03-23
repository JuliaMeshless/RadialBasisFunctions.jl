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
