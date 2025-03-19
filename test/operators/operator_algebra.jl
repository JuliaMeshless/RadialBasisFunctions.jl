using RadialBasisFunctions
using StaticArraysCore
using LinearAlgebra
using Statistics
using HaltonSequences

mean_percent_error(test, correct) = mean(abs.((test .- correct) ./ correct)) * 100

f(x) = 2 * x[1] + 3 * x[2]
df_dx(x) = 2
df_dy(x) = 3

N = 1000
x = SVector{2}.(HaltonPoint(2)[1:N])
y = f.(x)

dx = partial(x, 1, 1)
dy = partial(x, 1, 2)

dxdy = dx + dy

##

@test mean_percent_error(dxdy(y), df_dx.(x) .+ df_dy.(x)) < 1e-6

dxdy = dx - dy
@test mean_percent_error(dxdy(y), df_dx.(x) .- df_dy.(x)) < 1e-6
