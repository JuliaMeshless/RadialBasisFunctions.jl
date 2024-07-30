using RadialBasisFunctions
using StaticArrays
using Statistics
using Random

rsme(test, correct) = sqrt(sum((test - correct) .^ 2) / sum(correct .^ 2))
mean_percent_error(test, correct) = mean(abs.((test .- correct) ./ correct)) * 100

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
N = 100
Δ = 1 / (N - 1)
points = 0:Δ:1
structured_points = ((x, y) for x in points for y in points)
x = map(x -> SVector{2}(x .+ (Δ / 5 .* rand(2))), structured_points)
y = f.(x)

x2 = map(x -> SVector{2}(rand(2)), 1:100)
r = regrid(x, x2, PHS(3; poly_deg=2))
@test mean_percent_error(r(y), f.(x2)) < 0.1
