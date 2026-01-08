using BenchmarkTools
using RadialBasisFunctions
using StaticArraysCore
using HaltonSequences
using LinearAlgebra

f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
N = 100_000
x = SVector{3}.(HaltonPoint(3)[1:N])
y = f.(x)

const SUITE = BenchmarkGroup()

basis = PHS(3; poly_deg = 2)

∂x = partial(x, 1, 1, basis)
SUITE["Partial"] = let s = BenchmarkGroup()
    s["build weights"] = @benchmarkable update_weights!($∂x)
    s["eval"] = @benchmarkable ∂x($y)
end

grad = gradient(x, basis)
SUITE["Gradient"] = let s = BenchmarkGroup()
    s["build weights"] = @benchmarkable update_weights!($grad)
    s["eval"] = @benchmarkable grad($y)
end

v = SVector(2.0, 1.0, 0.5)
v /= norm(v)
∇v = directional(x, v, basis)
SUITE["Directional"] = let s = BenchmarkGroup()
    s["build weights"] = @benchmarkable update_weights!($∇v)
    s["eval"] = @benchmarkable ∇v($y)
end

v = map(1:length(x)) do i
    v = SVector{3}(rand(3))
    return v /= norm(v)
end
∇v = directional(x, v, basis)
SUITE["Directional (per point)"] = let s = BenchmarkGroup()
    s["build weights"] = @benchmarkable update_weights!($∇v)
    s["eval"] = @benchmarkable ∇v($y)
end

x1 = SVector{3}(rand(3))
x2 = SVector{3}(rand(3))

function benchmark_basis(SUITE, basis, poly_deg, x1, x2)
    SUITE["RBF"]["$basis"]["$poly_deg"]["∂"] = @benchmarkable rbf($x1, $x2) setup = (
        rbf = RadialBasisFunctions.∂($basis, 1)
    )
    SUITE["RBF"]["$basis"]["$poly_deg"]["∂²"] = @benchmarkable rbf($x1, $x2) setup = (
        rbf = RadialBasisFunctions.∂²($basis, 1)
    )
    SUITE["RBF"]["$basis"]["$poly_deg"]["∇"] = @benchmarkable rbf($x1, $x2) setup = (
        rbf = RadialBasisFunctions.∇($basis)
    )
    return SUITE["RBF"]["$basis"]["$poly_deg"]["∇²"] = @benchmarkable rbf($x1, $x2) setup = (
        rbf = RadialBasisFunctions.∇²($basis)
    )
end

for poly_deg in 0:2
    for basis in (IMQ, Gaussian)
        rbf = basis(; poly_deg = poly_deg)
        benchmark_basis(SUITE, rbf, poly_deg, x1, x2)
    end
    for basis in (PHS1, PHS3, PHS5, PHS7)
        rbf = basis(poly_deg)
        benchmark_basis(SUITE, rbf, poly_deg, x1, x2)
    end
end

for dim in 1:3, deg in 0:2
    b = zeros(binomial(dim + deg, dim))
    SUITE["MonomialBasis"]["dim=$dim"]["deg=$deg"] = @benchmarkable mon($b, $x1) setup = (
        mon = MonomialBasis($dim, $deg)
    )
end
