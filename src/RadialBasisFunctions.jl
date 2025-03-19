module RadialBasisFunctions

using ChunkSplitters
using Combinatorics
using Distances
using LinearAlgebra
using NearestNeighbors
using SparseArrays
using StaticArraysCore
using SymRCM

include("basis/basis.jl")
export AbstractRadialBasis
export RadialBasisFunction
export AbstractPHS, PHS, PHS1, PHS3, PHS5, PHS7
export IMQ
export Gaussian
export MonomialBasis
export degree, dim

include("utils.jl")
export find_neighbors, reorder_points!

include("operators/operators.jl")
export RadialBasisOperator, ScalarValuedOperator, VectorValuedOperator
export update_weights!, is_cache_valid

include("solve.jl")

include("operators/custom.jl")
export Custom, custom

include("operators/partial.jl")
export Partial, partial

include("operators/laplacian.jl")
export Laplacian, laplacian

include("operators/gradient.jl")
export Gradient, gradient

include("operators/directional.jl")
export Directional, directional

include("operators/virtual.jl")
export ∂virtual

include("operators/monomial/monomial.jl")

include("operators/operator_algebra.jl")

include("interpolation.jl")
export Interpolator

include("operators/regridding.jl")
export Regrid, regrid

# Some consts and aliases
const Δ = ∇² # some people like this notation for the Laplacian
const AVOID_INF = 1e-16

using PrecompileTools
@setup_workload begin
    f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
    x = map(x -> SVector{2}(rand(2)), 1:100)
    z = f.(x)
    @compile_workload begin
        # basis functions
        basis_funcs = [
            IMQ(1),
            Gaussian(1),
            PHS(1; poly_deg=0),
            PHS(3; poly_deg=0),
            PHS(5; poly_deg=1),
            PHS(7; poly_deg=2),
        ]

        for b in basis_funcs
            # partial
            ∂ = partial(x, 1, 1, b)
            ∂x = ∂(z)

            # gradient
            ∇ = gradient(x, b)
            ∇z = ∇(z)

            # laplacian
            ∇² = laplacian(x, b)
            ∇²z = ∇²(z)

            # interpolation
            interp = Interpolator(x, z, b)
            zz = interp([SVector{2}(rand(2)), SVector{2}(rand(2))])
        end
    end
end

end # module
