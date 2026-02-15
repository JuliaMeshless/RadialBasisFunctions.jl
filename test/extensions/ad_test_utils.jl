"""
Shared test utilities for AD extension tests (Enzyme + Mooncake).

Provides common data setup, loss function generators, and FD validation helpers
to eliminate duplication between enzyme_ext.jl and mooncake_ext.jl.
"""

using RadialBasisFunctions
using StaticArraysCore
using FiniteDifferences
using LinearAlgebra
using Test

const FD = FiniteDifferences

# =============================================================================
# Data Setup
# =============================================================================

"""
    make_operator_test_data(; N=50)

Create shared test data for operator differentiation tests.
Returns (points, N, values).
"""
function make_operator_test_data(; N=50)
    points = [SVector{2}(0.1 + 0.8 * i / N, 0.1 + 0.8 * j / N) for i in 1:isqrt(N) for j in 1:isqrt(N)]
    N = length(points)
    values = sin.(getindex.(points, 1)) .+ cos.(getindex.(points, 2))
    return points, N, values
end

"""
    make_build_weights_test_data(; N=25, k=10)

Create shared test data for _build_weights differentiation tests.
Returns (points, N, adjl, pts_flat).
"""
function make_build_weights_test_data(; N=25, k=10)
    points = [SVector{2}(0.1 + 0.8 * i / 5, 0.1 + 0.8 * j / 5) for i in 1:5 for j in 1:5]
    adjl = RadialBasisFunctions.find_neighbors(points, k)
    pts_flat = vcat([collect(p) for p in points]...)
    return points, N, adjl, pts_flat
end

# =============================================================================
# Loss Function Generators
# =============================================================================

"""
    make_build_weights_loss(ℒ, adjl, basis, N)

Generate a loss function over flattened points for _build_weights differentiation.
"""
function make_build_weights_loss(ℒ, adjl, basis, N)
    return function(pts)
        pts_vec = [SVector{2}(pts[2 * i - 1], pts[2 * i]) for i in 1:N]
        W = RadialBasisFunctions._build_weights(ℒ, pts_vec, pts_vec, adjl, basis)
        return sum(W.nzval .^ 2)
    end
end

"""
    make_eps_loss(ℒ, points, adjl, BasisType)

Generate a loss function over shape parameter ε for basis optimization tests.
"""
function make_eps_loss(ℒ, points, adjl, BasisType)
    return function(ε)
        basis = BasisType(ε; poly_deg=2)
        W = RadialBasisFunctions._build_weights(ℒ, points, points, adjl, basis)
        return sum(W.nzval .^ 2)
    end
end

# =============================================================================
# Validation Helpers
# =============================================================================

"""
    validate_gradient(ad_grad, loss, input; rtol=1e-4, check_nonzero=true)

Compare AD gradient against finite differences. Asserts non-zero and approximate match.
"""
function validate_gradient(ad_grad, loss, input; rtol=1e-4, check_nonzero=true)
    fd_grad = FD.grad(FD.central_fdm(5, 1), loss, input)[1]
    if check_nonzero
        @test !all(iszero, ad_grad)
    end
    @test isapprox(ad_grad, fd_grad; rtol=rtol)
end

"""
    validate_scalar_gradient(ad_val, loss, input; rtol=1e-3, check_nonzero=true)

Compare AD scalar derivative against finite differences.
"""
function validate_scalar_gradient(ad_val, loss, input; rtol=1e-3, check_nonzero=true)
    fd_val = FD.central_fdm(5, 1)(loss, input)
    if check_nonzero
        @test !iszero(ad_val)
    end
    @test isapprox(ad_val, fd_val; rtol=rtol)
end
