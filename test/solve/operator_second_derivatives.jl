"""
Unit tests for the applied-operator second derivatives used by the AD backward pass
(src/solve/operator_second_derivatives.jl).

The gradients of ℒφ are hand-written per basis, so each is pinned against
ForwardDiff of the corresponding ∇² functor — in both 2D and 3D, since the
PHS Laplacian constants are dimension-dependent (Δrᵏ = k(k+d−2)·r^(k−2)).
"""

using Test
using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
import ForwardDiff as FD

@testset "grad_applied_laplacian vs ForwardDiff of ∇²" begin
    bases = (
        PHS(1; poly_deg = -1),
        PHS(3; poly_deg = -1),
        PHS(5; poly_deg = -1),
        PHS(7; poly_deg = -1),
        IMQ(1.0),
        Gaussian(1.0),
    )
    for basis in bases
        grad_x = RBF.grad_applied_laplacian_wrt_x(basis)
        grad_xi = RBF.grad_applied_laplacian_wrt_xi(basis)
        for (x, xᵢ) in (
                (SVector(1.0, 2.0), SVector(2.0, 4.0)),           # 2D
                (SVector(1.0, 2.0, 3.0), SVector(2.0, 4.0, 1.0)), # 3D
            )
            fd = FD.gradient(y -> RBF.∇²(basis)(y, xᵢ), x)
            @testset "$(nameof(typeof(basis))) $(length(x))D" begin
                @test grad_x(x, xᵢ) ≈ fd rtol = 1.0e-8
                @test grad_xi(x, xᵢ) ≈ -fd rtol = 1.0e-8
            end
        end
    end
end

@testset "grad_applied_partial vs ForwardDiff of ∂" begin
    bases = (
        PHS(1; poly_deg = -1),
        PHS(3; poly_deg = -1),
        IMQ(1.0),
        Gaussian(1.0),
    )
    for basis in bases
        for (x, xᵢ) in (
                (SVector(1.0, 2.0), SVector(2.0, 4.0)),           # 2D
                (SVector(1.0, 2.0, 3.0), SVector(2.0, 4.0, 1.0)), # 3D
            )
            for dim in eachindex(x)
                grad_x = RBF.grad_applied_partial_wrt_x(basis, dim)
                grad_xi = RBF.grad_applied_partial_wrt_xi(basis, dim)
                fd = FD.gradient(y -> RBF.∂(basis, dim)(y, xᵢ), x)
                @testset "$(nameof(typeof(basis))) $(length(x))D dim=$dim" begin
                    @test grad_x(x, xᵢ) ≈ fd rtol = 1.0e-8
                    @test grad_xi(x, xᵢ) ≈ -fd rtol = 1.0e-8
                end
            end
        end
    end
end

@testset "PHS1 gradients at r = 0" begin
    # PHS1's ℒφ is singular at r = 0; the backward pass defines the gradient as 0 there.
    x = SVector(1.0, 2.0)
    phs1 = PHS(1; poly_deg = -1)
    @test RBF.grad_applied_laplacian_wrt_x(phs1)(x, x) == zero(x)
    @test RBF.grad_applied_partial_wrt_x(phs1, 1)(x, x) == zero(x)
end
