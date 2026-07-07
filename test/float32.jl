using RadialBasisFunctions
using StaticArraysCore
using HaltonSequences

N = 200
x32 = [SVector{2, Float32}(p) for p in HaltonPoint(2)[1:N]]
x64 = [SVector{2, Float64}(p) for p in x32]

basis = PHS(3; poly_deg = 2)

@testset "Float32 end-to-end Laplacian" begin
    op32 = laplacian(x32; basis = basis)
    op64 = laplacian(x64; basis = basis)
    update_weights!(op32)
    update_weights!(op64)

    @testset "weights are Float32" begin
        @test eltype(op32.weights) == Float32
    end

    @testset "weights match Float64 build" begin
        W32 = Float64.(Matrix(op32.weights))
        W64 = Matrix(op64.weights)
        @test isapprox(W32, W64; rtol = sqrt(eps(Float32)))
    end

    @testset "operator application preserves Float32" begin
        y32 = sin.(4 .* getindex.(x32, 1)) .+ cos.(3 .* getindex.(x32, 2))
        out = op32(y32)
        @test out isa Vector{Float32}
    end
end
