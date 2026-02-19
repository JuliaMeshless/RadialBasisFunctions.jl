using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using KernelAbstractions: CPU
using HaltonSequences
using LinearAlgebra: Symmetric, bunchkaufman!, ldiv!, I
using Adapt

N = 100
x = SVector{2}.(HaltonPoint(2)[1:N])
z = sin.(getindex.(x, 1)) .+ cos.(getindex.(x, 2))

@testset "device field defaults to CPU" begin
    op = partial(x, 1, 1)
    @test op.device isa CPU

    op2 = laplacian(x)
    @test op2.device isa CPU

    op3 = gradient(x)
    @test op3.device isa CPU
end

@testset "device kwarg accepted by convenience constructors" begin
    cpu = CPU()

    op_partial = partial(x, 1, 1; device = cpu)
    @test op_partial.device isa CPU

    op_lap = laplacian(x; device = cpu)
    @test op_lap.device isa CPU

    op_grad = gradient(x; device = cpu)
    @test op_grad.device isa CPU

    op_dir = directional(x, [1.0, 0.0]; device = cpu)
    @test op_dir.device isa CPU

    x2 = SVector{2}.(HaltonPoint(2)[101:150])
    op_rg = regrid(x, x2; device = cpu)
    @test op_rg.device isa CPU

    op_custom = custom(x, (b) -> (x, xi) -> b(x, xi); device = cpu)
    @test op_custom.device isa CPU
end

@testset "explicit device=CPU() matches default" begin
    op_default = laplacian(x)
    op_explicit = laplacian(x; device = CPU())

    z_default = op_default(z)
    z_explicit = op_explicit(z)
    @test z_default ≈ z_explicit
end

@testset "update_weights! uses stored device" begin
    op = partial(x, 1, 1; device = CPU())
    original_weights = copy(op.weights)

    # Corrupt weights and verify update restores them
    op.weights .= 0.0
    update_weights!(op)
    @test op.weights ≈ original_weights
    @test is_cache_valid(op)
end

@testset "operator algebra preserves device" begin
    op1 = partial(x, 1, 2)
    op2 = partial(x, 2, 2)
    combined = op1 + op2
    @test combined.device isa CPU
end

@testset "backward-compat positional constructors accept device" begin
    cpu = CPU()

    # Data + basis (positional)
    op1 = RadialBasisOperator(Partial(1, 1), x, PHS(3; poly_deg = 2); device = cpu)
    @test op1.device isa CPU

    # Data + eval_points + basis (positional)
    x2 = SVector{2}.(HaltonPoint(2)[101:150])
    op2 = RadialBasisOperator(Partial(1, 1), x, x2, PHS(3; poly_deg = 2); device = cpu)
    @test op2.device isa CPU
end

# ============================================================================
# GPU Support Tests
# ============================================================================

@testset "_to_cpu" begin
    # No-op for plain Vector
    v = [SVector(1.0, 2.0), SVector(3.0, 4.0)]
    @test RBF._to_cpu(v) === v

    # Converts non-Vector AbstractVector to Array
    v_sub = view([SVector(1.0, 2.0), SVector(3.0, 4.0)], 1:2)
    result = RBF._to_cpu(v_sub)
    @test result isa Vector
    @test result == v_sub
end

@testset "_solve_system! Symmetric path matches bunchkaufman" begin
    # Build a small symmetric positive definite system
    n = 5
    A_data = randn(n, n)
    A_data = A_data' * A_data + 5.0 * I  # SPD
    b_vec = randn(n)

    # Reference: direct bunchkaufman
    A_ref = Symmetric(copy(A_data), :U)
    λ_ref = similar(b_vec)
    ldiv!(λ_ref, bunchkaufman!(A_ref, true), copy(b_vec))

    # _solve_system! with Symmetric
    A_sym = Symmetric(copy(A_data), :U)
    λ_sym = similar(b_vec)
    RBF._solve_system!(λ_sym, A_sym, copy(b_vec))
    @test λ_sym ≈ λ_ref

    # _solve_system! generic fallback (AbstractMatrix)
    A_plain = copy(A_data)
    λ_gen = similar(b_vec)
    RBF._solve_system!(λ_gen, A_plain, copy(b_vec))
    @test λ_gen ≈ λ_ref atol = 1e-10
end

@testset "_solve_system! with matrix RHS" begin
    n = 5
    nrhs = 3
    A_data = randn(n, n)
    A_data = A_data' * A_data + 5.0 * I
    B_mat = randn(n, nrhs)

    # Reference
    A_ref = Symmetric(copy(A_data), :U)
    λ_ref = similar(B_mat)
    ldiv!(λ_ref, bunchkaufman!(A_ref, true), copy(B_mat))

    # Symmetric path
    A_sym = Symmetric(copy(A_data), :U)
    λ_sym = similar(B_mat)
    RBF._solve_system!(λ_sym, A_sym, copy(B_mat))
    @test λ_sym ≈ λ_ref

    # Generic path
    A_plain = copy(A_data)
    λ_gen = similar(B_mat)
    RBF._solve_system!(λ_gen, A_plain, copy(B_mat))
    @test λ_gen ≈ λ_ref atol = 1e-10
end

@testset "Adapt.adapt_structure for RadialBasisOperator" begin
    op = laplacian(x)
    result = op(z)

    # Adapt with identity (CPU → CPU)
    adapted = Adapt.adapt(CPU(), op)
    @test adapted isa RadialBasisOperator
    @test adapted(z) ≈ result
    @test adapted.data === op.data
    @test adapted.adjl === op.adjl
    @test adapted.eval_points === op.eval_points
    @test is_cache_valid(adapted)
end

@testset "Adapt.adapt_structure for VectorValuedOperator" begin
    op = gradient(x)
    result = op(z)

    adapted = Adapt.adapt(CPU(), op)
    @test adapted isa RadialBasisOperator
    @test adapted(z) ≈ result
    @test adapted.data === op.data
    @test adapted.adjl === op.adjl
end

@testset "Adapt.adapt_structure for Interpolator" begin
    interp = Interpolator(x, z)
    test_pts = [SVector{2}(rand(2)) for _ in 1:5]
    result = interp(test_pts)

    adapted = Adapt.adapt(CPU(), interp)
    @test adapted isa Interpolator
    @test adapted(test_pts) ≈ result
    @test adapted.rbf_basis === interp.rbf_basis
    @test adapted.monomial_basis === interp.monomial_basis
end

@testset "_get_writable" begin
    A = randn(3, 3)
    @test RBF._get_writable(A) === A

    S = Symmetric(A, :U)
    @test RBF._get_writable(S) === A
end
