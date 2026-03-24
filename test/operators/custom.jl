using RadialBasisFunctions
import RadialBasisFunctions as RBF
using StaticArraysCore
using Statistics
using HaltonSequences
using Test

include("../test_utils.jl")

# Setup
N = 1000
x = SVector{2}.(HaltonPoint(2)[1:N])

@testset "Custom struct" begin
    c = Custom{0}(identity)
    @test c(1) == 1
end

@testset "Printing" begin
    c = Custom{0}(identity)
    @test RadialBasisFunctions.print_op(c) == "Custom Operator"
end

@testset "custom() Keyword Constructor" begin
    # Test custom.jl lines 38-40: primary keyword constructor
    op = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ); rank = 0)
    @test op isa RadialBasisOperator

    # Test with explicit keyword arguments
    op2 = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ); rank = 0, basis = PHS(5; poly_deg = 3))
    @test op2 isa RadialBasisOperator
end

@testset "custom() Positional Basis Constructor" begin
    # Test custom.jl lines 43-45: backward compatible positional basis
    op = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ), PHS(3; poly_deg = 2); rank = 0)
    @test op isa RadialBasisOperator

    # Test with different basis types
    op_imq = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ), IMQ(1; poly_deg = 2); rank = 0)
    @test op_imq isa RadialBasisOperator
end

@testset "custom() Different Eval Points" begin
    # Test custom.jl lines 47-55: separate evaluation points
    x2 = SVector{2}.(HaltonPoint(2)[(N + 1):(N + 100)])
    op = custom(x, x2, basis -> (x, xᵢ) -> basis(x, xᵢ); rank = 0)
    @test op isa RadialBasisOperator
    @test length(op.eval_points) == 100

    # With explicit basis
    op2 = custom(x, x2, basis -> (x, xᵢ) -> basis(x, xᵢ), PHS(5; poly_deg = 3); rank = 0)
    @test op2 isa RadialBasisOperator
end

@testset "Custom rank-1 operator (gradient-like)" begin
    # Build a Custom{1} operator that mimics gradient: basis -> ntuple(dim -> ∂(basis, dim), 2)
    op = custom(x, basis -> ntuple(dim -> RBF.∂(basis, dim), 2); rank = 1)
    @test op isa RadialBasisOperator

    # Weights should be a tuple of matrices (one per spatial dimension)
    @test op.weights isa NTuple{2}
    @test op.weights[1] isa AbstractMatrix
    @test op.weights[2] isa AbstractMatrix

    # Apply to scalar field: f(x,y) = x² + 2xy
    # ∂f/∂x = 2x + 2y, ∂f/∂y = 2x
    u = getindex.(x, 1) .^ 2 .+ 2 .* getindex.(x, 1) .* getindex.(x, 2)
    result = op(u)

    @test result isa Matrix
    @test size(result) == (N, 2)

    expected_dx = 2 .* getindex.(x, 1) .+ 2 .* getindex.(x, 2)
    expected_dy = 2 .* getindex.(x, 1)
    @test mean_percent_error(result[:, 1], expected_dx) < 10
    @test mean_percent_error(result[:, 2], expected_dy) < 10

    # Apply to vector field (matrix input): each column differentiated independently
    u_vec = hcat(u, getindex.(x, 2) .^ 2)  # second component: y², ∂/∂x=0, ∂/∂y=2y
    result_vec = op(u_vec)

    @test result_vec isa Array{<:Any, 3}
    @test size(result_vec) == (N, 2, 2)

    # First component derivatives (same as above)
    @test mean_percent_error(result_vec[:, 1, 1], expected_dx) < 10
    @test mean_percent_error(result_vec[:, 1, 2], expected_dy) < 10

    # Second component: ∂(y²)/∂x ≈ 0, ∂(y²)/∂y = 2y
    @test maximum(abs.(result_vec[:, 2, 1])) < 0.1
    @test mean_percent_error(result_vec[:, 2, 2], 2 .* getindex.(x, 2)) < 10
end

@testset "@operator Macro Syntax" begin
    f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
    d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
    d2f_dyy(x) = -4 * sin(2 * x[2])

    N_macro = 10_000
    x_macro = SVector{2}.(HaltonPoint(2)[1:N_macro])
    y_macro = f.(x_macro)
    basis_macro = PHS(5; poly_deg = 3)

    κ = [3.0, 0.5]
    op = custom(x_macro, @operator(∇ ⋅ (κ * ∇)); rank = 0, basis = basis_macro)
    exact = κ[1] .* d2f_dxx.(x_macro) .+ κ[2] .* d2f_dyy.(x_macro)
    @test mean_percent_error(op(y_macro), exact) < 5
end

@testset "@operator c ⋅ ∇ (advection)" begin
    f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
    df_dx(x) = 4 * cos(4 * x[1]) - 3 * sin(3 * x[1])
    df_dy(x) = 2 * cos(2 * x[2])

    N_macro = 10_000
    x_macro = SVector{2}.(HaltonPoint(2)[1:N_macro])
    y_macro = f.(x_macro)
    basis_macro = PHS(5; poly_deg = 3)

    c = SVector(1.0, 0.5)
    op = custom(x_macro, @operator(c ⋅ ∇); basis = basis_macro)
    exact = c[1] .* df_dx.(x_macro) .+ c[2] .* df_dy.(x_macro)
    @test mean_percent_error(op(y_macro), exact) < 5
end

@testset "@operator Macro Composition" begin
    f(x) = 1 + sin(4 * x[1]) + cos(3 * x[1]) + sin(2 * x[2])
    d2f_dxx(x) = -16 * sin(4 * x[1]) - 9 * cos(3 * x[1])
    d2f_dyy(x) = -4 * sin(2 * x[2])

    N_macro = 10_000
    x_macro = SVector{2}.(HaltonPoint(2)[1:N_macro])
    y_macro = f.(x_macro)
    basis_macro = PHS(5; poly_deg = 3)

    κ = 2.0
    k² = 1.0
    op = custom(x_macro, @operator(∇ ⋅ (κ * ∇) + k² * f); rank = 0, basis = basis_macro)
    exact = κ .* (d2f_dxx.(x_macro) .+ d2f_dyy.(x_macro)) .+ k² .* f.(x_macro)
    @test mean_percent_error(op(y_macro), exact) < 5
end

@testset "_infer_rank" begin
    @test RBF._infer_rank(basis -> (x, xᵢ) -> basis(x, xᵢ)) == 0
    @test RBF._infer_rank(basis -> ntuple(dim -> RBF.∂(basis, dim), 2)) == 1
    # rank auto-inferred when keyword omitted
    op = custom(x, basis -> (x, xᵢ) -> basis(x, xᵢ))
    @test op isa RadialBasisOperator
end

@testset "@operator symbol variants" begin
    @test @operator(∇²) isa Laplacian
    @test @operator(Δ) isa Laplacian
    @test @operator(∇ ⋅ ∇) isa Laplacian
    @test @operator(I) isa Identity
    @test @operator(∂(1)) isa Partial
    @test @operator(∂²(2)) isa Partial
end

@testset "@operator edge cases" begin
    # Scalar literal fallback (_transform_operator_expr line 34)
    @test @operator(3 * ∇²) isa RBF.ScaledOperator
    # Unrecognized call fallback (_transform_operator_call line 76)
    k = 2.0
    op = @operator(k^2 * f)
    @test op isa RBF.ScaledOperator
    # Multi-coefficient ∇⋅(a*b*∇) (_extract_nabla_coefficient else branch, line 99)
    a = 2.0
    b = 3.0
    op2 = @operator(∇ ⋅ (a * b * ∇))
    @test op2 isa RBF.ScaledOperator
    # Division by scalar
    op3 = @operator(∇² / 2)
    @test op3 isa RBF.ScaledOperator
    @test op3.α ≈ 0.5
    # Invalid ∂ arity (LoadError wraps the ArgumentError from macro expansion)
    @test_throws LoadError @eval @operator(∂(1, 2))
    @test_throws LoadError @eval @operator(∂²(1, 2))
    # Non-:call Expr fallthrough (_transform_operator_expr line 31)
    ref_expr = Expr(:ref, :κ, 1)
    @test RBF._transform_operator_expr(ref_expr) === ref_expr
    # Generic dot passthrough (_transform_dot line 90, neither side is ∇)
    dot_result = RBF._transform_dot(:a, :b)
    @test dot_result.head == :call
    @test dot_result.args[1] === :⋅
end

@testset "@operator subtraction and negation" begin
    f(x) = 2 * x[1] + 3 * x[2]
    df_dx(x) = 2
    df_dy(x) = 3

    N_macro = 10_000
    x_macro = SVector{2}.(HaltonPoint(2)[1:N_macro])
    y_macro = f.(x_macro)
    basis_macro = PHS(5; poly_deg = 3)

    # Binary subtraction
    op = custom(x_macro, @operator(∂(1) - ∂(2)); basis = basis_macro)
    exact = df_dx.(x_macro) .- df_dy.(x_macro)
    @test mean_percent_error(op(y_macro), exact) < 5

    # Unary negation
    op_neg = custom(x_macro, @operator(-∂(1)); basis = basis_macro)
    @test mean_percent_error(op_neg(y_macro), -df_dx.(x_macro)) < 5
end

@testset "custom() Hermite Boundary Conditions" begin
    # Test custom.jl lines 58-72: Hermite interpolation with boundary conditions
    # Create a simple 1D domain for testing
    spacing = 0.1
    domain = [SVector{1}(x) for x in 0.0:spacing:1.0]
    N_domain = length(domain)

    # Identify boundary points (first and last)
    is_boundary = zeros(Bool, N_domain)
    is_boundary[1] = true
    is_boundary[end] = true

    # Set up boundary conditions (Dirichlet at both ends)
    boundary_conditions = [RBF.Dirichlet(), RBF.Dirichlet()]

    # Normals for 1D (outward pointing)
    normals = [SVector(-1.0), SVector(1.0)]

    # Test that the constructor works
    op = custom(
        domain,
        domain,
        basis -> (x, xᵢ) -> basis(x, xᵢ),
        PHS(3; poly_deg = 2),
        is_boundary,
        boundary_conditions,
        normals;
        rank = 0,
    )
    @test op isa RadialBasisOperator
end
