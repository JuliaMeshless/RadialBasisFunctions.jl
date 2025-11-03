using SafeTestsets

@safetestset "Basis - General" begin
    include("basis/basis.jl")
end

@safetestset "Polyharmonic Splines" begin
    include("basis/polyharmonic_spline.jl")
end

@safetestset "Gaussian" begin
    include("basis/gaussian.jl")
end

@safetestset "Inverse Multiquadric" begin
    include("basis/inverse_multiquadric.jl")
end

@safetestset "Monomial" begin
    include("basis/monomial.jl")
end

@safetestset "Operators" begin
    include("operators/operators.jl")
end

@safetestset "Partial Derivatives" begin
    include("operators/partial.jl")
end

@safetestset "Gradient" begin
    include("operators/gradient.jl")
end

@safetestset "Directional Derivative" begin
    include("operators/directional.jl")
end

@safetestset "Laplacian" begin
    include("operators/laplacian.jl")
end

@safetestset "Custom" begin
    include("operators/custom.jl")
end

@safetestset "Interpolation" begin
    include("operators/interpolation.jl")
end

@safetestset "Regridding" begin
    include("operators/regrid.jl")
end

@safetestset "Virtual" begin
    include("operators/virtual.jl")
end

@safetestset "Operator Algebra" begin
    include("operators/operator_algebra.jl")
end

@safetestset "Monomial Operators" begin
    include("operators/monomial.jl")
end

@safetestset "Stencil" begin
    include("solve.jl")
end

@safetestset "Boundary Types" begin
    include("boundary_types.jl")
end

@safetestset "Solve Unit Tests" begin
    # include("solve/unit/matrix_entries.jl")
    # include("solve/unit/collocation_matrix.jl")
    # include("solve/unit/rhs_vector.jl")
end

#these are still work in progress:
@safetestset "Solve Integration Tests" begin
    # include("solve/integration/hermite_integration.jl")
    # include("solve/integration/solve_utils_integration.jl")
    include("solve/integration/end_to_end_utils.jl")
    include("solve/integration/laplacian_end_to_end.jl")
    include("solve/integration/gradient_end_to_end.jl")
    include("solve/integration/partial_x_end_to_end.jl")
    include("solve/integration/partial_y_end_to_end.jl")
end
