using SafeTestsets

@safetestset "Basis - General Utils" begin
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

@safetestset "Stencil" begin
    include("solve.jl")
end

@safetestset "Utils" begin
    include("utils.jl")
end
