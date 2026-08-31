using RadialBasisFunctions
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(
    RadialBasisFunctions, :DocTestSetup, :(using RadialBasisFunctions); recursive = true
)

makedocs(;
    modules = [RadialBasisFunctions],
    authors = "Kyle Beggs",
    sitename = "RadialBasisFunctions.jl",
    repo = Documenter.Remotes.GitHub("JuliaMeshless", "RadialBasisFunctions.jl"),
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/JuliaMeshless/RadialBasisFunctions.jl",
        devbranch = "main",
        devurl = "dev",
        build_vitepress = (!haskey(ENV, "VITEPRESS_DEV")),
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Guides" => [
            "Operators & Type Hierarchy" => "guides/operators.md",
            "Building PDE Operators" => "guides/pde_operators.md",
            "Custom Operators" => "guides/custom_operators.md",
            "Vector & Tensor Operators" => "guides/vector_calculus.md",
            "Automatic Differentiation" => "guides/autodiff.md",
            "RBF Neural Networks" => "guides/lux.md",
            "Convergence & Parameter Selection" => [
                "Overview" => "guides/convergence/index.md",
                "Scalar Operators" => "guides/convergence/scalar-operators.md",
                "Vector & Higher-Rank Operators" => "guides/convergence/vector-operators.md",
                "Shape-Parameter Bases" => "guides/convergence/shape-parameter-bases.md",
                "Work-Precision" => "guides/convergence/work-precision.md",
            ],
        ],
        "Reference" => [
            "Quick Reference" => "reference/quickref.md",
            "Troubleshooting" => "reference/troubleshooting.md",
            "Theory" => "reference/theory.md",
            "Internals" => "reference/internals.md",
            "API" => "reference/api.md",
        ],
    ],
    clean = false,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaMeshless/RadialBasisFunctions.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
