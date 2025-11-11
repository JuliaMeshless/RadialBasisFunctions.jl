using RadialBasisFunctions
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(
    RadialBasisFunctions, :DocTestSetup, :(using RadialBasisFunctions); recursive=true
)

makedocs(;
    modules=[RadialBasisFunctions],
    authors="Kyle Beggs",
    sitename="RadialBasisFunctions.jl",
    repo=Documenter.Remotes.GitHub("JuliaMeshless", "RadialBasisFunctions.jl"),
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="https://github.com/JuliaMeshless/RadialBasisFunctions.jl",
        devbranch="main",
        devurl="dev",
        md_output_path=".",
        build_vitepress=false,
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Theory" => "theory.md",
        "Internals" => "internals.md",
        "API" => "api.md",
    ],
    clean=false,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/JuliaMeshless/RadialBasisFunctions.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main", # or master, trunk, ...
    push_preview=true,
)
