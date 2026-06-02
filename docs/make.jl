using Documenter: Documenter
using HelmholtzDecomposition: HelmholtzDecomposition

Documenter.makedocs(;
    modules=[HelmholtzDecomposition],
    sitename="HelmholtzDecomposition.jl",
    pages=[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Architecture" => "architecture.md",
        "Examples" => "examples.md",
        "Coarse-Graining Workflow" => "coarse_graining_workflow.md",
    ],
)
