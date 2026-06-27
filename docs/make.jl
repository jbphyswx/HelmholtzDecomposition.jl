using Documenter: Documenter
using HelmholtzDecomposition: HelmholtzDecomposition

Documenter.makedocs(;
    modules = [HelmholtzDecomposition],
    sitename = "HelmholtzDecomposition.jl",
    authors = "Jordan Benjamin",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jbphyswx.github.io/HelmholtzDecomposition.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Architecture" => "architecture.md",
        "Examples" => "examples.md",
        "Coarse-Graining Workflow" => "coarse_graining_workflow.md",
    ],
    warnonly = true,
)

Documenter.deploydocs(;
    repo = "github.com/jbphyswx/HelmholtzDecomposition.jl",
    devbranch = "main",
    push_preview = true,
)
