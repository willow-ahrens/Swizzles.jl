using Documenter
using Swizzles

makedocs(
    sitename="Swizzles",
    modules = [Swizzles]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/peterahrens/Swizzles.jl"
)
