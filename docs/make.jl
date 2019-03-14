using Documenter
using Swizzles

makedocs(
    sitename="Swizzles.jl",
    modules = [Swizzles]
    authors = "Peter Ahrens"
    pages = [
        "Home" => "index.md"
        "Getting Started" => "getting_started.md"
        "Swizzle Friends" => "swizzle_friends.md"
        "Abstracter Arrays" => "abstracter_arrays.md"
        "API" => "api.md"
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/peterahrens/Swizzles.jl"
)
