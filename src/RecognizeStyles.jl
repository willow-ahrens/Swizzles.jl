module RecognizeStyles

using Swizzles
using Swizzles.Antennae

using LinearAlgebra
using Base.Broadcast: Broadcasted, broadcasted

include("simplify.jl")

#=
"""
    match(arr)

Apply global rules to find best copyto implementation for expression `arr`.
"""
@generated function match(arr)
    simple_expr = Rewrite.with_context(RecognizeStyles.CONTEXT)) do
        normalize(@term(reprexpr(:arr, arr)), RecognizeStyles.RULES)
    end
    for kernel in RecognizeStyles.KERNELS
        match(kernel, simple_expr)
        return quote
            $kernel
        end
    end
end

struct RecognizeStyle <: BroadcastStyle end


function Base.copy(arr::Broadcasted{RecognizeStyle})
    lookup(simplify(arr))
end

function lookup(arr::Broadcasted{+, Tuple{Array{Int}, Array{Int}}})
    A = arr.args[1]
    B = arr.args[2]
    for i in eachindex(A)

    end
end
=#
end
