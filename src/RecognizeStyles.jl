module RecognizeStyles

using Swizzles
using LinearAlgebra

"""
    reprexpr(root::Union{Symbol, Expr}, T::Type)

Given a root expression of type T, produce the most detailed constructor
expression you can to create a new object that should be equal to root.

# Examples
```jldoctest
julia> A = [1 2 3 4; 5 6 7 8]'
4×2 LinearAlgebra.Adjoint{Int64,Array{Int64,2}}:
 1  5
 2  6
 3  7
 4  8
julia> r = reprexpr(:A, typeof(A))
:(LinearAlgebra.Adjoint(parent(A)::Array{Int64,2}))
julia> eval(Main, r) == A
true
```
"""
reprexpr(root, T) = :($root::$T)

function reprexpr(root, ::Type{Adjoint{T, Arg}}) where {T, Arg}
    root′ = :($root.parent)
    arg′ = reprexpr(root′, Arg)
    :($(Adjoint)($arg′))
end

function reprexpr(root, ::Type{Transpose{T, Arg}}) where {T, Arg}
    root′ = :($root.parent)
    arg′ = reprexpr(root′, Arg)
    :($(Transpose)($arg′))
end

# TODO(Tony): get reprexpr working for Broadcasts

#=
"""
    simplify(arr)

Apply global rules to simplify the array expression `arr`.
"""
@generated function simplify(arr)
    simple_expr = Rewrite.with_context(RecognizeStyles.CONTEXT)) do
        normalize(@term(reprexpr(:arr, arr)), RecognizeStyles.RULES)
    end
    return quote
        $simple_expr
    end
end

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
