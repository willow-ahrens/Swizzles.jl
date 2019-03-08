module RecognizeStyles

using Swizzles
using Swizzles.Antennae

using LinearAlgebra
using Base.Broadcast: Broadcasted, broadcasted 


"""
    reprexpr(root::Union{Symbol, Expr}, T::Type) :: Expr

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
reprexpr(root::Union{Symbol, Expr}, T) :: Expr = :($root::$T) 

function reprexpr(root::Union{Symbol, Expr},
                  ::Type{Adjoint{<:Any, Arg}}) :: Expr where Arg 
    root′ = :($root.parent)
    arg′ = reprexpr(root′, Arg)
    :($(Adjoint)($arg′))
end

function reprexpr(root::Union{Symbol, Expr},
                  ::Type{Transpose{<:Any, Arg}}) :: Expr where Arg
    root′ = :($root.parent)
    arg′ = reprexpr(root′, Arg)
    :($(Transpose)($arg′))
end

function reprexpr(root::Union{Symbol, Expr},
                  ::Type{Broadcasted{<:Any, <:Any, <:Any, Args}}) :: Expr where Args<:Tuple
    f′ = :($root.f)
    arg_exprs = tuple_reprexpr(:($root.args), Args)

    :($Antenna($f′)($(arg_exprs...)))
end

function tuple_reprexpr(root::Union{Symbol, Expr},
                        ::Type{TType}) :: Array{Expr, 1} where TType<:Tuple
    [reprexpr(:($root[$idx]), EType)
         for (idx, EType) in enumerate(TType.parameters)]
end

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