module Properties
using Base.FastMath: add_fast, mul_fast, min_fast, max_fast

"""
    initial(f, T, S)

Return the value `i` such that `something(i)::T`, `f(something(i), x)::T`, and
`f(something(i), x) == x` for all values `x` of type `S`. Return `nothing` if
you cannot reasonably return such a value.

See also: [`zero`](@ref), [`oneunit`](@ref).

# Examples
```jldoctest
julia> initial(+, Int, Int)
0
julia> initial(+, Real, Real)
false
julia> initial(+, Int, Real)
nothing
```
"""

@inline initial(f, T, S) = nothing

@inline initial(::typeof(+), T, S::Type{<:Number}) =
    S <: T                                  ? Some(zero(S)) :
    return_type(+, typeof(zero(T)), S) <: T ? Some(zero(T)) :
                                              nothing
@inline initial(::typeof(add_fast), T, S::Type{<:Number}) =
    S <: T                                         ? Some(zero(S)) :
    return_type(add_fast, typeof(zero(T)), S) <: T ? Some(zero(T)) :
                                                     nothing
@inline initial(::typeof(*), T, S::Type{<:Number}) =
    S <: T                                                ? Some(oneunit(S)) :
    one(S) <: T && return_type(*, typeof(one(S)), S) <: T ? Some(one(S))     :
                                                            nothing
@inline initial(::typeof(mul_fast), T, S::Type{<:Number}) =
    S <: T                                                       ? Some(oneunit(S)) :
    one(S) <: T && return_type(mul_fast, typeof(one(S)), S) <: T ? Some(one(S))     :
                                                                   nothing
@inline initial(::typeof(max), T::Type{<:Number}, S::Type{<:Number}) =
    Some(typemin(T))
@inline initial(::typeof(max_fast), T::Type{<:Number}, S::Type{<:Number}) =
    Some(typemin(T))
@inline initial(::typeof(min), T::Type{<:Number}, S::Type{<:Number}) =
    Some(typemax(T))
@inline initial(::typeof(min_fast), T::Type{<:Number}, S::Type{<:Number}) =
    Some(typemax(T))

"""
    return_type(f, args...)

Returns a type `T` containing the return type of the function `f` called with
arguments of type `args...`. Should run in time linear with the description of
the types of its arguments.

See also: [`Core.Compiler.return_type`](@ref).

# Examples
```jldoctest
julia> return_type(+, Int, Int)
Int64
julia> return_type(+, Real, Real)
Any
```
"""
@inline function return_type(f, args...)
    rt = Core.Compiler.return_type(f, args)
    if rt === Union{}
        return Any
    else
        return rt
    end
end
@inline return_type(::typeof(+), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline return_type(::typeof(add_fast), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline return_type(::typeof(*), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline return_type(::typeof(mul_fast), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline return_type(::typeof(max), a, b) = Union{a, b}
@inline return_type(::typeof(max_fast), a, b) = Union{a, b}
@inline return_type(::typeof(min), a, b) = Union{a, b}
@inline return_type(::typeof(min_fast), a, b) = Union{a, b}


"""
    eltype_bound(A)

Return the strictest bound on the element type of `A` that is possible to obtain
in constant time. This may be stricter than `eltype(A)`, which corresponds to
the declared element type of `A`

See also: [`eltype`](@ref).
"""
@inline eltype_bound(arg) = eltype(arg)

#declare opposite

end
