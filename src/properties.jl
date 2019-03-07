module Properties
using Base.FastMath: add_fast, mul_fast, min_fast, max_fast

export return_type, initial
export Guard, Assume

"""
    initial(op, T)

Return the default initial value for the reduction operator `op` on the type
`T`. Return `nothing` if you such a value does not exist.

See also: [`zero`](@ref), [`oneunit`](@ref).

# Examples
```jldoctest
julia> initial(+, Int)
0
julia> initial(+, Real)
false
```
"""

@inline initial(f, T) = nothing

@inline initial(::typeof(+), T::Type{<:Number}) = Some(zero(T))
@inline initial(::typeof(add_fast), T::Type{<:Number}) = Some(zero(T))
@inline initial(::typeof(*), T::Type{<:Number}) = Some(oneunit(T))
@inline initial(::typeof(mul_fast), T::Type{<:Number}) = Some(oneunit(T))
@inline initial(::typeof(max), T::Type{<:Number}) = Some(typemin(T))
@inline initial(::typeof(max_fast), T::Type{<:Number}) = Some(typemin(T))
@inline initial(::typeof(min), T::Type{<:Number}) = Some(typemax(T))
@inline initial(::typeof(min_fast), T::Type{<:Number}) = Some(typemax(T))

"""
    initial_value(op, x)

Return the default initial value for the reduction operator `op` on the value
`x`. Return `nothing` if you such a value does not exist.

See also: [`zero`](@ref), [`oneunit`](@ref).

# Examples
```jldoctest
julia> initial_value(+, 1)
0
julia> initial_value(+, 1.0)
-0.0
```
"""
@inline initial_value(f::F, x) where {F} = initial(f, typeof(x))



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
    return Core.Compiler.return_type(f, args)
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




struct Guard{Op}
    op::Op
end

(op::Guard)(x::Nothing, y) = y
(op::Guard)(x, y) = op.op(x, y)
@inline return_type(op::Guard, T, S) = return_type(op.op, T, S)
@inline return_type(op::Guard, ::Type{Union{Nothing, T}}, S) where {T} = return_type(op.op, T, S)
@inline return_type(op::Guard, ::Type{Nothing}, S) = S

@inline initial(::Guard, ::Any) = Some(nothing)



struct Assume{T, Op}
    op::Op
end

@inline ((op::Assume{T})(x, y)::T) where {T} = op.op(x, y)
@inline function return_type(op::Assume{T}, S, R)::T where {T}
    Q = Properties.return_type(op.op, S, R)
    if Q <: T
        return Q
    else
        return T
    end
end

@inline initial(op::Assume, T) = initial(op.op, T)

end
