struct Undefined end

@inline function get_return_type(f, args...)
    if declare_return_type(f, args...) isa Undefined
        Core.Compiler.return_type(f, args)
    else
        declare_return_type(f, args...)
    end
end

@inline declare_return_type(f, args...) = Undefined()
@inline declare_return_type(::typeof(+), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline declare_return_type(::typeof(-), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline declare_return_type(::typeof(*), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline declare_return_type(::typeof(/), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline declare_return_type(::typeof(max), a, b) = Union{a, b}
@inline declare_return_type(::typeof(min), a, b) = Union{a, b}
@inline declare_return_type(::typeof(+), a::Type{<:Number}) = a
@inline declare_return_type(::typeof(-), a::Type{<:Number}) = a

@inline has_identity(f, T, S) = !(declare_identity(f, T, S) isa Undefined)
@inline get_identity(f, T, S) = declare_identity(f, T, S)

@inline declare_identity(f, T, S) = Undefined()
@inline declare_identity(::typeof(+), T::Type{<:Number}, S::Type{<:Number}) = zero(T)
@inline declare_identity(::typeof(*), T::Type{<:Number}, S::Type{<:Number}) = one(T)
@inline declare_identity(::typeof(max), T::Type{<:Number}, S::Type{<:Number}) = typemin(T)
@inline declare_identity(::typeof(min), T::Type{<:Number}, S::Type{<:Number}) = typemax(T)

#declare opposite
