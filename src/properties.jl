using SpecialSets

@inline get_return_type(f, args...) = Core.Compiler.return_type(f, args)
@inline get_return_type(::typeof(+), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline get_return_type(::typeof(-), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline get_return_type(::typeof(*), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline get_return_type(::typeof(/), a::Type{<:Number}, b::Type{<:Number}) = promote_type(a, b)
@inline get_return_type(::typeof(max), a, b) = Union{a, b}
@inline get_return_type(::typeof(min), a, b) = Union{a, b}
@inline get_return_type(::typeof(+), a::Type{<:Number}) = a
@inline get_return_type(::typeof(-), a::Type{<:Number}) = a
