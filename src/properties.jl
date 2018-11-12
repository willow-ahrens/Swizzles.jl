using SpecialSets

@inline return_type(f, arg_types) = Core.Compiler.return_type(f, arg_types)
@inline return_type(::typeof(+), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(-), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(*), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(/), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(max), (a, b)) = Union{a, b}
@inline return_type(::typeof(min), (a, b)) = Union{a, b}
@inline return_type(::typeof(+), (a,)::Tuple{Type{<:Number}}) = a
@inline return_type(::typeof(-), (a,)::Tuple{Type{<:Number}}) = a
