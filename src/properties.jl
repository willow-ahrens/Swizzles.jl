@inline return_type(f, arg_types) = Core.Compiler.return_type(f, arg_types)
for f in [+, -, *, /]
    @inline return_type(::typeof(f), (a::Type{<:Number}, b::Type{<:Number})) = promote_type(a, b)
end
for f in [max, min]
    @inline return_type(::typeof(f), (a, b)) = Union(a, b)
end
for f in [+, -]
    @inline return_type(::typeof(f), (a::Type{<:Number},)) = a
end
