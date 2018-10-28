@inline return_type(f, arg_types) = Core.Compiler.return_type(f, arg_types)
@inline return_type(::typeof(+), (a, b)) = promote_type(a, b)
