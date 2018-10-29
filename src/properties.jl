using SpecialSets

@inline return_type(f, arg_types) = Core.Compiler.return_type(f, arg_types)
@inline return_type(::typeof(+), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(-), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(*), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(/), (a, b)::Tuple{Type{<:Number}, Type{<:Number}}) = promote_type(a, b)
@inline return_type(::typeof(max), (a, b)) = Union(a, b)
@inline return_type(::typeof(min), (a, b)) = Union(a, b)
@inline return_type(::typeof(+), (a,)::Tuple{Type{<:Number}}) = a
@inline return_type(::typeof(-), (a,)::Tuple{Type{<:Number}}) = a


using Base.Broadcast: broadcasted, BroadcastStyle

struct Broadcaster{F}
  f::F
end

(bc::Broadcaster)(args...) = broadcasted(bc.f, args...)

Base.isequal(a::Broadcaster, b::Broadcaster) = a.f == b.f

function metatransform(expr, T)
    # essentially, this function performs `metatransform(expr, ::Type{typeof(+)}) = QuoteNode(+)`
    if T isa DataType && isdefined(T, :instance)
        return QuoteNode(T.instance)
    end
    return :($expr::$T)
end

function metatransform(expr, ::Type{<:Broadcasted{Style, Axes, F, Args}}) where {Style, Axes, F, Args}
    f = metatransform(:($expr.f), F)
    args = [metatransform(:($expr.args[$i]), arg) for (i, arg) in enumerate(Args.parameters)]
    if f isa QuoteNode
        return :($(Broadcaster(f.value))($(args...)))
    else
        return :((Broadcaster($expr.f))($(args...)))
    end
end

function metatransform(expr, ::Type{<:BroadcastedArray{T, N, Arg}}) where {T, N, Arg}
    arg = metatransform(:($expr.arg), Arg)
    return :(BroadcastedArray{$T}($arg))
end

function metatransform(expr, ::Type{<:SwizzledArray{T, N, Arg, mask, imask, Op}}) where {T, N, Arg, mask, imask, Op}
    op = metatransform(:($expr.op), Op)
    arg = metatransform(:($expr.arg), Arg)
    if op isa QuoteNode
        return :($(Swizzler(mask, op.value))($arg))
    else
        return :((Swizzler($mask, $expr.op))($arg))
    end
end

function metatransform(expr, ::Type{<:SwizzledArray{T, N, <:BroadcastedArray{<:Any, <:Any, Arg}, mask, imask, Op}}) where {T, N, Arg, mask, imask, Op}
    op = metatransform(:($expr.op), Op)
    arg = metatransform(:($expr.arg.arg), Arg)
    if op isa QuoteNode
        return :($(Swizzler(mask, op.value))($arg))
    else
        return :((Swizzler($mask, $expr.op))($arg))
    end
end
