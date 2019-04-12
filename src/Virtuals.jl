module Virtuals

struct Virtual{T}
    ex
    t::T
end

struct VirtualArray{T, N, Data<:AbstractArray{T, N}} <: AbstractArray{T, N}
    ex
end

function virtualize(root, Data::AbstractArray{T, N})
    return VirtualArray{T, N, Data}(root)
end

function virtualize(root, Data::Type{<:ValArray})
    return Data()
end

function virtualize(root, ::Type{<:Adjoint{T, Arg}}) where {T, Arg}
    arg = virtualize(:($root.parent), Arg)
    return Adjoint{T, typeof(arg)}(arg)
end

function virtualize(root, ::Type{<:Transpose{T, Arg}}) where {T, Arg}
    arg = virtualize(:($root.parent), Arg)
    return Transpose{T, typeof(arg)}(arg)
end

function virtualize(root, ::Type{<:ArrayifiedArray{T, N, Arg}}) where {T, N, Arg}
    arg = virtualize(:($root.arg), Arg)
    return ArrayifiedArray{T, N, typeof(arg)}(arg)
end

function virtualize(root, Data::Type)
    if hasproperty(Data, :instance)
        return Data.instance
    else
        return Virtual{Data}(root)
    end
end

function virtualize(root, Data::Type{<:Tuple})
    data = map(((i, Datum),) -> virtualize(:($root[$i]), Datum), enumerate(Data.parameters))
    return (data...,)
end

function virtualize(root, ::Type{<:Broadcasted{Style, Axes, F, Args}}, syms) where {Style, Axes, F, Args}
    f = virtualize(:($root.f), F)
    args = virtualize(:($root.args), Args)
    axes = virtualize(:($root.axes), Axes)
    return Broadcasted{Style}(f, args, axes)
end

function virtualize(root, T::Type{<:SwizzledArray{T, N, Op, mask, Init, Arg}}, syms) where {Op, mask, Init, Arg}
    init = virtualize(:($root.init), Init)
    arg = virtualize(:($root.arg), Arg)
    op = virtualize(:($root.op), Op)
    return SwizzledArray{T, N, Op, mask}(op, init, arg)
end

end
