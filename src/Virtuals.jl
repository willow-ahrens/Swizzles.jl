module Virtuals

using Base.Broadcast: Broadcasted, Extruded
using LinearAlgebra

using Swizzles.ArrayifiedArrays
using Swizzles.Properties

export Virtual, VirtualArray, virtualize

struct Virtual{T}
    ex
end

Base.ndims(::Virtual{T}) where {T} = ndims(T)
Base.ndims(::Type{Virtual{T}}) where {T} = ndims(T)
Base.axes(::Virtual{T}) where {T} = ndims(T) == 0 ? () : error() #FIXME return virtual axes!
Base.Broadcast.broadcastable(x::Virtual{<:Union{AbstractArray,Number,Ref,Tuple,Broadcasted}}) = x

struct VirtualArray{T, N, Data<:AbstractArray{T, N}} <: AbstractArray{T, N}
    ex
end

function virtualize(root, Data::AbstractArray{T, N}) where {T, N}
    return VirtualArray{T, N, Data}(root)
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
    try
        return Data.instance
    catch Error
        return Virtual{Data}(root)
    end
end

function virtualize(root, Data::Type{<:AbstractArray{T, N}}) where {T, N}
    try
        return Data.instance
    catch Error
        return VirtualArray{T, N, Data}(root)
    end
end

function virtualize(root, Data::Type{<:Tuple})
    data = map(((i, Datum),) -> virtualize(:($root[$i]), Datum), enumerate(Data.parameters))
    return (data...,)
end

function virtualize(root, ::Type{<:Broadcasted{Style, Axes, F, Args}}) where {Style, Axes, F, Args}
    f = virtualize(:($root.f), F)
    args = virtualize(:($root.args), Args)
    axes = virtualize(:($root.axes), Axes)
    return Broadcasted{Style}(f, args, axes)
end

function virtualize(root, ::Type{<:Extruded{Arg, Keeps, Defaults}}) where {Arg, Keeps, Defaults}
    arg = virtualize(:($root.arg), Arg)
    keeps = virtualize(:($root.keeps), Keeps)
    defaults = virtualize(:($root.defaults), Defaults)
    return Extruded(arg, keeps, defaults)
end

end
