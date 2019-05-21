module Virtuals

using Base.Broadcast: Broadcasted, Extruded
using LinearAlgebra

using Swizzles.ArrayifiedArrays
using Swizzles.Properties

export Virtual, VirtualArray, virtualize, virtual

struct Virtual{T}
    ex
end

function virtual(root, Data::Type)
    ins = instance(Data)
    if ins !== nothing
        return something(ins)
    else
        return Virtual{Data}(root)
    end
end

Base.ndims(::Virtual{T}) where {T} = ndims(T)
Base.ndims(::Virtual{<:Tuple}) = 1
Base.ndims(::Type{Virtual{T}}) where {T} = ndims(T)
Base.ndims(::Type{<:Virtual{<:Tuple}}) = 1
Base.axes(::Virtual{T}) where {T} = ndims(T) == 0 ? () : error() #FIXME return virtual axes!
Base.Broadcast.broadcastable(x::Virtual{<:Union{AbstractArray,Number,Ref,Tuple,Broadcasted}}) = x
Base.eltype(::Virtual{T}) where {T} = eltype(T)



struct VirtualArray{T, N, Data<:AbstractArray{T, N}} <: AbstractArray{T, N}
    ex
end

function Base.show(io::IO, arr::VirtualArray{T, N, D}) where {T, N, D}
    print(io, "VirtualArray{$T, $N, $D}")
    nothing
end

function virtualtuple(root, Data::Type{<:Tuple})
    data = map(((i, Datum),) -> virtualize(:($root[$i]), Datum), enumerate(Data.parameters))
    return (data...,)
end

function virtualize(root, Data::Type)
    ins = instance(Data)
    if ins !== nothing
        return something(ins)
    else
        return Virtual{Data}(root)
    end
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

function virtualize(root, Data::Type{<:AbstractArray{T, N}}) where {T, N}
    try
        return Data.instance
    catch Error
        return VirtualArray{T, N, Data}(root)
    end
end

function virtualize(root, ::Type{<:Broadcasted{Style, Axes, F, Args}}) where {Style, Axes, F, Args}
    f = virtual(:($root.f), F)
    args = (map(((i, Arg),) -> virtualize(:($root.args[$i]), Arg), enumerate(Args.parameters))...,)
    axes = (map(((i, Axis),) -> virtualize(:($root.axes[$i]), Axes), enumerate(Axes.parameters))...,)
    return Broadcasted{Style}(f, args, axes)
end

function virtualize(root, ::Type{<:Extruded{Arg, Keeps, Defaults}}) where {Arg, Keeps, Defaults}
    arg = virtualize(:($root.arg), Arg)
    args = (map(((i, K),) -> virtual(:($root.keeps[$i]), K), enumerate(Keeps.parameters))...,)
    axes = (map(((i, D),) -> virtual(:($root.defaults[$i]), D), enumerate(Defaults.parameters))...,)
    return Extruded(arg, keeps, defaults)
end

end
