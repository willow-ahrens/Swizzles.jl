module ArrayifiedArrays

using Swizzles.Properties
using Swizzles.ScalarArrays
using Swizzles.WrapperArrays
using Swizzles.StylishArrays

using Swizzles

using Base: dataids, unaliascopy, unalias
using Base.Broadcast: Broadcasted, Extruded
using Base.Broadcast: BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle
using Base.Broadcast: instantiate, broadcastable, _broadcast_getindex, combine_eltypes, extrude, broadcast_unalias

export ArrayifiedArray, arrayify, preprocess_broadcasts

struct ArrayifiedArray{T, N, Arg} <: StylishArray{T, N}
    arg::Arg
    @inline function ArrayifiedArray{T, N, Arg}(arg::Arg) where {T, N, Arg}
        return new{T, N, typeof(arg)}(arg)
    end
end

@inline function ArrayifiedArray(arg)
    arr = ArrayifiedArray{Any}(arg)
    return ArrayifiedArray{Properties.eltype_bound(arr)}(arg)
end

@inline function ArrayifiedArray{T}(arg) where {T}
    arg = instantiate(broadcastable(arg))
    return ArrayifiedArray{T, ndims(arg), typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T}(arg::Tuple) where {T}
    return ArrayifiedArray{T, 1, typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T}(arg::Broadcasted{<:AbstractArrayStyle{0}, Nothing}) where {T}
    return ArrayifiedArray{T, 0, typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T}(arg::Broadcasted{Style{Tuple}, Nothing}) where {T}
    return ArrayifiedArray{T, 1, typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T, N}(arg) where {T, N}
    arg = instantiate(broadcastable(arg))
    return ArrayifiedArray{T, N, typeof(arg)}(arg)
end

@inline ArrayifiedArray{T}(arr::ArrayifiedArray{S, N, Arg}) where {T, S, N, Arg} = ArrayifiedArray{T, N, Arg}(arr.arg)
@inline ArrayifiedArray{T, N}(arr::ArrayifiedArray{S, N, Arg}) where {T, S, N, Arg} = ArrayifiedArray{T, N, Arg}(arr.arg)
@inline ArrayifiedArray{T, N, Arg}(arr::ArrayifiedArray{S, N, Arg}) where {T, S, N, Arg} = ArrayifiedArray{T, N, Arg}(arr.arg)

@inline function Properties.eltype_bound(arr::ArrayifiedArray)
    if arr.arg isa AbstractArray
        T = Properties.eltype_bound(arr.arg)
        if T <: eltype(arr)
            return T
        end
    else
        T = eltype(arr.arg)
        if T <: eltype(arr)
            return T
        else
            return eltype(arr)
        end
    end
end

@inline Base.eltype(bc::Broadcasted) = combine_eltypes(bc.f, bc.args)
@inline Base.similar(bc::Broadcasted) = similar(bc, eltype(bc))
@inline Base.similar(bc::Broadcasted, args...) = similar(bc, eltype(bc), args...)
@inline Base.similar(bc::Broadcasted{DefaultArrayStyle{0}}) = ScalarArray{eltype(bc)}()

@inline function Properties.eltype_bound(arr::ArrayifiedArray{<:Any, <:Any, <:Broadcasted})
    return combine_eltypes(arr.arg.f, arr.arg.args)
end

function Base.show(io::IO, arr::ArrayifiedArray{T, N}) where {T, N}
    print(io, ArrayifiedArray{T, N}) #Showing the arg type (although maybe useful since it's allowed to differ), will likely be redundant.
    print(io, '(', arr.arg, ')')
    nothing
end

arrayify(arg::AbstractArray) = arg
arrayify(arg) = ArrayifiedArray(arg)

#The general philosophy of a ArrayifiedArray is that it should use broadcast to answer questions unless it's arg is an abstract Array, then it should fall back to the parent
#We can go through and add more base Abstract Array stuff later.
Base.parent(arr::ArrayifiedArray) = arr
WrapperArrays.iswrapper(arr::ArrayifiedArray) = arr.arg isa AbstractArray

Base.dataids(arr::ArrayifiedArray) = dataids(arr.arg)
Base.unaliascopy(arr::ArrayifiedArray{T, N, Arg}) where {T, N, Arg} = ArrayifiedArray{T, N, Arg}(unaliascopy(arr.arg))
Base.unalias(dst, arr::ArrayifiedArray{T, N, Arg}) where {T, N, Arg} = ArrayifiedArray{T, N, Arg}(unalias(dst, arr.arg))

@inline Base.axes(arr::ArrayifiedArray) = axes(arr.arg)

@inline Base.size(arr::ArrayifiedArray) = map(length, axes(arr.arg))

@inline Base.eltype(arr::ArrayifiedArray{T}) where {T} = T

@inline Base.eachindex(arr::ArrayifiedArray{T, N, <:AbstractArray}) where {T, N} = eachindex(arr.arg)
@inline Base.eachindex(arr::ArrayifiedArray) = _eachindex(axes(arr))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices(t)

Base.ndims(::Type{<:ArrayifiedArray{T, N}}) where {T, N} = N
Base.ndims(::ArrayifiedArray{T, N}) where {T, N} = N

Base.length(arr::ArrayifiedArray{T, N, <:AbstractArray}) where {T, N} = length(arr.arg)
Base.length(arr::ArrayifiedArray) = prod(map(length, axes(arr)))

#FIXME define some IndexStyle stuffs
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, <:Broadcasted}, I::Int) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, <:Broadcasted}, I::CartesianIndex) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, <:Broadcasted}, I::Int...) = _broadcast_getindex(arr.arg, CartesianIndex(I))
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, <:Broadcasted}) = _broadcast_getindex(arr.arg, 1)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::Int) = getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::Int...) = getindex(arr.arg, I...)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray) = getindex(arr.arg)

Base.@propagate_inbounds Base.setindex!(arr::ArrayifiedArray, x, I::Int) = setindex(arr.arg, x, I)
Base.@propagate_inbounds Base.setindex!(arr::ArrayifiedArray, x, I::Int...) = setindex(arr.arg, x, I...)
Base.@propagate_inbounds Base.setindex!(arr::ArrayifiedArray, x) = setindex(arr.arg, x)

@inline myidentity(x) = x

@inline Base.Broadcast.broadcastable(arr::ArrayifiedArray) = broadcastable(arr.arg)


#This should do the same thing as Broadcast preprocess does, but apply the ArrayifiedArrays preprocess first
@inline Base.Broadcast.preprocess(dst, arr::AbstractArray) = extrude(broadcast_unalias(dst, preprocess_broadcasts(dst, arr)))
@inline Base.Broadcast.preprocess(dst, ext::Extruded) = Extruded(broadcast_unalias(dst, preprocess_broadcasts(dst, ext.x)), ext.keeps, ext.defaults) #make broadcast safe for double-preprocessing
@inline preprocess_broadcasts(dst, bc::Broadcasted) = Base.Broadcast.preprocess(dst, bc)
@inline function preprocess_broadcasts(dst, arr)
    if iswrapper(arr)
        adopt(preprocess_broadcasts(dst, parent(arr)), arr)
    else
        arr
    end
end
@inline function preprocess_broadcasts(dst, arr::ArrayifiedArray{T, N}) where {T, N}
    if arr.arg isa AbstractArray
        return preprocess_broadcasts(dst, arr)
    end
    arg = Base.Broadcast.preprocess(dst, arr.arg)
    return ArrayifiedArray{T, N, typeof(arg)}(arg)
end



@inline Base.Broadcast.BroadcastStyle(::Type{ArrayifiedArray{T, N, Arg}}) where {T, N, Arg} = BroadcastStyle(Arg)





end
