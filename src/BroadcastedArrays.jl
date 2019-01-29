module BroadcastedArrays

using Swizzles.Properties
using Swizzles.WrapperArrays
using Swizzles.GeneratedArrays
using Swizzles

using Base: checkbounds_indices, throw_boundserror, tail, dataids, unaliascopy, unalias
using Base.Iterators: repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, instantiate, broadcastable, _broadcast_getindex, combine_eltypes, extrude, broadcast_unalias

export BroadcastedArray, arrayify

struct BroadcastedArray{T, N, Arg} <: GeneratedArray{T, N}
    arg::Arg
    @inline function BroadcastedArray{T, N, Arg}(arg::Arg) where {T, N, Arg}
        return new{T, N, typeof(arg)}(arg)
    end
end

@inline function BroadcastedArray(arg)
    arr = BroadcastedArray{Any}(arg)
    return BroadcastedArray{Properties.eltype_bound(arr)}(arg)
end

@inline function BroadcastedArray{T}(arg) where {T}
    arg = instantiate(broadcastable(arg))
    return BroadcastedArray{T, ndims(arg), typeof(arg)}(arg)
end

@inline function BroadcastedArray{T}(arg::Tuple) where {T}
    return BroadcastedArray{T, 1, typeof(arg)}(arg)
end

@inline function BroadcastedArray{T, N}(arg) where {T, N}
    arg = instantiate(broadcastable(arg))
    return BroadcastedArray{T, N, typeof(arg)}(arg)
end

@inline BroadcastedArray{T}(arr::BroadcastedArray{S, N, Arg}) where {T, S, N, Arg} = BroadcastedArray{T, N, Arg}(arr.arg)
@inline BroadcastedArray{T, N}(arr::BroadcastedArray{S, N, Arg}) where {T, S, N, Arg} = BroadcastedArray{T, N, Arg}(arr.arg)
@inline BroadcastedArray{T, N, Arg}(arr::BroadcastedArray{S, N, Arg}) where {T, S, N, Arg} = BroadcastedArray{T, N, Arg}(arr.arg)

@inline function Properties.eltype_bound(arr::BroadcastedArray)
    if arr.arg isa AbstractArray
        T = Properties.eltype_bound(arr.arg)
        if T <: eltype(arr)
            return eltype(arr.arg)
        end
    end
    return eltype(arr)
end

@inline function Properties.eltype_bound(arr::BroadcastedArray{<:Any, <:Any, <:Broadcasted})
    return combine_eltypes(arr.arg.f, arr.arg.args)
end

function Base.show(io::IO, arr::BroadcastedArray{T, N}) where {T, N}
    print(io, BroadcastedArray{T, N}) #Showing the arg type (although maybe useful since it's allowed to differ), will likely be redundant.
    print(io, '(', arr.arg, ')')
    nothing
end

arrayify(arg::AbstractArray) = arg
arrayify(arg) = BroadcastedArray(arg)

#The general philosophy of a BroadcastedArray is that it should use broadcast to answer questions unless it's arg is an abstract Array, then it should fall back to the parent
#We can go through and add more base Abstract Array stuff later.
Base.parent(arr::BroadcastedArray) = arr
WrapperArrays.iswrapper(arr::BroadcastedArray) = arr.arg isa AbstractArray

Base.dataids(arr::BroadcastedArray) = dataids(arr.arg)
Base.unaliascopy(arr::BroadcastedArray{T, N, Arg}) where {T, N, Arg} = BroadcastedArray{T, N, Arg}(unaliascopy(arr.arg))
Base.unalias(dst, arr::BroadcastedArray{T, N, Arg}) where {T, N, Arg} = BroadcastedArray{T, N, Arg}(unalias(dst, arr.arg))

@inline Base.axes(arr::BroadcastedArray) = axes(arr.arg)

@inline Base.size(arr::BroadcastedArray) = map(length, axes(arr.arg))

@inline Base.eltype(arr::BroadcastedArray{T}) where {T} = T

@inline Base.eachindex(arr::BroadcastedArray{T, N, <:AbstractArray}) where {T, N} = eachindex(arr.arg)
@inline Base.eachindex(arr::BroadcastedArray) = _eachindex(axes(arr))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices(t)

Base.ndims(::Type{<:BroadcastedArray{T, N}}) where {T, N} = N
Base.ndims(::BroadcastedArray{T, N}) where {T, N} = N

Base.length(arr::BroadcastedArray{T, N, <:AbstractArray}) where {T, N} = length(arr.arg)
Base.length(arr::BroadcastedArray) = prod(map(length, axes(arr)))

Base.@propagate_inbounds Base.getindex(arr::BroadcastedArray, I::Int) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::BroadcastedArray, I::CartesianIndex) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::BroadcastedArray, I::Int...) = _broadcast_getindex(arr.arg, CartesianIndex(I))
Base.@propagate_inbounds Base.getindex(arr::BroadcastedArray) = getindex(arr.arg)

@inline myidentity(x) = x

#it may be that instead of specializing copy, we should just specialize similar
@inline Base.copy(arr::BroadcastedArray) = copy(arr.arg)
@inline Base.Broadcast.materialize(arr::BroadcastedArray) = copy(arr)

@inline Base.copyto!(dst, arr::BroadcastedArray) = copyto!(dst, arr.arg)
@inline Base.Broadcast.materialize!(dst, arr::BroadcastedArray) = copyto!(dst, arr)

#This should do the same thing as Broadcast preprocess does, but apply the BroadcastedArrays preprocess first
@inline Base.Broadcast.preprocess(dst, arr::AbstractArray) = extrude(broadcast_unalias(dst, preprocess(dst, arr)))
function preprocess(dst, arr)
    if iswrapper(arr)
        adopt(preprocess(dst, parent(arr)), arr)
    else
        arr
    end
end
function preprocess(dst, arr::BroadcastedArray{T, N}) where {T, N}
    if arr.arg isa AbstractArray
        return arr
    end
    arg = Base.Broadcast.preprocess(dst, arr.arg)
    return BroadcastedArray{T, N, typeof(arg)}(arg)
end

@inline Base.Broadcast.BroadcastStyle(::Type{BroadcastedArray{T, N, Arg}}) where {T, N, Arg} = BroadcastStyle(Arg)

end
