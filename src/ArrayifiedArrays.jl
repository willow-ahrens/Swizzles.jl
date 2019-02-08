module ArrayifiedArrays

using Swizzles.Properties
using Swizzles.WrapperArrays
using Swizzles.GeneratedArrays
using Swizzles

using Base: checkbounds_indices, throw_boundserror, tail, dataids, unaliascopy, unalias
using Base.Iterators: repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, instantiate, broadcastable, _broadcast_getindex, combine_eltypes, extrude, broadcast_unalias

export ArrayifiedArray, arrayify

struct ArrayifiedArray{T, N, Arg} <: GeneratedArray{T, N}
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
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, Broadcasted}, I::Int) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, Broadcasted}, I::CartesianIndex) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, Broadcasted}, I::Int...) = _broadcast_getindex(arr.arg, CartesianIndex(I))
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray{<:Any, <:Any, Broadcasted}) = error("FIXME")
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I...) = getindex(arr.arg, I...)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::Int) = getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::CartesianIndex) = getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::Int...) = getindex(arr.arg, I...)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray) = getindex(arr.arg)

@inline myidentity(x) = x

#it may be that instead of specializing copy, we should just specialize similar
@inline Base.copy(arr::ArrayifiedArray) = copy(arr.arg)
@inline Base.Broadcast.materialize(arr::ArrayifiedArray) = copy(arr)

@inline Base.copyto!(dst, arr::ArrayifiedArray) = copyto!(dst, arr.arg)
@inline Base.Broadcast.materialize!(dst, arr::ArrayifiedArray) = copyto!(dst, arr)

#This should do the same thing as Broadcast preprocess does, but apply the ArrayifiedArrays preprocess first
@inline Base.Broadcast.preprocess(dst, arr::AbstractArray) = extrude(broadcast_unalias(dst, preprocess(dst, arr)))
function preprocess(dst, arr)
    if iswrapper(arr)
        adopt(preprocess(dst, parent(arr)), arr)
    else
        arr
    end
end
function preprocess(dst, arr::ArrayifiedArray{T, N}) where {T, N}
    if arr.arg isa AbstractArray
        return arr
    end
    arg = Base.Broadcast.preprocess(dst, arr.arg)
    return ArrayifiedArray{T, N, typeof(arg)}(arg)
end

@inline Base.Broadcast.BroadcastStyle(::Type{ArrayifiedArray{T, N, Arg}}) where {T, N, Arg} = BroadcastStyle(Arg)

end
