module BroadcastedArrays

using Base: checkbounds_indices, throw_boundserror, tail, dataids, unaliascopy, unalias
using Base.Iterators: repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, broadcast_axes, instantiate, broadcastable, preprocess, _broadcast_getindex, combine_eltypes, extrude, broadcast_unalias
using Swizzle.WrapperArrays

export ArrayifiedArray, Arrayifier, arrayify
export BroadcastedArray

abstract type BroadcastedArray{T, N} <: AbstractArray{T, N} end

struct ArrayifiedArray{T, N, Arg} <: BroadcastedArray{T, N}
    arg::Arg
    @inline function ArrayifiedArray{T, N, Arg}(arg::Arg) where {T, N, Arg}
        arg = instantiate(broadcastable(arg))
        @assert typeof(arg) <: Arg
        @assert ndims(typeof(arg)) == N
        @assert T <: eltype(arg)
        return new{T, N, typeof(arg)}(arg)
    end
    @inline function ArrayifiedArray{T, 1, Arg}(arg::Arg) where {T, Arg <: Tuple}
        arg = instantiate(broadcastable(arg))
        @assert typeof(arg) <: Arg
        @assert T <: eltype(arg)
        return new{T, 1, typeof(arg)}(arg)
    end
end


@inline function ArrayifiedArray(arg)
    arg = instantiate(broadcastable(arg))
    return ArrayifiedArray{eltype(arg), ndims(typeof(arg)), typeof(arg)}(arg)
end

@inline function ArrayifiedArray(arg::Broadcasted)
    arg = instantiate(arg)
    return ArrayifiedArray{combine_eltypes(arg.f, arg.args), ndims(typeof(arg)), typeof(arg)}(arg)
end

@inline function ArrayifiedArray(arg::Tuple)
    return ArrayifiedArray{eltype(arg), 1, typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T}(arg) where {T}
    arg = instantiate(broadcastable(arg))
    return ArrayifiedArray{T, ndims(typeof(arg)), typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T}(arg::Tuple) where {T}
    return ArrayifiedArray{T, 1, typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T, N}(arg) where {T, N}
    arg = instantiate(broadcastable(arg))
    return ArrayifiedArray{T, N, typeof(arg)}(arg)
end

@inline function ArrayifiedArray{T, 1}(arg::Tuple) where {T, N}
    return ArrayifiedArray{T, 1, typeof(arg)}(arg)
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
WrapperArrays.iswrapper(arr::ArrayifiedArray) = false

Base.dataids(arr::ArrayifiedArray) = dataids(arr.arg)
Base.unaliascopy(arr::ArrayifiedArray{T, N, Arg}) where {T, N, Arg} = ArrayifiedArray{T, N, Arg}(unaliascopy(arr.arg))
Base.unalias(dest, arr::ArrayifiedArray{T, N, Arg}) where {T, N, Arg} = ArrayifiedArray{T, N, Arg}(unalias(dest, arr.arg))

@inline Base.axes(arr::ArrayifiedArray) = broadcast_axes(arr.arg)

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

Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::Int) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::CartesianIndex) = _broadcast_getindex(arr.arg, I)
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray, I::Int...) = _broadcast_getindex(arr.arg, CartesianIndex(I))
Base.@propagate_inbounds Base.getindex(arr::ArrayifiedArray) = getindex(arr.arg)

@inline myidentity(x) = x

@inline Base.copy(arr::ArrayifiedArray{T, N, <:AbstractArray}) where {T, N} = copy(arr.arg)
@inline Base.copy(arr::ArrayifiedArray{T, N, <:Broadcasted}) where {T, N} = copy(arr.arg)
@inline Base.copy(arr::ArrayifiedArray) = copy(instantiate(Broadcasted(myidentity, (arr,))))
@inline Base.Broadcast.materialize(arr::ArrayifiedArray) = copy(arr)

@inline Base.copyto!(dest, arr::ArrayifiedArray{T, N, <: AbstractArray}) where {T, N} = copyto!(dest, arr.arg)
@inline Base.copyto!(dest, arr::ArrayifiedArray{T, N, <: Broadcasted}) where {T, N} = copyto!(dest, arr.arg)
@inline Base.copyto!(dest, arr::ArrayifiedArray) = copyto!(dest, instantiate(Broadcasted(myidentity, (arr,))))
@inline Base.Broadcast.materialize!(dest, arr::ArrayifiedArray) = copyto!(dest, arr)

@inline Base.Broadcast.preprocess(dest, arr::AbstractArray) = extrude(broadcast_unalias(dest, preprocess_storage(dest, arr)))
function preprocess_storage(dest, arr)
    if iswrapper(arr)
        adopt(preprocess_storage(dest, parent(arr)), arr)
    else
        arr
    end
end
function preprocess_storage(dest, arr::ArrayifiedArray{T, N}) where {T, N}
    arg = preprocess(dest, arr.arg)
    x = ArrayifiedArray{T, N, typeof(arg)}(arg)
end

@inline Base.Broadcast.BroadcastStyle(::Type{ArrayifiedArray{T, N, Arg}}) where {T, N, Arg} = BroadcastStyle(Arg)

abstract type Arrayifier end

@inline Base.Broadcast.broadcasted(style::BroadcastStyle, cstr::Arrayifier, args...) = cstr(map(arrayify, args)...)

#Now we start BroadcastedArrays

"""
    BroadcastedArray <: AbstractArray

An AbstractArray that is mostly implemented using Broadcast.

See also: [`parent`](@ref)
"""
abstract type BroadcastedArray{T, N} <: AbstractArray{T, N} end

Base.copyto!(dst::BroadcastedArray, src) = _copyto!(dst, arrayify(src))
Base.copyto!(dst, src::BroadcastedArray) = _copyto!(arrayify(dst), src)
Base.copyto!(dst::BroadcastedArray, src::BroadcastedArray) = _copyto!(dst, src)
function _copyto!(dst, src)
    if axes(dst) != axes(src)
        assign!(reshape(dst, :), reshape(src, :))
    else
        assign!(dst, src)
    end
end

Base.copy!(dst::BroadcastedArray, src) = _copy!(dst, arrayify(src))
Base.copy!(dst, src::BroadcastedArray) = _copy!(arrayify(dst), src)
Base.copy!(dst::BroadcastedArray, src::BroadcastedArray) = _copy!(dst, src)
function _copy!(dst, src)
    if axes(dst) != axes(src)
        throw(ArgumentError("axes of $dst and $src do not match."))
    else
        assign!(dst, src)
    end
end

#=
struct NullArray{N} <: AbstractArray{<:Any, N}
    axes::NTuple{N}
end

Base.axes(arr::NullArray) = arr.axes
Base.setindex!(arr, val, inds...) = val

Base.foreach(f, a::BroadcastedArray) = assign!(NullArray(axes(a)), a)

=#

end
