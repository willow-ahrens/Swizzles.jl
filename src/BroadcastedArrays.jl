using Base: checkbounds_indices, throw_boundserror, tail
using Base.Iterators: repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, broadcast_axes, instantiate, broadcastable, preprocess, _broadcast_getindex

struct BroadcastedArray{T, N, Arg} <: AbstractArray{T, N}
    arg::Arg
    @inline function BroadcastedArray{T, N, Arg}(arg::Arg) where {T, N, Arg}
        arg = instantiate(broadcastable(arg))
        @assert typeof(arg) <: Arg
        @assert ndims(typeof(arg)) == N
        @assert T <: eltype(arg)
        return new{T, N, typeof(arg)}(arg)
    end
    @inline function BroadcastedArray{T, 1, Arg}(arg::Arg) where {T, Arg <: Tuple}
        arg = instantiate(broadcastable(arg))
        @assert typeof(arg) <: Arg
        @assert T <: eltype(arg)
        return new{T, 1, typeof(arg)}(arg)
    end
end

@inline function BroadcastedArray(arg)
    arg = instantiate(broadcastable(arg))
    return BroadcastedArray{eltype(arg), ndims(typeof(arg)), typeof(arg)}(arg)
end

@inline function BroadcastedArray(arg::Tuple)
    return BroadcastedArray{eltype(arg), 1, typeof(arg)}(arg)
end

@inline function BroadcastedArray{T}(arg) where {T}
    arg = instantiate(broadcastable(arg))
    return BroadcastedArray{T, ndims(typeof(arg)), typeof(arg)}(arg)
end

@inline function BroadcastedArray{T}(arg::Tuple) where {T}
    return BroadcastedArray{T, 1, typeof(arg)}(arg)
end

@inline function BroadcastedArray{T, N}(arg) where {T, N}
    arg = instantiate(broadcastable(arg))
    return BroadcastedArray{T, N, typeof(arg)}(arg)
end

@inline function BroadcastedArray{T, 1}(arg::Tuple) where {T, N}
    return BroadcastedArray{T, 1, typeof(arg)}(arg)
end

function Base.show(io::IO, A::BroadcastedArray{T, N}) where {T, N}
    print(io, BroadcastedArray{T, N}) #Showing the arg type (although maybe useful since it's allowed to differ), will likely be redundant.
    print(io, '(', A.arg, ')')
    nothing
end

arrayify(arg::AbstractArray) = arg
arrayify(arg) = BroadcastedArray(arg)

#The general philosophy of a BroadcastedArray is that it should use broadcast to answer questions unless it's arg is an abstract Array, then it should fall back to the parent
#We can go through and add more base Abstract Array stuff later.

@inline Base.axes(A::BroadcastedArray) = broadcast_axes(A.arg)

@inline Base.size(A::BroadcastedArray) = map(length, axes(A.arg))

@inline Base.eltype(A::BroadcastedArray{T}) where {T} = T

@inline Base.eachindex(A::BroadcastedArray{T, N, <:AbstractArray}) where {T, N} = eachindex(A.arg)
@inline Base.eachindex(A::BroadcastedArray) = _eachindex(axes(A))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices(t)

Base.ndims(::Type{<:BroadcastedArray{T, N}}) where {T, N} = N
Base.ndims(::BroadcastedArray{T, N}) where {T, N} = N

Base.length(A::BroadcastedArray{T, N, <:AbstractArray}) where {T, N} = length(A.arg)
Base.length(A::BroadcastedArray) = prod(map(length, axes(A)))

Base.@propagate_inbounds Base.getindex(A::BroadcastedArray, I::Int) = _broadcast_getindex(A.arg, I)
Base.@propagate_inbounds Base.getindex(A::BroadcastedArray, I::CartesianIndex) = _broadcast_getindex(A.arg, I)
Base.@propagate_inbounds Base.getindex(A::BroadcastedArray, I::Int...) = _broadcast_getindex(A.arg, CartesianIndex(I))
Base.@propagate_inbounds Base.getindex(A::BroadcastedArray) = _broadcast_getindex(A.arg)

@inline myidentity(x) = x

@inline Base.copy(A::BroadcastedArray{T, N, <:AbstractArray}) where {T, N} = copy(A.arg)
@inline Base.copy(A::BroadcastedArray{T, N, <:Broadcasted}) where {T, N} = copy(A.arg)
@inline Base.copy(A::BroadcastedArray) = copy(instantiate(Broadcasted(myidentity, (A,))))
@inline Base.Broadcast.materialize(A::BroadcastedArray) = copy(A)

@inline Base.copyto!(dest, A::BroadcastedArray{T, N, <: AbstractArray}) where {T, N} = copyto!(dest, A.arg)
@inline Base.copyto!(dest, A::BroadcastedArray{T, N, <: Broadcasted}) where {T, N} = copyto!(dest, A.arg)
@inline Base.copyto!(dest, A::BroadcastedArray) = copyto!(dest, instantiate(Broadcasted(myidentity, (A,))))
@inline Base.Broadcast.materialize!(dest, A::BroadcastedArray) = copyto!(dest, A)

function Base.Broadcast.preprocess(dest, A::BroadcastedArray{T, N}) where {T, N}
    arg = preprocess(dest, A.arg)
    BroadcastedArray{T, N, typeof(arg)}(arg)
end

@inline Broadcast.BroadcastStyle(A::Type{BroadcastedArray{T, N, Arg}}) where {T, N, Arg} = BroadcastStyle(Arg)

abstract type WrappedArrayConstructor end

@inline Base.Broadcast.broadcasted(style::BroadcastStyle, C::WrappedArrayConstructor, args...) = C(map(arrayify, args)...)

