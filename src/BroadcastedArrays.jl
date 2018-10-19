using Base: checkbounds_indices, throw_boundserror, tail
using Base.Iterators: repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, broadcast_axes, instantiate, broadcastable, longest_tuple, preprocess

struct BroadcastableArray{T, N, Arg} <: AbstractArray{T, N}
    arg::Arg
    @inline function BroadcastableArray{T, N, Arg}(arg::Arg) where {T, N}
        arg = instantiate(broadcastable(arg))
        @assert typeof(arg) isa Arg
        @assert ndims(typeof(arg)) == N
        @assert T isa eltype(arg)
        return BroadcastableArray{T, N, typeof(arg)}(arg)
    end
end

@inline function BroadcastableArray(arg)
    arg = instantiate(broadcastable(arg))
    return BroadcastableArray{eltype(arg), ndims(typeof(arg)), typeof(arg)}(arg)
end

@inline function BroadcastableArray{T}(arg) where {T}
    arg = instantiate(broadcastable(arg))
    return BroadcastableArray{T, ndims(typeof(arg)), typeof(arg)}(arg)
end

@inline function BroadcastableArray{T, N}(arg) where {T, N}
    arg = instantiate(broadcastable(arg))
    return BroadcastableArray{T, N, typeof(arg)}(arg)
end

function Base.show(io::IO, A::BroadcastableArray{T, N}) where {T, N}
    print(io, BroadcastableArray{T, N}) #Showing the arg type (although maybe useful since it's allowed to differ), will likely be redundant.
    print(io, '(', A.arg, ')')
    nothing
end

broadcastable_array(arg::AbstractArray) = arg
broadcastable_array(arg) = BroadcastableArray(arg)

#The general philosophy of a BroadcastableArray is that it should use broadcast to answer questions unless it's arg is an abstract Array, then it should fall back to the parent
#We can go through and add more base Abstract Array stuff later.

@inline Base.axes(A::BroadcastableArray{T, N}) = axes(a.arg)
@inline Base.axes(A::BroadcastableArray{T, N}, d::Int) = axes(a.arg, d)

@inline Base.eltype(A::BroadcastableArray{T}) where {T} = T

@inline Base.eachindex(A::BroadcastableArray{T, N, <:AbstractArray}) = eachindex(A.arg)
@inline Base.eachindex(A::BroadcastableArray) = _eachindex(axes(A))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices(t)

Base.ndims(::Type{<:BroadcastableArray{T, N}}) where {T, N} = N
Base.ndims(::BroadcastableArray{T, N}) where {T, N} = N

Base.length(A::BroadcastableArray{T, N, <:AbstractArray}) where {T, N} = length(A.arg)
Base.length(A::BroadcastableArray) = prod(map(length, axes(sz)))

Base.@propagate_inbounds Base.getindex(A::BroadcastableArray, I::Int) = getindex(A.arg, I)
Base.@propagate_inbounds Base.getindex(A::BroadcastableArray, I::CartesianIndex) = getindex(A, I)
Base.@propagate_inbounds Base.getindex(A::BroadcastableArray, I::Int...) = getindex(A, I)
Base.@propagate_inbounds Base.getindex(A::BroadcastableArray) = getindex(A.arg)

@inline copy(A::BroadcastableArray{T, N, <:AbstractArray}) where {T, N} = copy(A.arg)
@inline copy(A::BroadcastableArray{T, N, <:Broadcasted}) where {T, N} = copy(A.arg)
@inline copy(A::BroadcastableArray) = identity.(A.arg)
@inline copyto!(dest, A::BroadcastableArray{T, N, <: AbstractArray}) where {T, N} = copyto!(dest, A.arg)
@inline copyto!(dest, A::BroadcastableArray{T, N, <: Broadcasted}) where {T, N} = copyto!(dest, A.arg)
@inline copyto!(dest, A::BroadcastableArray) = dest .= A.arg


#TODO looks something like this.
@inline Base.BroadcastStyle(::Style{Tuple}, A::BroadcastableArray) = BroadcastStyle(A.arg)
Broadcast.longest_tuple(::Nothing, t::Tuple{<:BroadcastableArray{T, N, Arg},Vararg{Any}}) where {T, N, Arg <: Broadcasted{TupleStyle}} = longest_tuple(longest_tuple(nothing, (t[1].arg,)), tail(t))
@inline Base.BroadcastStyle(::DefaultArrayStyle{N}, A::BroadcastableArray{T, N}) where {T, N} = DefaultArrayStyle(Val(N))
@inline Base.BroadcastStyle(::AbstractArrayStyle{N}, A::BroadcastableArray) = AbstractArrayStyle(Val(N))
@inline Base.BroadcastStyle(::BroadcastStyle, A::BroadcastableArray{T, N}) = AbstractArrayStyle(Val(N))
@inline Base.BroadcastStyle(::ArrayConflict, A::BroadcastableArray) = ArrayConflict()


abstract type WrappedArrayConstructor end

@inline Base.Broadcast.broadcasted(style::BroadcastStyle, C::WrappedArrayConstructor, args) = C(map(broadcastable_array, args)...)

"""
operating on lazy broadcast expressions, arrays,
tuples, collections, [`Ref`](@ref)s and/or scalars `As`
"""

end
