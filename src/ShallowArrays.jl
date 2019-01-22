module ShallowArrays

using Swizzles.WrapperArrays

using Base.Broadcast: broadcast_axes, BroadcastStyle
using Base: dataids, unaliascopy, unalias

export ShallowArray

"""
    ShallowArray

A convenience type for constructing simple wrapper arrays that behave almost
exactly like their parent. Provides default implementations of `AbstractArray`
methods. Subtypes of `ShallowArray` must define [`parent`](@ref) and
[`adopt`](@ref)

See also: [`parent`](@ref), [`adopt`](@ref)
"""
abstract type ShallowArray{T, N, Arg} <: AbstractArray{T, N} end

Base.parent(arr::ShallowArray) = throw(MethodError(parent, (arr)))

iswrapper(arr::ShallowArray) = true

IndexStyle(arr::ShallowArray) = IndexStyle(parent(arr))

Base.dataids(arr::ShallowArray) = dataids(parent(arr))
Base.unaliascopy(arr::A) where {A <:ShallowArray} = adopt(unaliascopy(parent(arr)), arr)::A
Base.unalias(dest, arr::A) where {A <:ShallowArray} = adopt(unalias(dest, arr.arg), arr)::A

Base.eltype(::Type{<:ShallowArray{T}}) where {T} = T
Base.eltype(::ShallowArray{T}) where {T} = T

Base.ndims(::Type{<:ShallowArray{<:Any, N}}) where {N} = N
Base.ndims(::ShallowArray{<:Any, N}) where {N} = N

Base.size(arr::ShallowArray{<:Any, <:Any, <:AbstractArray}) = size(parent(arr))

Base.axes(arr::ShallowArray{<:Any, <:Any, <:AbstractArray}) = axes(parent(arr))

Base.getindex(arr::ShallowArray, inds...) = getindex(parent(arr), inds...)

Base.setindex!(arr::ShallowArray, val, inds...) = setindex!(parent(arr), val, inds...)

@inline Broadcast.BroadcastStyle(arr::Type{<:ShallowArray{<:Any, <:Any, Arg}}) where {Arg} = BroadcastStyle(Arg)

function Base.show(io::IO, arr::ShallowArray{T, N, Arg}) where {T, N, Arg}
    print(io, typeof(arr))
    print(io, '(', parent(A), ')')
    nothing
end

end
