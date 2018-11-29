module WrapperArrays

using Base.Broadcast: broadcast_axes, BroadcastStyle

export WrapperArray

export parenttype, storagetype, storage, map_parent, map_storage

abstract type WrapperArray{T, N, P} <: AbstractArray{T, N} end

#In the future, WrapperArray should just be a convenience abstract type for pass-through arrays, and map_parent, map_storage, parenttype, storagetype, and storage should be
#functions that live in Adapt

#To be clear, map_storage is implemented by map_parent. A wrapper array should at a minimum define parent and map_parent.

#=
To be a wrapper, define:
  parent
  mapparent
  iswrapper






  parenttype(x) = typeof(parent(x))
  storagetype(x) = typeof(storage(x))

  parent(::Typeof{WrapperArray})
=#


map_parent(f, x) = throw(MethodError(map_parent, (f, x)))
map_storage(f::F, arr)
  if iswrapper(arr)
    map_parent(x->map_storage(f, x), arr)
  else
    f(arr)
  end
end

parenttype(::Type{<:WrapperArray{<:Any, <:Any, P}}) where {P} = P
parenttype(::WrapperArray{<:Any, <:Any, P}) where {P} = P

storagetype(arr_type::Type{<:WrapperArray}) = storagetype(parenttype(arr_type))
storagetype(arr::WrapperArray) = storagetype(parenttype(arr))
storagetype(arr) = arr

storage(arr::WrapperArray) = storage(parent(arr))
#FIXME this line should probably always call parent and then check if the parent is the array itself do determine if we should stop recursing
storage(arr) = arr

Base.eltype(::Type{<:WrapperArray{T}}) where {T} = T
Base.eltype(::WrapperArray{T}) where {T} = T

Base.ndims(::Type{<:WrapperArray{<:Any, N}}) where {N} = N
Base.ndims(::WrapperArray{<:Any, N}) where {N} = N

Base.size(arr::WrapperArray{<:Any, <:Any, <:AbstractArray}) = size(parent(arr))
#Base.size(arr::WrapperArray) = map(length, broadcast_axes(parent(arr))) #is this the right thing to do?

Base.axes(arr::WrapperArray{<:Any, <:Any, <:AbstractArray}) = axes(parent(arr))
#Base.axes(arr::WrapperArray) = broadcast_axes(parent(arr)) #is this the right thing to do?

Base.getindex(arr::WrapperArray, inds...) = getindex(parent(arr), inds...)

Base.setindex!(arr::WrapperArray, val, inds...) = setindex!(parent(arr), val, inds...)

@inline Broadcast.BroadcastStyle(arr::Type{<:WrapperArray}) = BroadcastStyle(parenttype(arr))

#=
function Base.show(io::IO, arr::WrapperArray{T, N, P}) where {T, N, P}
    print(io, arr, )
    print(io, '(', A.arg, ')')
    nothing
end

function Base.Broadcast.preprocess(dest, A::BroadcastedArray{T, N}) where {T, N}
    arg = preprocess(dest, A.arg)
    BroadcastedArray{T, N, typeof(arg)}(arg)
end
=#

end
