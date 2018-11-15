using Base.Broadcast: broadcast_axes

abstract type WrappedArray{T, N, P} <: AbstractArray{T, N} end

parenttype(::Type{<:WrappedArray{<:Any, <:Any, P}}) where {P} = P
parenttype(::WrappedArray{<:Any, <:Any, P}) where {P} = P

storagetype(arr_type::Type{<:WrappedArray}) = storagetype(parenttype(arr_type))
storagetype(arr::WrappedArray) = storagetype(parenttype(arr))
storagetype(arr) = arr

storage(arr::WrappedArray) = storage(parent(arr))
storage(arr) = arr

Base.eltype(::Type{<:WrappedArray{T}}) where {T} = T
Base.eltype(::WrappedArray{T}) where {T} = T

Base.ndims(::Type{<:WrappedArray{<:Any, N}}) where {N} = N
Base.ndims(::WrappedArray{<:Any, N}) where {N} = N

Base.size(arr::WrappedArray{<:Any, <:Any, <:AbstractArray}) = size(parent(arr))
#Base.size(arr::WrappedArray) = map(length, broadcast_axes(parent(arr))) #is this the right thing to do?

Base.axes(arr::WrappedArray{<:Any, <:Any, <:AbstractArray}) = axes(parent(arr))
#Base.axes(arr::WrappedArray) = broadcast_axes(parent(arr)) #is this the right thing to do?

Base.getindex(arr::WrappedArray, inds...) = getindex(parent(arr), inds...)

Base.setindex!(arr::WrappedArray, val, inds...) = setindex!(parent(arr), val, inds...)

@inline Broadcast.BroadcastStyle(arr::Type{<:WrappedArray}) = BroadcastStyle(parenttype(arr))

#=
function Base.show(io::IO, arr::WrappedArray{T, N, P}) where {T, N, P}
    print(io, arr, )
    print(io, '(', A.arg, ')')
    nothing
end

function Base.Broadcast.preprocess(dest, A::BroadcastedArray{T, N}) where {T, N}
    arg = preprocess(dest, A.arg)
    BroadcastedArray{T, N, typeof(arg)}(arg)
end
=#

