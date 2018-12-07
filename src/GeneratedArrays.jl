module GeneratedArrays

abstract type GeneratedArray{T, N} <: AbstractArray{T, N} end

using Base.Broadcast: Broadcasted
using Base.Broadcast: broadcast_axes, instantiate

export GeneratedArray

"""
    GeneratedArray <: AbstractArray

A convenience type for defining array types that are mostly implemented using
copyto!(::Any, ::Broadcasted) and copy(::Broadcasted). Consult the [Interfaces
chapter](@ref man-interfaces-broadcasting) on broadcasting for more info about
broadcast. Many Base functions are implemented for GeneratedArrays in terms of
`copyto!` including `map`, `getindex`, and `foreach`.

See also: [`copy`](@ref), [`copyto!`](@ref)
"""
abstract type GeneratedArray{T, N} <: AbstractArray{T, N} end

@inline Base.Broadcast.materialize(A::GeneratedArray) = copy(A)
@inline Base.Broadcast.materialize!(dst, A::GeneratedArray) = copyto!(dst, A)

#Beware infinite recursion!

Base.copyto!(dst, src::GeneratedArray) = copyto!(dst, Array(src))

Base.copyto!(dst::GeneratedArray, src) = _copyto!(dst, broadcastable(src))
Base.copyto!(dst::GeneratedArray, src::AbstractArray) = _copyto!(dst, src)
Base.copyto!(dst::AbstractArray, src::GeneratedArray) = _copyto!(dst, src)
Base.copyto!(dst::GeneratedArray, src::GeneratedArray) = _copyto!(dst, src)

Base.copyto!(dst::GeneratedArray, src::Broadcasted) = invoke(copyto!, Tuple{AbstractArray, typeof(src)}, dst, src)
totallynotidentity(x) = x
function _copyto!(dst::AbstractArray, src)
    if axes(dst) != broadcast_axes(src)
        copyto!(reshape(dst, broadcast_axes(src)), instantiate(Broadcasted(totallynotidentity, (src,))))
    else
        copyto!(dst, instantiate(Broadcasted(totallynotidentity, (src,))))
    end
end

#=

#do overrides for wierd copyto!s, map, foreach, etc...

struct NullArray{N} <: AbstractArray{<:Any, N}
    axes::NTuple{N}
end

Base.axes(arr::NullArray) = arr.axes
Base.setindex!(arr, val, inds...) = val

Base.foreach(f, a::GeneratedArray) = assign!(NullArray(axes(a)), a)
function BroadcastedArrays.assign!(dst::NullArray, MetaArray(op, arg)) #foreach

=#

end
