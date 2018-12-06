module MetaArrays

abstract type MetaArray{T, N} <: AbstractArray{T, N} end

"""
    MetaArray <: AbstractArray

An AbstractArray that is mostly implemented using copyto!(::Any, ::Broadcasted).

See also: [`parent`](@ref)
"""
abstract type MetaArray{T, N} <: AbstractArray{T, N} end

@inline Base.Broadcast.materialize(A::MetaArray) = copy(A)
@inline Base.Broadcast.materialize!(dst, A::MetaArray) = copyto!(dst, A)

Base.copyto!(dst, src::MetaArray) = copyto!(dst, Array(src))

Base.copyto!(dst::MetaArray, src) = _copyto!(dst, broadcastable(src))
Base.copyto!(dst::MetaArray, src::AbstractArray) = _copyto!(dst, src)
Base.copyto!(dst::AbstractArray, src::MetaArray) = _copyto!(dst, src)
Base.copyto!(dst::MetaArray, src::MetaArray) = _copyto!(dst, src)

Base.copyto!(dst::MetaArray, src::Broadcasted) = invoke(copyto!, Tuple{AbstractArray, typeof(src)}, dst, src)
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

Base.foreach(f, a::MetaArray) = assign!(NullArray(axes(a)), a)

=#

end
