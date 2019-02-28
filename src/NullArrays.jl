module NullArrays

using Base.Broadcast: Broadcasted, DefaultArrayStyle

export NullArray

mutable struct NullArray{N, Axes<:Tuple{Vararg{Any, N}}} <: AbstractArray{Any, N}
    axes::Axes
end

Base.axes(arr::NullArray) = arr.axes
Base.size(arr::NullArray) = map(length(axes(arr)))
Base.setindex!(::NullArray, x) = x
Base.copyto!(dst, ::NullArray) = x
function Base.copyto!(dst, src::Broadcasted{Nothing, <:Any, typeof(identity), <:Tuple{<:NullArray}})
    @assert axes(dst) == axes(src)
    return dst
end

end
