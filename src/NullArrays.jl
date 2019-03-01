module NullArrays

using Base.Broadcast: Broadcasted, DefaultArrayStyle

export NullArray

mutable struct NullArray{N, Axes<:Tuple{Vararg{Any, N}}} <: AbstractArray{Any, N}
    axes::Axes
end

Base.axes(arr::NullArray) = arr.axes
Base.size(arr::NullArray) = map(length(axes(arr)))
Base.setindex!(::NullArray, x, i...) = x
Base.setindex!(::NullArray, x, i::Integer) = x
Base.setindex!(::NullArray, x, i::CartesianIndex) = x

end
