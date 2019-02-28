module BoxArrays

using Base.Broadcast: Broadcasted, DefaultArrayStyle

export BoxArray

mutable struct BoxArray{T} <: AbstractArray{T, 0}
    val::T
    function BoxArray{T}() where {T}
        new{T}()
    end
    function BoxArray{T}(val) where {T}
        new{T}(val)
    end
end

Base.axes(::BoxArray) = ()
Base.size(::BoxArray) = ()
(Base.getindex(arr::BoxArray{T})::T) where {T} = arr.val
Base.setindex!(arr::BoxArray, x) = (arr.val = x)

end
