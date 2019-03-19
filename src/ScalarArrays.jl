module ScalarArrays

using Base.Broadcast: Broadcasted, DefaultArrayStyle

export ScalarArray

mutable struct ScalarArray{T} <: AbstractArray{T, 0}
    val::T
    function ScalarArray{T}() where {T}
        new{T}()
    end
    function ScalarArray{T}(val) where {T}
        new{T}(val)
    end
end

Base.axes(::ScalarArray) = ()
Base.size(::ScalarArray) = ()
(Base.getindex(arr::ScalarArray{T})::T) where {T} = arr.val
Base.setindex!(arr::ScalarArray, x) = (arr.val = x)

end
