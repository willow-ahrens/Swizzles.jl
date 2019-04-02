module ValArrays

using Swizzles.ArrayifiedArrays
using Swizzles.WrapperArrays
using Base.Broadcast: Broadcasted, BroadcastStyle

export ValArray
export lift_vals

mutable struct ValArray{T, c} <: AbstractArray{T, 0}
    @inline function ValArray{T, c}() where {T, c}
        @assert c isa T
        new{T, c}()
    end
end
@inline function ValArray{T}(c) where {T}
    ValArray{T, c}()
end
@inline function ValArray(c)
    ValArray{typeof(c), c}()
end

Base.axes(::ValArray) = ()
Base.size(::ValArray) = ()
@inline (Base.getindex(arr::ValArray{T, c})::T) where {T, c} = c
@inline (Base.getindex(arr::ValArray{T, c}, ::Integer)::T) where {T, c} = c

@inline function lift_vals(x)
    if axes(x) == ()
        c = x[]
        if isbits(c)
            return ValArray(c)
        end
    end
    return x
end
@inline lift_vals(arr::ArrayifiedArray{T, N}) where {T, N} = ArrayifiedArray{T, N}(lift_vals(arr.arg))
@inline function lift_vals(arr::AbstractArray)
    if iswrapper(arr)
        return adopt(lift_vals(parent(arr)), arr)
    else
        return arr
    end
end
@inline lift_vals(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, map(lift_vals, bc.args))

end
