module Swizzles

export Drop, drop
export masktuple, imasktuple #maybe dont export...
export BroadcastedArray, arrayify
export SwizzledArray, mask
export swizzle, swizzle!
export Swizzle, Reduce, Sum, Max, Min, Beam
export SwizzleTo, ReduceTo, SumTo, MaxTo, MinTo, BeamTo
export Delay, Intercept


export nooperator

include("util.jl")
include("properties.jl")

include("WrapperArrays.jl")
include("GeneratedArrays.jl")
include("BroadcastedArrays.jl")
include("ShallowArrays.jl")
include("ExtrudedArrays.jl")
include("SwizzledArrays.jl")



struct Swizzle{T, Op, _mask} <: Swizzles.Intercept
    op::Op
end

@inline Swizzle(op, _mask...) = Swizzle{nothing}(op, _mask...)
@inline Swizzle{T}(op::Op, _mask...) where {T, Op} = Swizzle{T, Op, _mask}(op)
@inline Swizzle{T}(op::Op, _mask::Tuple) where {T, Op} = Swizzle{T, Op, _mask}(op)
@inline Swizzle{T}(op::Op, ::Val{_mask}) where {T, Op, _mask} = Swizzle{T, Op, _mask}(op)

@inline (sz::Swizzle)(arg) = sz(BroadcastedArray(arg))
@inline function(sz::Swizzle{T, Op, _mask})(arg::AbstractArray) where {T, Op, _mask}
    return SwizzledArray{T}(arg, sz.op, Val(_mask))
end



struct Beam{T} end

@inline function Beam(_mask...)
    Swizzle(nooperator, _mask...)
end
@inline function Beam{T}(_mask...) where {T}
    Swizzle{T}(nooperator, _mask...)
end



struct SwizzleTo{T, Op, _imask} <: Swizzles.Intercept
    op::Op
end

@inline SwizzleTo(op, _imask...) = SwizzleTo{nothing}(op, _imask...)
@inline SwizzleTo{T}(op::Op, _imask...) where {T, Op} = SwizzleTo{T, Op, _imask}(op)
@inline SwizzleTo{T}(op::Op, _imask::Tuple) where {T, Op} = SwizzleTo{T, Op, _imask}(op)
@inline SwizzleTo{T}(op::Op, ::Val{_imask}) where {T, Op, _imask} = SwizzleTo{T, Op, _imask}(op)

@inline (sz::SwizzleTo)(arg) = sz(BroadcastedArray(arg))
@inline function(sz::SwizzleTo{T, Op, _imask})(arg::Arg) where {T, Op, _imask, Arg <: AbstractArray}
    if @generated
        mask = parse_swizzle_mask(arg, imasktuple(d->drop, identity, _imask))
        return :(return SwizzledArray{T}(arg, sz.op, $(Val(mask))))
    else
        return SwizzledArray{T}(arg, sz.op, imasktuple(d->drop, identity, _imask))
    end
end



struct BeamTo{T} end

@inline function BeamTo(_imask...)
    Swizzle(nooperator, _imask...)
end
@inline function BeamTo{T}(_imask...) where {T}
    Swizzle{T}(nooperator, _imask...)
end



struct Reduce{T, Op, dims} <: Swizzles.Intercept
    op::Op
end

@inline Reduce(op, dims...) = Reduce{nothing}(op, dims...)
@inline Reduce{T}(op::Op, dims...) where {T, Op} = Reduce{T, Op, dims}(op)
@inline Reduce{T}(op::Op, dims::Tuple) where {T, Op} = Reduce{T, Op, dims}(op)
@inline Reduce{T}(op::Op, ::Val{dims}) where {T, Op, dims} = Reduce{T, Op, dims}(op)

@inline (rd::Reduce)(arg) = rd(BroadcastedArray(arg))
@inline function parse_reduce_mask(arr, dims::Tuple{Vararg{Int}}) where {M, N}
    c = 0
    return ntuple(d -> d in dims ? drop : c += 1, Val(ndims(arr)))
end
@inline function parse_reduce_mask(arr, dims::Tuple{}) where {M, N}
    return ntuple(d -> drop, Val(ndims(arr)))
end
@inline function(rd::Reduce{T, <:Any, dims})(arg::AbstractArray{<:Any, N}) where {T, dims, N}
    if @generated
        mask = parse_reduce_mask(arg, dims)
        return :(return SwizzledArray{T}(arg, rd.op, $(Val(mask))))
    else
        mask = parse_reduce_mask(arg, dims)
        return SwizzledArray{T}(arg, rd.op, mask)
    end
end



struct Sum{T} end

@inline function Sum(dims...)
    Reduce(+, dims...)
end

@inline function Sum{T}(dims...) where {T}
    Reduce{T}(+, dims...)
end



struct Max{T} end
@inline function Max(dims...)
    Reduce(max, dims...)
end

@inline function Max{T}(dims...) where {T}
    Reduce{T}(max, dims...)
end



struct Min{T} end

@inline function Min(dims...)
    Reduce(min, dims...)
end

@inline function Min{T}(dims...) where {T}
    Reduce{T}(min, dims...)
end

end
