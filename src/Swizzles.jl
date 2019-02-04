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



"""
    `swizzle(A, mask, op=nooperator)`

Create a new object `B` such that the dimension `i` of `A` is mapped to
dimension `mask[i]` of `B`. If `mask[i]` is an instance of the singleton type
`Drop`, the dimension is reduced over using `op`. `mask` may be any (possibly
infinite) iterable over elements of type `Int` and `Drop`. The integers in
`mask` must be unique, and if `mask` is not long enough, additional `Drop`s are
added to the end.
The resulting container type from `materialize(B)` is established by the following rules:
 - If all elements of `mask` are `Drop`, it returns an unwrapped scalar.
 - All other combinations of arguments default to returning an `Array`, but
   custom container types can define their own implementation rules to
   customize the result when they appear as an argument.
The swizzle operation is represented with a special lazy `SwizzledArray` type.
`swizzle` results in `materialize(SwizzledArray(...))`.  The swizzle operation can use the
`Swizzle` type to take advantage of special broadcast syntax. A statement like:
```
   y = Swizzle((1,), +).(x .* (Swizzle((2, 1)).x .+ 1))
```
will result in code that is essentially:
```
   y = materialize(SwizzledArray(BroadcastedArray(Broadcasted(*, SwizzledArray(x, (2, 1)), Broadcasted(+, x, 1))), (1,), +))
```
If `SwizzledArray`s are mixed with `Broadcasted`s, the result is fused into one big operation.

See also: [`swizzle!`](@ref), [`Swizzle`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> swizzle(A, (1,), +)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> swizzle(A, (), +)
55
julia> swizzle(parse.(Int, ["1", "2"]), (2,))
1x2-element Array{Int64,1}:
 1 2
```
"""
swizzle(A, mask, op=nooperator) = Swizzle(mask, op).(A)

"""
    `swizzle!(dest, A, mask, op=nooperator)`

Like [`swizzle`](@ref), but store the result of `swizzle(A, mask, op)` in the
`dest` type.  Results in `materialize!(dest, SwizzledArray(...))`.

See also: [`swizzle`](@ref), [`Swizzle`](@ref).

# Examples
```jldoctest
julia> B = [1; 2; 3; 4; 5]
5x1-element Array{Int64,1}:
 1
 2
 3
 4
 5
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> swizzle!(B, A, (1,), +)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> B
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
```
"""
swizzle!(dest, A, mask, op=nooperator) = dest .= Swizzle(mask, op).(A)

end
