module Swizzles

export Drop, drop
export masktuple, imasktuple #maybe dont export...
export BroadcastedArray, arrayify
export SwizzledArray, mask
export swizzle, swizzle!
export Swizzle, Reduce, Sum, Max, Min, Beam
export SwizzleTo, ReduceTo, SumTo, MaxTo, MinTo, BeamTo
export Delay, Intercept

include("util.jl")
include("properties.jl")

include("WrapperArrays.jl")
include("GeneratedArrays.jl")
include("BroadcastedArrays.jl")
include("ShallowArrays.jl")
include("ExtrudedArrays.jl")
include("SwizzledArrays.jl")



struct Swizzle{T, _mask, Op} <: Swizzles.Intercept
    op::Op
end

"""
    `Swizzle(mask, op=nooperator)`

Produce an object `s` such that when `s` is broadcasted as a function over an
argument `arg`, the result is a lazy view of the result of `swizzle(arg, mask,
op)`.

See also: [`swizzle`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Swizzle((1,), +).(A)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> Swizzle((), +).(A)
55
julia> Swizzle((2,)).(parse.(Int, ["1", "2"]))
1x2-element Array{Int64,1}:
 1 2
```
"""
@inline Swizzle(_mask, op::Op) where {Op} = Swizzle{nothing}(_mask, op)

"""
    `Swizzle{T}(mask, op=nooperator)`

Similar to [`Swizzle`](@ref), but the eltype of the resulting swizzle is
declared to be `T`.

See also: [`Swizzle`](@ref).
"""
@inline Swizzle{T}(_mask, op::Op) where {T, Op} = Swizzle{T, _mask, Op}(op)

@inline (sz::Swizzle)(arg) = sz(BroadcastedArray(arg))

@inline function(sz::Swizzle{nothing, _mask})(arg::AbstractArray{<:Any, N}) where {_mask, N}
    return SwizzledArray(arg, Val(_mask), sz.op)
end
@inline function(sz::Swizzle{T, _mask})(arg::AbstractArray{<:Any, N}) where {T, _mask, N}
    return SwizzledArray{T}(arg, Val(_mask), sz.op)
end



struct Beam{T} end

"""
    `Beam(mask...)`

Produce an object `s` such that when `s` is broadcasted as a function over an
argument `arg`, the dimension `arg[i]` appears as dimension `mask[i]` in the
output. If dimension `i` is known to have size `1`, it may be dropped by setting
`mask[i] = drop`.

See also: [`Swizzle`](@ref).

# Examples
```jldoctest
julia> A = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> Beam(drop, 3).(A)
1×1×5 Array{Int64,3}:
[:, :, 1] =
 1
[:, :, 2] =
 2
[:, :, 3] =
 3
[:, :, 4] =
 4
[:, :, 5] =
 5
```
"""
function Beam(_mask::Union{Int, Drop}...)
    Swizzle(_mask, nooperator)
end

"""
    `Beam{T}(mask...)`

Similar to [`Beam`](@ref), but the eltype of the resulting swizzle is
declared to be `T`.

See also: [`Beam`](@ref).
"""
function Beam{T}(_mask::Union{Int, Drop}...) where {T}
    Swizzle{T}(_mask, nooperator)
end



struct Reduce{T, Op, dims} <: Swizzles.Intercept
    op::Op
end

"""
    `Reduce(op, dims...)`

Produce an object `s` such that when `s` is broadcasted as a function over an
argument `arg`, the result is a lazy view of a reduction over the specified
dimensions, collapsing remaining dimensions downward. If no dimensions are
specified, all dimensions are reduced over.

See also: [`Swizzle`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Reduce(+, 2).(A)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
```
"""
@inline Reduce(op::Op, dims...) where {Op} = Reduce{nothing, Op, dims}(op)

"""
    `Reduce{T}(op, dims...)`

Similar to [`Reduce`](@ref), but the eltype of the resulting swizzle is
declared to be `T`.

See also: [`Reduce`](@ref).
"""
@inline Reduce{T}(op::Op, dims...) where {T, Op} = Reduce{T, Op, dims}(op)

@inline (rd::Reduce)(arg) = rd(BroadcastedArray(arg))
@inline function parse_reduce_mask(arr, dims::Tuple{Vararg{Int}}) where {M, N}
    c = 0
    return ntuple(d -> d in dims ? drop : c += 1, Val(ndims(arr)))
end
@inline function parse_reduce_mask(arr, dims::Tuple{}) where {M, N}
    return ntuple(d -> drop, Val(ndims(arr)))
end
@inline function(rd::Reduce{nothing, <:Any, dims})(arg::AbstractArray{<:Any, N}) where {dims, N}
    if @generated
        mask = parse_reduce_mask(arg, dims)
        return :(return SwizzledArray(arg, $(Val(mask)), rd.op))
    else
        mask = parse_reduce_mask(arg, dims)
        return SwizzledArray(arg, mask, rd.op)
    end
end
@inline function(rd::Reduce{T, <:Any, dims})(arg::AbstractArray{<:Any, N}) where {T, dims, N}
    if @generated
        mask = parse_reduce_mask(arg, dims)
        return :(return SwizzledArray{T}(arg, $(Val(mask)), rd.op))
    else
        mask = parse_reduce_mask(arg, dims)
        return SwizzledArray{T}(arg, mask, rd.op)
    end
end


struct Sum{T} end

"""
    `Sum(dims...)`

Produce an object `s` such that when `s` is broadcasted as a function over an
argument `arg`, the result is a lazy view of the sum over the specified
dimensions, collapsing remaining dimensions downward. If no dimensions are
specified, all dimensions are summed.

See also: [`Reduce`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Sum(2).(A)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
```
"""
function Sum(dims::Int...)
    Reduce(+, dims...)
end

"""
    `Sum{T}(dims...)`

Similar to [`Sum`](@ref), but the eltype of the resulting swizzle is
declared to be `T`.

See also: [`Sum`](@ref).
"""
function Sum{T}(dims::Int...) where {T}
    Reduce{T}(+, dims...)
end



struct Max{T} end

"""
    `Max(dims...)`

Produce an object `s` such that when `s` is broadcasted as a function over an
argument `arg`, the result is a lazy view of the maximum over the specified
dimensions, collapsing remaining dimensions downward. If no dimensions are
specified, all dimensions are reduced.

See also: [`Reduce`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Max(2).(A)
5×1 Array{Int64,2}:
 2
 4
 6
 8
 10
```
"""
function Max(dims::Int...)
    Reduce(max, dims...)
end

"""
    `Max{T}(dims...)`

Similar to [`Max`](@ref), but the eltype of the resulting swizzle is
declared to be `T`.

See also: [`Max`](@ref).
"""
function Max{T}(dims::Int...) where {T}
    Reduce{T}(max, dims...)
end



struct Min{T} end

"""
    `Min(dims...)`

Produce an object `s` such that when `s` is broadcasted as a function over an
argument `arg`, the result is a lazy view of the minimum over the specified
dimensions, collapsing remaining dimensions downward. If no dimensions are
specified, all dimensions are reduced.

See also: [`Reduce`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Min(2).(A)
5×1 Array{Int64,2}:
 1
 2
 5
 7
 9
```
"""
function Min(dims::Int...)
    Reduce(min, dims...)
end

"""
    `Min{T}(dims...)`

Similar to [`Min`](@ref), but the eltype of the resulting swizzle is
declared to be `T`.

See also: [`Min`](@ref).
"""
function Min{T}(dims::Int...) where {T}
    Reduce{T}(min, dims...)
end



function SwizzleTo(_imask, op)
    Swizzle(imasktuple(d->drop, identity, _imask), op) #FIXME computation should occur in type domain.
end

function BeamTo(dims::Union{Int, Drop}...)
    SwizzleTo(dims, nooperator)
end

function SumTo(dims::Union{Int, Drop}...)
    SwizzleTo(dims, +)
end

function MaxTo(dims::Union{Int, Drop}...)
    SwizzleTo(dims, max)
end

function MinTo(dims::Union{Int, Drop}...)
    SwizzleTo(dims, min)
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
