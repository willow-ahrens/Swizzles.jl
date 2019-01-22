module Swizzles

export Drop, drop
export getindexinto, setindexinto #maybe dont export...
export BroadcastedArray, WrapperArrayConstructor, wrap
export SwizzledArray
export swizzle, swizzle!
export Swizzle, Reduce, Sum, Max, Min, Beam
export SwizzleTo, ReduceTo, SumTo, MaxTo, MinTo, BeamTo
export Delay, Intercept

include("util.jl")

include("WrapperArrays.jl")
include("GeneratedArrays.jl")
include("BroadcastedArrays.jl")
include("ShallowArrays.jl")
include("ExtrudedArrays.jl")
include("MatchedArrays.jl")
include("SwizzledArrays.jl")

include("properties.jl")

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
function Beam(dims::Union{Int, Drop}...)
    Swizzle(dims, nooperator)
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
Reduce(op, dims::Int...) = _Reduce(op, Val(dims))
function _Reduce(op, ::Val{dims}) where {dims}
    if @generated
        m = maximum((0, dims...))
        s = Set(dims)
        c = 0
        mask = flatten((ntuple(d -> d in s ? drop : c += 1, m), countfrom(m - length(s) + 1)))
        return :(return Swizzle($(mask), op))
    else
        m = maximum((0, dims...))
        s = Set(dims)
        c = 0
        return Swizzle(flatten((ntuple(d -> d in s ? drop : c += 1, m), countfrom(m - length(s) + 1))), op)
    end
end
Reduce(op) = Swizzle(repeated(drop), op)

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

function SwizzleTo(imask, op)
    Swizzle(setindexinto(ntuple(d->drop, maximum((0, imask...))), 1:length(imask), imask), op)
end

function BeamTo(dims::Union{Int, Drop}...)
    SwizzleTo(dims, nooperator)
end

function ReduceTo(op, dims::Union{Int, Drop}...)
    SwizzleTo(dims, op)
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

end
