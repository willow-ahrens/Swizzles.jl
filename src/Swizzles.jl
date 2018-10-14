module Swizzles

export Drop, drop
export getindexinto, setindexinto

include("base.jl")
include("util.jl")

using Base: checkbounds_indices, throw_boundserror, tail
using Base.Iterators: repeated, countfrom, flatten, product, take, peel
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, broadcast_axes, instantiate, broadcastable, longest_tuple

export swizzle, swizzle!
export Swizzle, Reduce, Sum, Max, Min, Beam
export SwizzleTo, ReduceTo, SumTo, MaxTo, MinTo, BeamTo

### An operator which does not expect to be called.

"""
  `nooperator(a, b)` is an operator which does not expect to be called. It startles easily.
"""
nooperator(a, b) = throw(ArgumentError("unspecified operator"))

struct Swizzled{Arg, mask, imask, Op}
    arg::Arg
    op::Op
    function Swizzled{Arg, mask, imask, Op}(arg::Arg, op::Op) where {Arg, mask, imask, Op}
        #FIXME check swizzles.
        new(arg, op)
    end
end

function Swizzled(arg, mask, op)
    arg = instantiate(broadcastable(arg))
    mask = (take(flatten((mask, repeated(drop))), ndims(typeof(arg)))...,)
    return _Swizzled(arg, mask, op)
end
function Swizzled(arg::Tuple, mask, op)
    mask = (take(flatten((mask, repeated(drop))), 1)...,)
    return _Swizzled(arg, mask, op)
end
function _Swizzled(arg, mask::Tuple{}, op)
    imask = ()
    Swizzled{typeof(arg), mask, imask, Core.Typeof(op)}(arg, op)
end
function _Swizzled(arg, mask::Tuple, op)
    imask = setindexinto(ntuple(d->drop, max(0, maximum(mask))), 1:length(mask), mask)
    Swizzled{typeof(arg), mask, imask, Core.Typeof(op)}(arg, op)
end

mask(::Type{Swizzled{Arg, _mask, _imask, Op}}) where {Arg, _mask, _imask, Op} = _mask
mask(sz::S) where {S <: Swizzled} = mask(S)
imask(::Type{Swizzled{Arg, _mask, _imask, Op}}) where {Arg, _mask, _imask, Op} = _imask
imask(sz::S) where {S <: Swizzled} = imask(S)

function Base.show(io::IO, sz::Swizzled)
    print(io, Swizzled)
    print(io, '(', sz.arg, ", ", mask(sz), ", ", sz.op, ')')
    nothing
end

@inline Base.axes(sz::Swizzled) = getindexinto(ntuple(d->Base.OneTo(1), length(imask(sz))), broadcast_axes(sz.arg), imask(sz))

@inline Base.eachindex(sz::Swizzled) = _eachindex(axes(sz))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices(t)

Base.ndims(::Type{S}) where {S <: Swizzled} = length(imask(S))

Base.length(sz::Swizzled) = prod(map(length, axes(sz)))

function Base.iterate(sz::Swizzled)
    iter = eachindex(sz)
    iterate(sz, (iter,))
end
Base.@propagate_inbounds function Base.iterate(sz::Swizzled, s)
    y = iterate(s...)
    y === nothing && return nothing
    i, newstate = y
    return (sz[i], (s[1], newstate))
end

Base.IteratorSize(::Type{<:Swizzled{<:Any,<:NTuple{N}}}) where {N} = Base.HasShape{N}()
Base.IteratorEltype(::Type{<:Swizzled}) = Base.EltypeUnknown()

@inline function _getindex(sz::Swizzled, I::Tuple{Vararg{Int}})
    @boundscheck checkbounds_indices(Bool, axes(sz), I) || throw_boundserror(sz, I)
    inds = product(getindexinto(broadcast_axes(sz.arg), I, mask(sz))...)
    (i, inds) = peel(inds)
    res = @inbounds getindex(sz.arg, i...)
    for i in inds
        res = sz.op(res, @inbounds getindex(sz.arg, i...))
    end
    return res
end
@inline Base.getindex(sz::Swizzled, I::Int) = _getindex(sz, I)
@inline Base.getindex(sz::Swizzled, I::CartesianIndex) = _getindex(sz, Tuple(I))
@inline Base.getindex(sz::Swizzled, I::Int...) = _getindex(sz, I)
@inline Base.getindex(sz::Swizzled) = _getindex(sz, ())

"""
    swizzle(A, mask, op=nooperator)
Create a new obect `B` such that the dimension `i` of `A` is mapped to
dimension `mask[i]` of `B`, operating on lazy broadcast expressions, arrays,
tuples, collections, [`Ref`](@ref)s and/or scalars `As`. If `mask[i]` is an
instance of the singleton type `Drop`, the dimension is reduced over using
`op`. `mask` may be any (possibly infinite) iterable over elements of type
`Int` and `Drop`. The integers in `mask` must be unique, and if `mask` is
not long enough, additional `Drop`s are added to the end.
The resulting container type is established by the following rules:
 - If all elements of `mask` are `Drop`, it returns an unwrapped scalar.
 - All other combinations of arguments default to returning an `Array`, but
   custom container types can define their own implementation rules to
   customize the result when they appear as an argument.
The swizzle operation is represented with a special lazy `Swizzled` type.
`swizzle` results in `copy(Swizzled(...))`.
The swizzle operation can use a special `Swizzler` type to take advantage of
the special broadcast syntax. A statement like:
   y = Swizzler((1,), +).(x .* (Swizzler((2, 1)).x .+ 1))
will result in code that is essentially:
   y = copy(Swizzled(Broadcasted(*, Swizzled(x, (2, 1)), Broadcasted(+, x, 1)), (1,), +))
If Swizzleds are mixed with Broadcasteds, the result is fused into one big for loop.
# Examples
```jldoctest
julia> A = [1, 2, 3, 4, 5]
5-element Array{Int64,1}:
 1
 2
 3
 4
 5
julia> B = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> swizzle(B, (1,), +)
5×2 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> swizzle(B, (), +)
55
julia> Swizzler((2,)).(parse.(Int, ["1", "2"]))
1x2-element Array{Int64,1}:
 1 2
"""
swizzle(A, mask, op=nooperator) = copy(Swizzled(A, mask, op))

"""
    swizzle!(A, mask, op=nooperator)
Like [`swizzle`](@ref), but store the result of
`swizzle(A, mask, op)` in the `dest` array.
Note that `dest` is only used to store the result
`swizzle!` results in `copyto!(dest, Swizzled(...))`.
# Examples
```jldoctest
julia> A = [1.0; 0.0]; B = [0.0; 0.0];
julia> broadcast!(+, B, A, (0, -2.0));
julia> B
2-element Array{Float64,1}:
  1.0
 -2.0
julia> A
2-element Array{Float64,1}:
 1.0
 0.0
julia> broadcast!(+, A, A, (0, -2.0));
julia> A
2-element Array{Float64,1}:
  1.0
 -2.0
```
"""
swizzle!(dest, A, mask, op=nooperator) = copyto!(dest, Swizzled(A, mask, op))

Base.Broadcast.broadcastable(sz::Swizzled) = sz

@inline Base.Broadcast.materialize(sz::Swizzled) = copy(sz)

@inline Base.Broadcast.materialize!(dest, sz::Swizzled) = copyto!(dest, sz)

Base.copy(sz::Swizzled) = copy(instantiate(Broadcasted(identity, (sz,))))

Base.copyto!(dest, sz::Swizzled) = copyto!(dest, instantiate(Broadcasted(identity, (sz,))))

Base.copyto!(dest::AbstractArray, sz::Swizzled) = copyto!(dest, instantiate(Broadcasted(identity, (sz,))))

#Base.Broadcast.preprocess(dest, sz::Swizzled{Style}) where {Style} = instantiate(Swizzled{Style}(sz.arg, mask(sz), sz.op, sz.axes, imask(sz))) #TODO problem here too.
#Base.Broadcast.broadcasted(style::BroadcastStyle, szr::Swizzler, arg) = Swizzled(arg, szr.mask, szr.op) #Should use style here duh.

"""
  `SwizzleStyle(style, mask, op=nooperator)`
Broadcast styles are used to determine behavior of objects under swizzling.  To
customize the swizzling behavior of a type, one can first define an appropriate
Broadcast style for the the type, then declare how the broadcast style should
behave under broadcasting after the swizzle by overriding the
`SwizzleStyle` method.
"""
SwizzleStyle

SwizzleStyle(::Style{Tuple}, mask, op) = first(mask) == 1 ? Style{Tuple}() : DefaultArrayStyle(Val(max(0, first(mask))))
Broadcast.longest_tuple(::Nothing, t::Tuple{<:Swizzled{<:Any, (1,)},Vararg{Any}}) = longest_tuple(longest_tuple(nothing, (t[1].arg,)), tail(t))
SwizzleStyle(style::A, mask, op) where {A <: AbstractArrayStyle{0}} = A(Val(0))
SwizzleStyle(style::A, mask, op) where {N, A <: AbstractArrayStyle{N}} = A(Val(max(0, maximum(take(mask, N)))))
SwizzleStyle(style::AbstractArrayStyle{Any}, mask, op) = style
SwizzleStyle(::BroadcastStyle, mask, op) = Unknown()
SwizzleStyle(::ArrayConflict, mask, op) = ArrayConflict() #FIXME

Broadcast.BroadcastStyle(::Type{<:Swizzled{Arg, mask, imask, Op}}) where {Arg, mask, imask, Op} = SwizzleStyle(BroadcastStyle(Arg), mask, Op)


struct Swizzle
    mask
    op
end

Base.Broadcast.broadcasted(style::BroadcastStyle, sz::Swizzle, arg) = Swizzled(arg, sz.mask, sz.op)

function Reduce(op, dims::Int...)
    m = maximum((0, dims...))
    s = Set(dims)
    c = 0
    Swizzle(flatten((ntuple(d -> d in s ? drop : c += 1, m), countfrom(m - length(s) + 1))), op)
end

function Sum(dims::Int...)
    Reduce(+, dims...)
end

function Max(dims::Int...)
    Reduce(max, dims...)
end

function Min(dims::Int...)
    Reduce(min, dims...)
end

function Beam(dims::Union{Int, Drop}...)
    Swizzle(dims, nooperator)
end

function SwizzleTo(imask::Tuple{Vararg{<:Union{Int, Drop}}}, op)
    Swizzle(setindexinto(ntuple(d->drop, maximum((0, imask...))), 1:length(imask), imask), op)
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

function BeamTo(dims::Union{Int, Drop}...)
    SwizzleTo(dims, nooperator)
end

end
