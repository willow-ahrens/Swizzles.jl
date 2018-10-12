module Swizzles

export Out, out

include("base.jl")
include("util.jl")

export swizzle, swizzle!
export Swizzle, Beam, Reduce, Sum

### An operator which does not expect to be called.

"""
  `nooperator(a, b)` is a reduction operator which does not expect to be called. It startles easily.
"""
nooperator(a, b) = throw(ArgumentError("unspecified operator"))

struct Swizzled{Arg, mask, imask, Op}
    arg::Arg
    op::Op
    function Swizzled{Arg, mask, imask, Op}(arg::Arg, op::Op) where {Arg, mask, imask, Op}
        #TODO assert a bunch.
        new(arg, op)
    end
end

function Swizzled(arg, mask, op)
    arg = broadcastable(arg)
    mask = (take(mask, ndims(typeof(arg)))...,)
    imask = setindexinto((repeated(pass, max(0, maximum(mask)))...,), 1:length(mask), mask)
    Swizzled{typeof(arg), mask, imask, Core.Typeof(op)}(arg, op)
end

mask(::Type{Swizzled{Arg, _mask, _imask, Op}}) where {Arg, _mask, _imask, Op} = _mask
mask(sz::S) where {S <: Swizzled} = mask(S)
imask(::Type{Swizzled{Arg, _mask, _imask, Op}}) where {Arg, _mask, _imask, Op} = _imask
mask(sz::S) where {S <: Swizzled} = imask(S)

function Base.show(io::IO, sz::Swizzled)
    print(io, Swizzled)
    print(io, '(', sz.arg, ", ", mask(sz), ", ", sz.op, ')')
    nothing
end

@inline Base.axes(sz::Swizzled) = getindexinto((repeated(Base.OneTo(1), length(imask))...,), broadcast_axes(sz.arg), imask)

@inline Base.eachindex(sz::Swizzled) = _eachindex(axes(sz))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices(t)

Base.ndims(::Type{S}) where {S <: Swizzled} = length(mask(S))

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
    @boundscheck checkbounds_indices(axes(sz), I)
    inds = eachindex(getindexinto(axes(sz.arg), I, _mask(sz)))
    (i, inds) = peel(inds)
    res = @inbounds getindex(sz.arg, i)
    for i in inds
        res = sz.op(res, @inbounds getindex(sz.arg, i))
    end
    return res
end
@inline Base.getindex(sz::Swizzled, I::Int) = _getindex(I)
@inline Base.getindex(sz::Swizzled, I::CartesianIndex) = _getindex(Tuple(I))
@inline Base.getindex(sz::Swizzled, I::Int...) = _getindex(I)
@inline Base.getindex(sz::Swizzled) = _getindex(())



swizzle(A, mask, op=unspecifiedop) = copy(Swizzled(A, mask, op))

swizzle!(dest, A, mask, op=unspecifiedop) = copyto!(dest, Swizzled(A, mask, op))



Base.Broadcast.broadcastable(sz::Swizzled) = sz

@inline Base.Broadcast.materialize(sz::Swizzled) = copy(sz)

@inline Base.Broadcast.materialize!(dest, sz::Swizzled) = copyto!(dest, sz)

Base.copy(sz::Swizzled) = copy(instantiate(Broadcasted(identity, (sz,))))

Base.copyto!(dest, sz::Swizzled) = copyto!(dest, instantiate(Broadcasted(identity, (sz,))))

Base.Broadcast.preprocess(dest, sz::Swizzled{Style}) where {Style} = instantiate(Swizzled{Style}(sz.arg, _mask(sz), sz.op, sz.axes, _imask(sz))) #TODO problem here too.
Base.Broadcast.broadcasted(style::BroadcastStyle, szr::Swizzler, arg) = Swizzled(arg, szr.mask, szr.op) #Should use style here duh.

"""
  `swizzle_style(style, mask, op=unspecifiedop)`
Broadcast styles are used to determine behavior of objects under swizzling.  To
customize the swizzling behavior of a type, one can first define an appropriate
Broadcast style for the the type, then declare how the broadcast style should
behave under broadcasting after the swizzle by overriding the
`swizzle_style` method.
"""
swizzle_style

swizzle_style(::Style{Tuple}, mask, op) = first(mask) == 1 ? Style{Tuple}() : DefaultArrayStyle(Val(first(mask)))
swizzle_style(style::A, mask, op) where {N, A <: Broadcast.AbstractArrayStyle{N}} = A(Val(maximum(take(mask, N))))
swizzle_style(style::Broadcast.AbstractArrayStyle{Any}, mask, op) = style
swizzle_style(::BroadcastStyle, mask, op) = Broadcast.Unknown()
swizzle_style(::ArrayConflict, mask, op) = Broadcast.ArrayConflict() #FIXME

Broadcast.BroadcastStyle(::Type{<:Swizzled{Nothing, Axes, imask, Arg, mask, Op}}) where {Style, Axes, imask, Arg, mask, Op} = swizzle_style(BroadcastStyle(Arg), mask, Op)
Broadcast.BroadcastStyle(::Type{<:Swizzled{Style}}) where {Style} = Style()


struct Swizzle
    mask
    op
    Swizzle(mask, op=nooperator) = new(mask, op)
end


function Beam(dims...)
    Swizzler(arg, flatten((dims, repeated(pass))), unspecifiedop)
end

function NoSum(dims::Int...)
    m = maximum((0, dims...))
    Swizzler(setindexinto((take(repeated(pass), m)...,), 1:length(dims), dims), +)
end

function Sum(dims::Int...)
    Reduce(dims, +)
end

function Reduce(dims, op)
    m = maximum((0, dims...))
    s = Set(dims)
    c = 1
    Swizzler(flatten((ntuple(d -> d in s ? pass : c += 1, m), countfrom(m - length(s) + 1))), op)
end

function Permute(perm::Int...)
    @assert isperm(perm)
    Swizzler(flattened(invperm(perm), count(length(perm) + 1)), unspecifiedop)
end
