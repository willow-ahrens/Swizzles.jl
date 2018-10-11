#=
using .Base.Cartesian
using .Base: Indices, OneTo, tail, to_shape, isoperator, promote_typejoin,
             _msk_end, unsafe_bitgetindex, bitcache_chunks, bitcache_size, dumpbitcache, unalias
import .Base: copy, copyto!
export broadcast, broadcast!, BroadcastStyle, broadcast_axes, broadcastable, dotview, @__dot__
=#

### An operator which does not expect to be called.

"""
  `unspecifiedop(a, b)` is a reduction operator which does not expect to be called. It startles easily.
"""
function unspecifiedop(a, b) = throw(ArgumentError("unspecified reduction operator"))

### Lazy-wrapper for swizzling

# `Swizzled` wraps the argument to `swizzle(A, dims, op=unspecifiedop)`. A
# statement like
#    y = Swizzler((1,), +).(x .* (Swizzler((2, 1)).x .+ 1))
# will result in code that is essentially
#    y = copy(Swizzled(Broadcasted(*, Swizzled(x, (2, 1)), Broadcasted(+, x, 1)), (1,), +))
# `swizzle!` results in `copyto!(dest, Swizzled(...))`.
# `dims` is an iterator of nonnegative integers specifying where dimensions of
# the wrapped `A` will appear in the output array. Each dimension of `A` may be
# used at most once in `dims`, but `skip` is a special value that may be used
# to specify that a dimension is expected to be dropped. Uniqueness of integers
# in `dims` cannot be checked, since `dims` may be infinitely long. If `dims`
# is not long enough, remaining dimensions are skipped. `dims` is lifted into
# the type domain

struct Swizzled{Style<:Union{Nothing,BroadcastStyle}, Axes, mask, imask, Arg, dims, Op}
    arg::Arg
    op::Op
    axes::Axes
    #TODO enforce inv swizzle/other properties?
end

#= TODO do we need this?
BroadcastStyle(::Type{<:Swizzled{Style}}) where {Style} = Style()
BroadcastStyle(::Type{<:Swizzled{S}}) where {S<:Union{Nothing,Unknown}} =
    throw(ArgumentError("Swizzled{Unknown} wrappers do not have a style assigned"))

argtype(::Type{Swizzled{<:Any,<:Any,<:Any,<:Any,Arg}}) where {Arg} = Arg
argtype(sz::Swizzled) = argtype(typeof(sz))
=#

Swizzled(arg, dims, op=unspecifiedop, axes=nothing, mask=nothing, imask=nothing) =
    Swizzled{typeof(swizzle_style(arg, dims, op))}(arg, dims, op, axes, mask, imask)
Swizzled{Style}(arg, dims, op=unspecifiedop, axes=nothing, mask=nothing, imask=nothing) where {Style} =
    # using Core.Typeof rather than F preserves inferrability when f is a type, according to Matt I guess
    Swizzled{Style, typeof(axes), mask, imask, typeof(arg), dims, Core.Typeof(op)}(arg, op, axes)
end

Base.convert(::Type{Swizzled{NewStyle}}, sz::Swizzled{Style,Axes,mask,imask,Arg,dims,Op}) where {NewStyle,Style,Axes,mask,imask,Arg,dims,Op} =
    Swizzled{NewStyle,Axes,mask,imask,Arg,dims,Op}(sz.arg, sz.dims, sz.op, sz.axes)

function Base.show(io::IO, sz::Swizzled{Style}) where {Style}
    print(io, Swizzled)
    # Only show the style parameter if we have a set of axes — representing an instantiated
    # "outermost" Swizzled. The styles of nested Broadcasteds represent an intermediate
    # computation that is not relevant for dispatch, confusing, and just extra line noise.
    sz.axes isa Tuple && print(io, '{', Style, '}')
    print(io, '(', sz.arg, ", ", getdims(sz), ", ", sz.op ')')
    nothing
end

## Allocating the output container
Base.similar(sz::Swizzled{DefaultArrayStyle}, ::Type{ElType}) where {ElType} =
    similar(Array{ElType}, axes(sz))
Base.similar(sz::Swizzled{DefaultArrayStyle}, ::Type{Bool}) =
    similar(BitArray, axes(sz))
# In cases of conflict we fall back on Array
Base.similar(sz::Swizzled{ArrayConflict}, ::Type{ElType}) where ElType =
    similar(Array{ElType}, axes(sz))
Base.similar(sz::Swizzled{ArrayConflict}, ::Type{Bool}) =
    similar(BitArray, axes(sz))

## Computing the result's axes. Most types probably won't need to specialize this.
swizzle_dims(::Type{Swizzled{<:Any,<:Any,<:Any,<:Any,<:Any, dims}}) where {dims} = dims
swizzle_dims(sz::Swizzled) = swizzle_dims(typeof(sz))

swizzle_imask(sz::Swizzled) = swizzle_imask(typeof(sz))
swizzle_imask(::Type{Swizzled{<:Any,<:Any,<:Any,imask}}) where {imask} = imask
swizzle_imask(sz::Swizzled{<:Any,<:Any,<:Any,nothing}) = (take(swizzle_dims(sz), length(broadcast_axes(arg(sz))))...,)

swizzle_mask(sz::Swizzled) = swizzle_mask(typeof(sz))
swizzle_mask(::Type{Swizzled{<:Any,<:Any,mask}}) where {mask} = mask
function swizzle_mask(sz::Type{Swizzled{<:Any,<:Any,nothing}})
  imask = swizzle_imask(sz)
  return setindexinto((repeated(skip, max(0, maximum(imask)))...,), 1:length(imask), imask)
end

@inline Base.axes(sz::Swizzled) = _axes(sz, sz.axes)
_axes(::Swizzled, axes::Tuple) = axes
@inline function _axes(sz::Swizzled, ::Nothing)
    b_a = broadcast_axes(sz.arg)
    mask = swizzle_mask(sz)
    return getindexinto((repeated(Base.OneTo(1), length(mask))...,), b_a, mask)
end

@inline Base.eachindex(sz::Swizzled) = _eachindex(axes(sz))
_eachindex(t::Tuple{Any}) = t[1]
_eachindex(t::Tuple) = CartesianIndices(t)

Base.ndims(::Swizzled{<:Any,<:NTuple{N,Any}}) where {N} = N
Base.ndims(::Type{<:Swizzled{<:Any,<:NTuple{N,Any}}}) where {N} = N
#TODO could also compute from mask or imask

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

Base.IteratorSize(::Type{<:Swizzled{<:Any,<:NTuple{N,Base.OneTo}}}) where {N} = Base.HasShape{N}()
Base.IteratorEltype(::Type{<:Swizzled}) = Base.EltypeUnknown()

## Instantiation fills in the "missing" fields in Swizzled.

"""
    Broadcast.instantiate(sz::Swizzled)
Construct and check the imask, mask, and axes and for the lazy Swizzle object `sz`.
Custom [`BroadcastStyle`](@ref)s may override this default in cases where it is fast and easy
to compute and verify the resulting `axes` on-demand, leaving the `axis` field
of the `Swizzled` object empty (populated with [`nothing`](@ref)).
"""
@inline function instantiate(sz::Swizzled{Style}) where {Style}
    if sz.axes isa Nothing || sz.mask isa Nothing || sz.imask isa Nothing
        imask = swizzle_imask(sz)
        mask = swizzle_mask(sz)
        axes = axes(sz)
        return Swizzled{Style}(sz.arg, sz.dims, sz.op, axes, mask, imask)
    else
        check_swizzle(sz)
        return sz
    end
end

function check_swizzle(sz)
    imask = swizzle_imask(sz)
    mask = swizzle_mask(sz)
    argaxes = axes(sz.arg)
    axes = axes(sz)
    @assert imask == (take(swizzle_dims(sz), length(imask))...,)
    @assert length(imask) == length(argaxes)
    @assert eltype(imask) isa Union{Int, Skip}
    @assert eltype(mask) isa Union{Int, Skip}
    @assert (minimum(imask) isa Int && minimum(imask) >= 1) || minimum(imask) isa Skip
    @assert (maximum(imask) isa Int && maximum(imask) == length(mask)) || (maximum(imask) isa Skip && length(mask) == 0)
    @assert all(map(((i, j),) -> m isa Skip || mask[m] == i, enumerate(imask)))
    @assert all(map(((i, j),) -> m isa Skip || imask[m] == i, enumerate(mask)))
    @assert all(map(((i, j),) -> (j isa Skip && axes[i] == 1:1) || axes[i] == argaxes[j], enumerate(mask)))
end

## Flattening swizzles is difficult and people shouldn't flatten broadcasts anyway. Harumph

### Objects with customized swizzling behavior should define a corresponding BroadcastStyle

"""
  `swizzle_style(style, dims, op=unspecifiedop)`
Broadcast styles are used to determine behavior of objects under swizzling.  To
customize the swizzling behavior of a type, one can first define an appropriate
Broadcast style for the the type, then declare how the broadcast style should
behave under broadcasting after the swizzle by overriding the
`swizzle_style` method.
"""
swizzle_style

swizzle_style(::Broadcast.Style{Tuple}, dims, op) = first(dims) == 1 ? Broadcast.Style{Tuple}() : Broadcast.DefaultArrayStyle(Val(first(dims)))
swizzle_style(style::A, dims, op) where {N, A <: Broadcast.AbstractArrayStyle{N}} = A(Val(maximum(take(idims, N))))
swizzle_style(style::Broadcast.AbstractArrayStyle{Any}, dims, op) = style
swizzle_style(::BroadcastStyle, dims, op) = Broadcast.Unknown()
swizzle_style(::ArrayConflict, dims, op) = Broadcast.ArrayConflict() #FIXME

@inline function Base.getindex(sz::Swizzled, I::Union{Integer,CartesianIndex})
    @boundscheck checkbounds(sz, I)
    inds = eachindex(getindexinto(axes(sz.arg), I, swizzle_imask(sz)))
    (i, inds) = peel(inds)
    res = @inbounds getindex(sz.arg, i)
    for i in inds
        res = sz.op(res, @inbounds getindex(sz.arg, i))
    end
    return res
end
Base.@propagate_inbounds Base.getindex(sz::Swizzled, i1::Integer, i2::Integer, I::Integer...) = sz[CartesianIndex((i1, i2, I...))]
Base.@propagate_inbounds Base.getindex(sz::Swizzled) = sz[CartesianIndex(())]

#=
@inline Base.checkbounds(bc::Broadcasted, I::Union{Integer,CartesianIndex}) =
    Base.checkbounds_indices(Bool, axes(bc), (I,)) || Base.throw_boundserror(bc, (I,))
=# #TODO why would anyone need this?

broadcastable(sz::Swizzle) = sz

## Swizzling core

"""
    swizzle(A, dims, op=unspecifiedop)
Create a new obect `B` such that the dimension `i` of `A` is mapped to
dimension `dims[i]` of `B`, operating on lazy broadcast expressions, arrays,
tuples, collections, [`Ref`](@ref)s and/or scalars `As`. If `dims[i]` is an
instance of the singleton type `Skip`, the dimension is reduced over using
`op`. `dims` may be any (possibly infinite) iterable over elements of type
`Int` and `Skip`. The integers in `dims` must be unique, and if `dims` is
not long enough, additional `skip` are added to the end.
The resulting container type is established by the following rules:
 - If all elements of `dims` are `Skip`, it returns an unwrapped scalar.
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
swizzle(A, dims, op=unspecifiedop) = materialize(Swizzled(A, dims, op))

"""
    swizzle!(A, dims, op=unspecifiedop)
Like [`swizzle`](@ref), but store the result of
`swizzle(A, dims, op)` in the `dest` array.
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
swizzle!(dest, A, dims, op=unspecifiedop) = (materialize!(dest, Swizzled(A, dims, op); dest)

@inline materialize(sz::Swizzled) = copy(sz)

@inline materialize!(dest, sz::Swizzled) = copyto!(dest, sz)



copy(sz::Swizzled{Style, Axes, mask, imask}) where {Style, Axes, mask, imask} = copy(Swizzled{Nothing, Axes, mask, imask}(sz.arg, swizzle_dims(sz), sz.op, sz.axes, mask, imask))

copyto!(dest, sz::Swizzled{Style, Axes, mask, imask}) where {Style, Axes, mask, imask} = copyto!(dest, Swizzled{Nothing, Axes, mask, imask}(sz.arg, swizzle_dims(sz), sz.op, sz.axes, mask, imask))

copy(sz::Swizzled{Nothing}) = copy(instantiate(preprocess(Broadcasted(identity, (sz,)))))

copyto!(dest, sz::Swizzled{Nothing}) = copyto(dest, instantiate(preprocess(Broadcasted(identity, (sz,)))))

preprocess(dest, sz::Swizzled{Style, Axes, mask, imask}) where {Style, Axes, mask, imask} = extrude(instantiate(Swizzled{Style, Axes, mask, imask}(preprocess(dest, sz.arg), swizzle_dims(sz), sz.op, sz.axes, mask, imask)))

struct Swizzler
    dims
    op
end

broadcasted(style, sz::Swizzler, arg) = Swizzled{style}(sz.dims, sz.op)



function Sum(dims::Int...)
    Reduce(dims, +)
end

function Reduce(dims, op)
    m = maximum((0, dims...))
    s = set(dims)
    c = 1
    Swizzler(flattened((ntuple(d->d in s ? Skip : c++, m), count(m - length(s) + 1))), op)
end



function Permute(perm::Int...)
    @assert isperm(perm)
    Swizzler(flattened(invperm(perm), count(length(perm) + 1)), unspecifiedop)
end
