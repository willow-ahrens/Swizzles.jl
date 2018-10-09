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

@inline materialize(sz::Swizzled) = copy(instantiate(sz))

@inline function materialize!(dest, x)
    return copyto!(dest, instantiate(Broadcasted(identity, (x,), axes(dest))))
end

## general `copy` methods
@inline copy(bc::Broadcasted{<:AbstractArrayStyle{0}}) = bc[CartesianIndex()]
copy(bc::Broadcasted{<:Union{Nothing,Unknown}}) =
    throw(ArgumentError("broadcasting requires an assigned BroadcastStyle"))

const NonleafHandlingStyles = Union{DefaultArrayStyle,ArrayConflict}

@inline function copy(bc::Broadcasted{Style}) where {Style}
    ElType = combine_eltypes(bc.f, bc.args)
    if Base.isconcretetype(ElType)
        # We can trust it and defer to the simpler `copyto!`
        return copyto!(similar(bc, ElType), bc)
    end
    # When ElType is not concrete, use narrowing. Use the first output
    # value to determine the starting output eltype; copyto_nonleaf!
    # will widen `dest` as needed to accommodate later values.
    bc′ = preprocess(nothing, bc)
    iter = eachindex(bc′)
    y = iterate(iter)
    if y === nothing
        # if empty, take the ElType at face value
        return similar(bc′, ElType)
    end
    # Initialize using the first value
    I, state = y
    @inbounds val = bc′[I]
    dest = similar(bc′, typeof(val))
    @inbounds dest[I] = val
    # Now handle the remaining values
    return copyto_nonleaf!(dest, bc′, iter, state, 1)
end

## general `copyto!` methods
# The most general method falls back to a method that replaces Style->Nothing
# This permits specialization on typeof(dest) without introducing ambiguities
@inline copyto!(dest::AbstractArray, bc::Broadcasted) = copyto!(dest, convert(Broadcasted{Nothing}, bc))

# Performance optimization for the common identity scalar case: dest .= val
@inline function copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractArrayStyle{0}})
    # Typically, we must independently execute bc for every storage location in `dest`, but:
    # IF we're in the common no-op identity case with no nested args (like `dest .= val`),
    if bc.f === identity && bc.args isa Tuple{Any} && isflat(bc)
        # THEN we can just extract the argument and `fill!` the destination with it
        return fill!(dest, bc.args[1][])
    else
        # Otherwise, fall back to the default implementation like above
        return copyto!(dest, convert(Broadcasted{Nothing}, bc))
    end
end

# For broadcasted assignments like `broadcast!(f, A, ..., A, ...)`, where `A`
# appears on both the LHS and the RHS of the `.=`, then we know we're only
# going to make one pass through the array, and even though `A` is aliasing
# against itself, the mutations won't affect the result as the indices on the
# LHS and RHS will always match. This is not true in general, but with the `.op=`
# syntax it's fairly common for an argument to be `===` a source.
broadcast_unalias(dest, src) = dest === src ? src : unalias(dest, src)
broadcast_unalias(::Nothing, src) = src

# Preprocessing a `Broadcasted` does two things:
# * unaliases any arguments from `dest`
# * "extrudes" the arguments where it is advantageous to pre-compute the broadcasted indices
@inline preprocess(dest, bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, preprocess_args(dest, bc.args), bc.axes)
preprocess(dest, x) = extrude(broadcast_unalias(dest, x))

@inline preprocess_args(dest, args::Tuple) = (preprocess(dest, args[1]), preprocess_args(dest, tail(args))...)
preprocess_args(dest, args::Tuple{Any}) = (preprocess(dest, args[1]),)
preprocess_args(dest, args::Tuple{}) = ()

# Specialize this method if all you want to do is specialize on typeof(dest)
@inline function copyto!(dest::AbstractArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = preprocess(dest, bc)
    @simd for I in eachindex(bc′)
        @inbounds dest[I] = bc′[I]
    end
    return dest
end

# Performance optimization: for BitArray outputs, we cache the result
# in a "small" Vector{Bool}, and then copy in chunks into the output
@inline function copyto!(dest::BitArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    ischunkedbroadcast(dest, bc) && return chunkedcopyto!(dest, bc)
    tmp = Vector{Bool}(undef, bitcache_size)
    destc = dest.chunks
    ind = cind = 1
    bc′ = preprocess(dest, bc)
    @simd for I in eachindex(bc′)
        @inbounds tmp[ind] = bc′[I]
        ind += 1
        if ind > bitcache_size
            dumpbitcache(destc, cind, tmp)
            cind += bitcache_chunks
            ind = 1
        end
    end
    if ind > 1
        @inbounds tmp[ind:bitcache_size] .= false
        dumpbitcache(destc, cind, tmp)
    end
    return dest
end

# For some BitArray operations, we can work at the level of chunks. The trivial
# implementation just walks over the UInt64 chunks in a linear fashion.
# This requires three things:
#   1. The function must be known to work at the level of chunks (or can be converted to do so)
#   2. The only arrays involved must be BitArrays or scalar Bools
#   3. There must not be any broadcasting beyond scalar — all array sizes must match
# We could eventually allow for all broadcasting and other array types, but that
# requires very careful consideration of all the edge effects.
const ChunkableOp = Union{typeof(&), typeof(|), typeof(xor), typeof(~), typeof(identity),
    typeof(!), typeof(*), typeof(==)} # these are convertible to chunkable ops by liftfuncs
const BroadcastedChunkableOp{Style<:Union{Nothing,BroadcastStyle}, Axes, F<:ChunkableOp, Args<:Tuple} = Broadcasted{Style,Axes,F,Args}
ischunkedbroadcast(R, bc::BroadcastedChunkableOp) = ischunkedbroadcast(R, bc.args)
ischunkedbroadcast(R, args) = false
ischunkedbroadcast(R, args::Tuple{<:BitArray,Vararg{Any}}) = size(R) == size(args[1]) && ischunkedbroadcast(R, tail(args))
ischunkedbroadcast(R, args::Tuple{<:Bool,Vararg{Any}}) = ischunkedbroadcast(R, tail(args))
ischunkedbroadcast(R, args::Tuple{<:BroadcastedChunkableOp,Vararg{Any}}) = ischunkedbroadcast(R, args[1]) && ischunkedbroadcast(R, tail(args))
ischunkedbroadcast(R, args::Tuple{}) = true

# Convert compatible functions to chunkable ones. They must also be green-lighted as ChunkableOps
liftfuncs(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, map(liftfuncs, bc.args), bc.axes)
liftfuncs(bc::Broadcasted{Style,<:Any,typeof(sign)}) where {Style} = Broadcasted{Style}(identity, map(liftfuncs, bc.args), bc.axes)
liftfuncs(bc::Broadcasted{Style,<:Any,typeof(!)}) where {Style} = Broadcasted{Style}(~, map(liftfuncs, bc.args), bc.axes)
liftfuncs(bc::Broadcasted{Style,<:Any,typeof(*)}) where {Style} = Broadcasted{Style}(&, map(liftfuncs, bc.args), bc.axes)
liftfuncs(bc::Broadcasted{Style,<:Any,typeof(==)}) where {Style} = Broadcasted{Style}((~)∘(xor), map(liftfuncs, bc.args), bc.axes)
liftfuncs(x) = x

liftchunks(::Tuple{}) = ()
liftchunks(args::Tuple{<:BitArray,Vararg{Any}}) = (args[1].chunks, liftchunks(tail(args))...)
# Transform scalars to repeated scalars the size of a chunk
liftchunks(args::Tuple{<:Bool,Vararg{Any}}) = (ifelse(args[1], typemax(UInt64), UInt64(0)), liftchunks(tail(args))...)
ithchunk(i) = ()
Base.@propagate_inbounds ithchunk(i, c::Vector{UInt64}, args...) = (c[i], ithchunk(i, args...)...)
Base.@propagate_inbounds ithchunk(i, b::UInt64, args...) = (b, ithchunk(i, args...)...)
@inline function chunkedcopyto!(dest::BitArray, bc::Broadcasted)
    isempty(dest) && return dest
    f = flatten(liftfuncs(bc))
    args = liftchunks(f.args)
    dc = dest.chunks
    @simd for i in eachindex(dc)
        @inbounds dc[i] = f.f(ithchunk(i, args...)...)
    end
    @inbounds dc[end] &= Base._msk_end(dest)
    return dest
end


@noinline throwdm(axdest, axsrc) =
    throw(DimensionMismatch("destination axes $axdest are not compatible with source axes $axsrc"))

function copyto_nonleaf!(dest, bc::Broadcasted, iter, state, count)
    T = eltype(dest)
    while true
        y = iterate(iter, state)
        y === nothing && break
        I, state = y
        @inbounds val = bc[I]
        S = typeof(val)
        if S <: T
            @inbounds dest[I] = val
        else
            # This element type doesn't fit in dest. Allocate a new dest with wider eltype,
            # copy over old values, and continue
            newdest = Base.similar(dest, promote_typejoin(T, S))
            for II in Iterators.take(iter, count)
                newdest[II] = dest[II]
            end
            newdest[I] = val
            return copyto_nonleaf!(newdest, bc, iter, state, count+1)
        end
        count += 1
    end
    return dest
end

## Tuple methods

@inline copy(bc::Broadcasted{Style{Tuple}}) =
    tuplebroadcast(longest_tuple(nothing, bc.args), bc)
@inline tuplebroadcast(::NTuple{N,Any}, bc) where {N} = ntuple(k -> @inbounds(_broadcast_getindex(bc, k)), Val(N))
# This is a little tricky: find the longest tuple (first arg) within the list of arguments (second arg)
# Start with nothing as a placeholder and go until we find the first tuple in the argument list
longest_tuple(::Nothing, t::Tuple{Tuple,Vararg{Any}}) = longest_tuple(t[1], tail(t))
# Or recurse through nested broadcast expressions
longest_tuple(::Nothing, t::Tuple{Broadcasted,Vararg{Any}}) = longest_tuple(longest_tuple(nothing, t[1].args), tail(t))
longest_tuple(::Nothing, t::Tuple) = longest_tuple(nothing, tail(t))
# And then compare it against all other tuples we find in the argument list or nested broadcasts
longest_tuple(l::Tuple, t::Tuple{Tuple,Vararg{Any}}) = longest_tuple(_longest_tuple(l, t[1]), tail(t))
longest_tuple(l::Tuple, t::Tuple) = longest_tuple(l, tail(t))
longest_tuple(l::Tuple, ::Tuple{}) = l
longest_tuple(l::Tuple, t::Tuple{Broadcasted}) = longest_tuple(l, t[1].args)
longest_tuple(l::Tuple, t::Tuple{Broadcasted,Vararg{Any}}) = longest_tuple(longest_tuple(l, t[1].args), tail(t))
# Support only 1-tuples and N-tuples where there are no conflicts in N
_longest_tuple(A::Tuple{Any}, B::Tuple{Any}) = A
_longest_tuple(A::Tuple{Any}, B::NTuple{N,Any}) where N = B
_longest_tuple(A::NTuple{N,Any}, B::Tuple{Any}) where N = A
_longest_tuple(A::NTuple{N,Any}, B::NTuple{N,Any}) where N = A
@noinline _longest_tuple(A, B) =
    throw(DimensionMismatch("tuples $A and $B could not be broadcast to a common size"))

## scalar-range broadcast operations ##
# DefaultArrayStyle and \ are not available at the time of range.jl
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r::OrdinalRange) = r
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r::StepRangeLen) = r
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r::LinRange) = r

broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r::OrdinalRange) = range(-first(r), step=-step(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r::StepRangeLen) = StepRangeLen(-r.ref, -r.step, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r::LinRange) = LinRange(-r.start, -r.stop, length(r))

broadcasted(::DefaultArrayStyle{1}, ::typeof(+), x::Real, r::AbstractUnitRange) = range(x + first(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r::AbstractUnitRange, x::Real) = range(first(r) + x, length=length(r))
# For #18336 we need to prevent promotion of the step type:
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r::AbstractRange, x::Number) = range(first(r) + x, step=step(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), x::Number, r::AbstractRange) = range(x + first(r), step=step(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r::StepRangeLen{T}, x::Number) where T =
    StepRangeLen{typeof(T(r.ref)+x)}(r.ref + x, r.step, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), x::Number, r::StepRangeLen{T}) where T =
    StepRangeLen{typeof(x+T(r.ref))}(x + r.ref, r.step, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r::LinRange, x::Number) = LinRange(r.start + x, r.stop + x, length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), x::Number, r::LinRange) = LinRange(x + r.start, x + r.stop, length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(+), r1::AbstractRange, r2::AbstractRange) = r1 + r2

broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r::AbstractUnitRange, x::Number) = range(first(r)-x, length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r::AbstractRange, x::Number) = range(first(r)-x, step=step(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), x::Number, r::AbstractRange) = range(x-first(r), step=-step(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r::StepRangeLen{T}, x::Number) where T =
    StepRangeLen{typeof(T(r.ref)-x)}(r.ref - x, r.step, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), x::Number, r::StepRangeLen{T}) where T =
    StepRangeLen{typeof(x-T(r.ref))}(x - r.ref, -r.step, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r::LinRange, x::Number) = LinRange(r.start - x, r.stop - x, length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), x::Number, r::LinRange) = LinRange(x - r.start, x - r.stop, length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(-), r1::AbstractRange, r2::AbstractRange) = r1 - r2

broadcasted(::DefaultArrayStyle{1}, ::typeof(*), x::Number, r::AbstractRange) = range(x*first(r), step=x*step(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(*), x::Number, r::StepRangeLen{T}) where {T} =
    StepRangeLen{typeof(x*T(r.ref))}(x*r.ref, x*r.step, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(*), x::Number, r::LinRange) = LinRange(x * r.start, x * r.stop, r.len)
# separate in case of noncommutative multiplication
broadcasted(::DefaultArrayStyle{1}, ::typeof(*), r::AbstractRange, x::Number) = range(first(r)*x, step=step(r)*x, length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(*), r::StepRangeLen{T}, x::Number) where {T} =
    StepRangeLen{typeof(T(r.ref)*x)}(r.ref*x, r.step*x, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(*), r::LinRange, x::Number) = LinRange(r.start * x, r.stop * x, r.len)

broadcasted(::DefaultArrayStyle{1}, ::typeof(/), r::AbstractRange, x::Number) = range(first(r)/x, step=step(r)/x, length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(/), r::StepRangeLen{T}, x::Number) where {T} =
    StepRangeLen{typeof(T(r.ref)/x)}(r.ref/x, r.step/x, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(/), r::LinRange, x::Number) = LinRange(r.start / x, r.stop / x, r.len)

broadcasted(::DefaultArrayStyle{1}, ::typeof(\), x::Number, r::AbstractRange) = range(x\first(r), step=x\step(r), length=length(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(\), x::Number, r::StepRangeLen) = StepRangeLen(x\r.ref, x\r.step, length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(\), x::Number, r::LinRange) = LinRange(x \ r.start, x \ r.stop, r.len)

broadcasted(::DefaultArrayStyle{1}, ::typeof(big), r::UnitRange) = big(r.start):big(last(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(big), r::StepRange) = big(r.start):big(r.step):big(last(r))
broadcasted(::DefaultArrayStyle{1}, ::typeof(big), r::StepRangeLen) = StepRangeLen(big(r.ref), big(r.step), length(r), r.offset)
broadcasted(::DefaultArrayStyle{1}, ::typeof(big), r::LinRange) = LinRange(big(r.start), big(r.stop), length(r))

## In specific instances, we can broadcast masked BitArrays whole chunks at a time
# Very intentionally do not support much functionality here: scalar indexing would be O(n)
struct BitMaskedBitArray{N,M}
    parent::BitArray{N}
    mask::BitArray{M}
    BitMaskedBitArray{N,M}(parent, mask) where {N,M} = new(parent, mask)
end
@inline function BitMaskedBitArray(parent::BitArray{N}, mask::BitArray{M}) where {N,M}
    @boundscheck checkbounds(parent, mask)
    BitMaskedBitArray{N,M}(parent, mask)
end
Base.@propagate_inbounds dotview(B::BitArray, i::BitArray) = BitMaskedBitArray(B, i)
Base.show(io::IO, B::BitMaskedBitArray) = foreach(arg->show(io, arg), (typeof(B), (B.parent, B.mask)))
# Override materialize! to prevent the BitMaskedBitArray from escaping to an overrideable method
@inline materialize!(B::BitMaskedBitArray, bc::Broadcasted{<:Any,<:Any,typeof(identity),Tuple{Bool}}) = fill!(B, bc.args[1])
@inline materialize!(B::BitMaskedBitArray, bc::Broadcasted{<:Any}) = materialize!(SubArray(B.parent, to_indices(B.parent, (B.mask,))), bc)
function Base.fill!(B::BitMaskedBitArray, b::Bool)
    Bc = B.parent.chunks
    Ic = B.mask.chunks
    @inbounds if b
        for i = 1:length(Bc)
            Bc[i] |= Ic[i]
        end
    else
        for i = 1:length(Bc)
            Bc[i] &= ~Ic[i]
        end
    end
    return B
end



############################################################

# x[...] .= f.(y...) ---> broadcast!(f, dotview(x, ...), y...).
# The dotview function defaults to getindex, but we override it in
# a few cases to get the expected in-place behavior without affecting
# explicit calls to view.   (All of this can go away if slices
# are changed to generate views by default.)

Base.@propagate_inbounds dotview(args...) = Base.maybeview(args...)

############################################################
# The parser turns @. into a call to the __dot__ macro,
# which converts all function calls and assignments into
# broadcasting "dot" calls/assignments:

dottable(x) = false # avoid dotting spliced objects (e.g. view calls inserted by @view)
# don't add dots to dot operators
dottable(x::Symbol) = (!isoperator(x) || first(string(x)) != '.' || x === :..) && x !== :(:)
dottable(x::Expr) = x.head != :$
undot(x) = x
function undot(x::Expr)
    if x.head == :.=
        Expr(:(=), x.args...)
    elseif x.head == :block # occurs in for x=..., y=...
        Expr(:block, map(undot, x.args)...)
    else
        x
    end
end
__dot__(x) = x
function __dot__(x::Expr)
    dotargs = map(__dot__, x.args)
    if x.head == :call && dottable(x.args[1])
        Expr(:., dotargs[1], Expr(:tuple, dotargs[2:end]...))
    elseif x.head == :comparison
        Expr(:comparison, (iseven(i) && dottable(arg) && arg isa Symbol && isoperator(arg) ?
                               Symbol('.', arg) : arg for (i, arg) in pairs(dotargs))...)
    elseif x.head == :$
        x.args[1]
    elseif x.head == :let # don't add dots to `let x=...` assignments
        Expr(:let, undot(dotargs[1]), dotargs[2])
    elseif x.head == :for # don't add dots to for x=... assignments
        Expr(:for, undot(dotargs[1]), dotargs[2])
    elseif (x.head == :(=) || x.head == :function || x.head == :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], dotargs[2])
    else
        if x.head == :&& || x.head == :||
            error("""
                Using `&&` and `||` is disallowed in `@.` expressions.
                Use `&` or `|` for elementwise logical operations.
                """)
        end
        head = string(x.head)
        if last(head) == '=' && first(head) != '.'
            Expr(Symbol('.',head), dotargs...)
        else
            Expr(x.head, dotargs...)
        end
    end
end
"""
    @. expr
Convert every function call or operator in `expr` into a "dot call"
(e.g. convert `f(x)` to `f.(x)`), and convert every assignment in `expr`
to a "dot assignment" (e.g. convert `+=` to `.+=`).
If you want to *avoid* adding dots for selected function calls in
`expr`, splice those function calls in with `\$`.  For example,
`@. sqrt(abs(\$sort(x)))` is equivalent to `sqrt.(abs.(sort(x)))`
(no dot for `sort`).
(`@.` is equivalent to a call to `@__dot__`.)
# Examples
```jldoctest
julia> x = 1.0:3.0; y = similar(x);
julia> @. y = x + 3 * sin(x)
3-element Array{Float64,1}:
 3.5244129544236893
 4.727892280477045
 3.4233600241796016
```
"""
macro __dot__(x)
    esc(__dot__(x))
end

@inline broadcasted_kwsyntax(f, args...; kwargs...) = broadcasted((args...)->f(args...; kwargs...), args...)
@inline function broadcasted(f, args...)
    args′ = map(broadcastable, args)
    broadcasted(combine_styles(args′...), f, args′...)
end
# Due to the current Type{T}/DataType specialization heuristics within Tuples,
# the totally generic varargs broadcasted(f, args...) method above loses Type{T}s in
# mapping broadcastable across the args. These additional methods with explicit
# arguments ensure we preserve Type{T}s in the first or second argument position.
@inline function broadcasted(f, arg1, args...)
    arg1′ = broadcastable(arg1)
    args′ = map(broadcastable, args)
    broadcasted(combine_styles(arg1′, args′...), f, arg1′, args′...)
end
@inline function broadcasted(f, arg1, arg2, args...)
    arg1′ = broadcastable(arg1)
    arg2′ = broadcastable(arg2)
    args′ = map(broadcastable, args)
    broadcasted(combine_styles(arg1′, arg2′, args′...), f, arg1′, arg2′, args′...)
end
@inline broadcasted(::S, f, args...) where S<:BroadcastStyle = Broadcasted{S}(f, args)

end # module
