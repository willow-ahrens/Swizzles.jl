module Swizzles

export Drop, drop
export getindexinto, setindexinto

include("base.jl")
include("util.jl")

using Base: checkbounds_indices, throw_boundserror, tail
using Base.Iterators: repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, broadcast_axes, instantiate, broadcastable, longest_tuple

export swizzle, swizzle!
export Swizzle, Reduce, Sum, Max, Min, Beam
export SwizzleTo, ReduceTo, SumTo, MaxTo, MinTo, BeamTo

### An operator which does not expect to be called.

"""
    `nooperator(a, b)`

An operator which does not expect to be called. It startles easily.
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

@inline function Swizzled(arg, mask, op)
    arg = instantiate(broadcastable(arg))
    return _Swizzled(arg, Val(mask), op)
end
@inline function _Swizzled(arg, ::Val{_mask}, op) where {_mask}
    if @generated
        mask = (take(flatten((_mask, repeated(drop))), arg isa Tuple ? 1 : ndims(arg))...,)
        imask = setindexinto(ntuple(d->drop, maximum((0, mask...))), 1:length(mask), mask)
        return :(return Swizzled{$arg, $mask, $imask, $op}(arg, op))
    else
        mask = (take(flatten((mask, repeated(drop))), arg isa Tuple ? 1 : ndims(typeof(arg)))...,)
        imask = setindexinto(ntuple(d->drop, maximum((0, mask...))), 1:length(mask), mask)
        return Swizzled{typeof(arg), mask, imask, typeof(op)}(arg, op)
    end
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

@inline function Base.axes(sz::Swizzled)
    if @generated
        args = getindexinto(ntuple(d->:(Base.OneTo(1)), length(imask(sz))), ntuple(d->:(bc_axes[$d]), length(mask(sz))), imask(sz))
        return quote
            bc_axes = broadcast_axes(sz.arg)
            return ($(args...),)
        end
    else
        getindexinto(ntuple(d->Base.OneTo(1), length(imask(sz))), broadcast_axes(sz.arg), imask(sz))
    end
end

function Base.eltype(sz::Swizzled; depth = 4)
    T = eltype(sz.arg)
    if eltype(mask(sz)) isa Int
        return T
    end
    for i in 1:depth
        T! = Union{T, Base._return_type(sz.op, (T, T))}
        if T! == T
            return T
        end
        T = T!
    end
    return Any
end

#=
@inline Base.eltype(sz::Swizzled) = _eltype(sz, Val(4))
@inline function _eltype(sz::Swizzled, ::Val{N}) where {N}
    T = eltype(sz.arg)
    if eltype(mask(sz)) isa Int
        return T
    end
    if @generated
        thunk = Expr(:block)
        for i in 1:N
            push!(thunk.args, begin
                T! = Union{T, Base._return_type(sz.op, (T, T))}
                if T! == T
                    return T
                end
                T = T!
            end)
        end
        thunk
    else
        for i in 1:N
            T! = Union{T, Base._return_type(sz.op, (T, T))}
            if T! == T
                return T
            end
            T = T!
        end
    end
    return Any
end
=#


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
Base.IteratorEltype(sz::Type{<:Swizzled}) = EltypeUnknown()

@inline function _getindex(sz::Swizzled, I::Int...)
    @boundscheck checkbounds_indices(Bool, axes(sz), I) || throw_boundserror(sz, I)
    if @generated
        args = getindexinto(ntuple(d->:(bc_axes[$d]), length(mask(sz))), ntuple(d->:(I[$d]), length(imask(sz))), mask(sz))
        quote
            bc_axes = broadcast_axes(sz.arg)
            inds = product($(args...))
        end
    else
        inds = product(getindexinto(broadcast_axes(sz.arg), I, mask(sz))...)
    end
    (i, inds) = peel(inds)
    res = @inbounds getindex(sz.arg, i...)
    for i in inds
        res = sz.op(res, @inbounds getindex(sz.arg, i...))
    end
    return res
end
@inline Base.getindex(sz::Swizzled, I::Int) = _getindex(sz, I)
@inline Base.getindex(sz::Swizzled, I::CartesianIndex) = _getindex(sz, Tuple(I)...)
@inline Base.getindex(sz::Swizzled, I::Int...) = _getindex(sz, I...)
@inline Base.getindex(sz::Swizzled) = _getindex(sz, ())

"""
    `swizzle(A, mask, op=nooperator)`

Create a new obect `B` such that the dimension `i` of `A` is mapped to
dimension `mask[i]` of `B`, operating on lazy broadcast expressions, arrays,
tuples, collections, [`Ref`](@ref)s and/or scalars `As`. If `mask[i]` is an
instance of the singleton type `Drop`, the dimension is reduced over using
`op`. `mask` may be any (possibly infinite) iterable over elements of type
`Int` and `Drop`. The integers in `mask` must be unique, and if `mask` is not
long enough, additional `Drop`s are added to the end.
The resulting container type is established by the following rules:
 - If all elements of `mask` are `Drop`, it returns an unwrapped scalar.
 - All other combinations of arguments default to returning an `Array`, but
   custom container types can define their own implementation rules to
   customize the result when they appear as an argument.
The swizzle operation is represented with a special lazy `Swizzled` type.
`swizzle` results in `copy(Swizzled(...))`.  The swizzle operation can use the
`Swizzle` type to take advantage of special broadcast syntax. A statement like:
```
   y = Swizzle((1,), +).(x .* (Swizzle((2, 1)).x .+ 1))
```
will result in code that is essentially:
```
   y = copy(Swizzled(Broadcasted(*, Swizzled(x, (2, 1)), Broadcasted(+, x, 1)), (1,), +))
```
If `Swizzled`s are mixed with `Broadcasted`s, the result is fused into one big operation.

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
swizzle(A, mask, op=nooperator) = copy(Swizzled(A, mask, op))

"""
    `swizzle!(dest, A, mask, op=nooperator)`

Like [`swizzle`](@ref), but store the result of `swizzle(A, mask, op)` in the
`dest` array.  Results in `copyto!(dest, Swizzled(...))`.

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
swizzle!(dest, A, mask, op=nooperator) = copyto!(dest, Swizzled(A, mask, op))

Base.Broadcast.broadcastable(sz::Swizzled) = sz

@inline Base.Broadcast.materialize(sz::Swizzled) = copy(sz)

@inline Base.Broadcast.materialize!(dest, sz::Swizzled) = copyto!(dest, sz)

@inline Base.copy(sz::Swizzled) = copy(instantiate(Broadcasted(identity, (sz,))))

@inline Base.copyto!(dest, sz::Swizzled) = copyto!(dest, instantiate(Broadcasted(identity, (sz,))))

@inline Base.copyto!(dest::AbstractArray, sz::Swizzled) = copyto!(dest, instantiate(Broadcasted(identity, (sz,))))

#Base.Broadcast.preprocess(dest, sz::Swizzled{Style}) where {Style} = instantiate(Swizzled{Style}(sz.arg, mask(sz), sz.op, sz.axes, imask(sz))) #TODO problem here too.
#Base.Broadcast.broadcasted(style::BroadcastStyle, szr::Swizzler, arg) = Swizzled(arg, szr.mask, szr.op) #Should use style here duh.

"""
    `SwizzleStyle(style, ::Type{<:Swizzled})`

Broadcast styles are used to determine behavior of objects under swizzling.  To
customize the swizzling behavior of a type, one can first define an appropriate
Broadcast style for the the type, then declare how the broadcast style should
behave under broadcasting after the swizzle by overriding the
`SwizzleStyle` method.
"""
SwizzleStyle #FIXME only define on ur stuff

SwizzleStyle(::Style{Tuple}, Sz) = first(mask(Sz)) == 1 ? Style{Tuple}() : DefaultArrayStyle(Val(max(0, first(mask(Sz)))))
Broadcast.longest_tuple(::Nothing, t::Tuple{<:Swizzled{<:Any, (1,)},Vararg{Any}}) = longest_tuple(longest_tuple(nothing, (t[1].arg,)), tail(t))
SwizzleStyle(style::A, Sz) where {A <: AbstractArrayStyle{0}} = A(Val(0))
function SwizzleStyle(style::A, ::Type{Sz}) where {N, A <: AbstractArrayStyle{N}, Sz}
    if @generated
        return :(return A(Val($(max(0, maximum(take(mask(Sz), N)))))))
    else
        return A(Val(max(0, maximum(take(mask(Sz), N)))))
    end
end
SwizzleStyle(style::AbstractArrayStyle{Any}, Sz) = style
SwizzleStyle(::BroadcastStyle, Sz) = Unknown()
SwizzleStyle(::ArrayConflict, Sz) = ArrayConflict() #FIXME

@inline Broadcast.BroadcastStyle(Sz::Type{Swizzled{Arg, mask, imask, Op}}) where {Arg, mask, imask, Op} = SwizzleStyle(BroadcastStyle(Arg), Sz)

#=
function _SwizzleStyle(style, sz)
    
end
=#


struct Swizzle{mask, Op}
    op::Op
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
    Swizzle(mask, op::Op=nooperator) where {Op} = new{mask, Op}(op)
end

mask(sz::Swizzle) = mask(typeof(sz))
mask(::Type{Swizzle{_mask, Op}}) where {_mask, Op} = _mask

@inline Base.Broadcast.broadcasted(style::BroadcastStyle, sz::Swizzle, arg) = Swizzled(arg, mask(sz), sz.op)

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
