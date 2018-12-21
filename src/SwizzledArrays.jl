using Swizzle.WrapperArrays
using Swizzle.BroadcastedArrays
using Swizzle.GeneratedArrays
using Swizzle.ExtrudedArrays
using Swizzle.MatchedArrays
using Base: checkbounds_indices, throw_boundserror, tail, dataids, unaliascopy, unalias
using Base.Iterators: reverse, repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, broadcast_axes, instantiate, broadcastable, preprocess, _broadcast_getindex, combine_eltypes

@inline myidentity(x) = x

"""
    `nooperator(a, b)`

An operator which does not expect to be called. It startles easily.
"""
nooperator(a, b) = throw(ArgumentError("unspecified operator"))

struct SwizzledArray{T, N, Arg<:AbstractArray, mask, Op} <: GeneratedArray{T, N}
    arg::Arg
    op::Op
    function SwizzledArray{T, N, Arg, mask, Op}(arg::Arg, op::Op) where {T, N, Arg, mask, Op}
        #FIXME check swizzles. also check noop axes!
        new(arg, op)
    end
end

@inline function SwizzledArray(arr::SwizzledArray)
    T = eltype(arr.arg)
    if eltype(mask(arr)) <: Int
        return SwizzledArray{T}(arr)
    end
    T! = Union{T, get_return_type(arr.op, T, T)}
    if T! <: T
        return SwizzledArray{T!}(arr)
    end
    T = T!
    T! = Union{T, get_return_type(arr.op, T, T)}
    if T! <: T
        return SwizzledArray{T!}(arr)
    end
    return SwizzledArray{Any}(arr)
end

@inline SwizzledArray{T}(arr::SwizzledArray{S, N, Arg, mask, Op}) where {T, S, N, Arg, mask, Op} = SwizzledArray{T, N, Arg, mask, Op}(arr.arg, arr.op)

@inline SwizzledArray(arg, mask, op) = SwizzledArray(_SwizzledArray(Any, arrayify(arg), Val(mask), op))

@inline SwizzledArray{T}(arg, mask, op) where {T} = _SwizzledArray(T, arrayify(arg), Val(mask), op)

@inline function _SwizzledArray(::Type{T}, arg::AbstractArray{S, N}, ::Val{mask}, op) where {T, S, N, mask}
    if @generated
        mask! = (take(flatten((mask, repeated(drop))), N)...,)
        M = maximum((0, mask!...))
        #return :(return SwizzledArray{T, $M, typeof(arg), $mask!, Core.Typeof(op)}(arg, op))
        return :(return SwizzledArray{T, $M, typeof(arg), $mask!, typeof(op)}(arg, op))
    else
        mask! = (take(flatten((mask, repeated(drop))), N)...,)
        M = maximum((0, mask!...))
        #return SwizzledArray{T, M, typeof(arg), mask!, Core.Typeof(op)}(arg, op)
        return SwizzledArray{T, M, typeof(arg), mask!, typeof(op)}(arg, op)
    end
end

mask(::Type{SwizzledArray{T, N, Arg, _mask, Op}}) where {T, N, Arg, _mask, Op} = _mask
mask(arr::S) where {S <: SwizzledArray} = mask(S)

operator(arr::S) where {S <: SwizzledArray} = arr.op

struct Swizzler{mask, Op} <: Arrayifier
    op::Op
end

"""
    `Swizzler(mask, op=nooperator)`

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
julia> Swizzler((1,), +).(A)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> Swizzler((), +).(A)
55
julia> Swizzler((2,)).(parse.(Int, ["1", "2"]))
1x2-element Array{Int64,1}:
 1 2
```
"""
@inline Swizzler(mask, op::Op) where {Op} = Swizzler{mask, Op}(op)

mask(::Type{Swizzler{_mask, Op}}) where {_mask, Op} = _mask
mask(szr::Szr) where {Szr <: Swizzler} = mask(Szr)

@inline (szr::Swizzler)(arg::AbstractArray) = SwizzledArray(_SwizzledArray(Any, arg, Val(mask(szr)), szr.op))

function Base.show(io::IO, arr::SwizzledArray)
    print(io, SwizzledArray)
    print(io, '(', arr.arg, ", ", mask(arr), ", ", arr.op, ')')
    nothing
end

Base.parent(arr::SwizzledArray) = arr.arg
Base.parent(::Type{<:SwizzledArray{<:Any, <:Any, Arg}}) where {Arg} = Arg
WrapperArrays.iswrapper(arr::SwizzledArray) = true
function WrapperArrays.adopt(arg, arr::SwizzledArray{T, N, <:Any, mask, Op}) where {T, N, mask, Op}
    SwizzledArray{T, N, typeof(arg), mask, Op}(arg, arr.op)
end

Base.dataids(arr::SwizzledArray) = (dataids(arr.arg), dataids(arr.op))
Base.unaliascopy(arr::SwizzledArray{T, N, Arg, mask, Op}) where {T, N, Arg, mask, Op} = SwizzledArray{T, N, Arg, mask, Op}(unaliascopy(arr.arg), unaliascopy(arr.op))
Base.unalias(dest, arr::SwizzledArray{T, N, Arg, mask, Op}) where {T, N, Arg, mask, Op} = SwizzledArray{T, N, Arg, mask, Op}(unalias(dest, arr.arg), unalias(dest, arr.op))

@inline function Base.size(arr::SwizzledArray)
    if @generated
        args = setindexinto(ntuple(d->:(1), ndims(arr)), ntuple(d->:(arg_size[$d]), length(mask(arr))), mask(arr))
        return quote
            arg_size = size(arr.arg)
            return ($(args...),)
        end
    else
        setindexinto(ntuple(d->1, ndims(arr)), size(arr.arg), mask(arr))
    end
end

@inline function Base.axes(arr::SwizzledArray)
    if @generated
        args = setindexinto(ntuple(d->:(Base.OneTo(1)), ndims(arr)), ntuple(d->:(arg_axes[$d]), length(mask(arr))), mask(arr))
        return quote
            arg_axes = axes(arr.arg)
            return ($(args...),)
        end
    else
        setindexinto(ntuple(d->Base.OneTo(1), ndims(arr)), axes(arr.arg), mask(arr))
    end
end

#=
    (i, inds) = peel(product(getindexinto(axes(arr.arg), I, mask(arr))...))
    res = @inbounds getindex(arr.arg, i...)
    for i in inds
        res = arr.op(res, @inbounds getindex(arr.arg, i...))
    end
    return res
=#

function Base.convert(::Type{SwizzledArray}, src::SubArray{T, <:Any, <:SwizzledArray{T, N}, <:NTuple{N}}) where {T, N}
    arr = parent(src)
    arg = parent(arr)
    return SwizzledArray{eltype(src)}(
        SubArray(arg,
            getindexinto(
                map(Base.Slice, axes(arg)),
                parentindices(src),
                mask(arr))),
        _convert_remask(parentindices(src), mask(arr)...),
        operator(arr))
end

@inline function _convert_remask(indices, d, mask...)
    if d isa Drop
        (drop, _convert_remask(indices, mask...)...)
    elseif Base.index_ndims(indices[d]) isa Tuple{}
        _convert_remask(indices, mask...)
    else
        (length(Base.index_ndims(indices[1:d])), _convert_remask(indices, mask...)...)
    end
end
@inline _convert_remask(indices) = ()

function Base.copyto!(dst::AbstractArray, src::SubArray{T, <:Any, <:SwizzledArray{T, N}, <:NTuple{N}}) where {T, N}
    #A view of a Swizzle can be computed as a swizzle of a view (hiding the
    #complexity of dropping view indices). Therefore, we convert to a
    #accumulating the swizzle. Therefore, we should allocate a suitable
    #destination and then accumulate.
    return Base.copyto!(dst, convert(SwizzledArray, src))
end

@generated function Base.copyto!(dst::AbstractArray, src::SwizzledArray)
    #This method gets called when the destination eltype is unsuitable for
    #accumulating the swizzle. Therefore, we should allocate a suitable
    #destination and then accumulate.
    copyto!(dst, copy(src))
end
@generated function Base.copyto!(dst::AbstractArray{T}, src::SwizzledArray{<:T}) where {T}
    thunk = quote
        Base.@_propagate_inbounds_meta
        arg = src.arg
        if has_identity(operator(src), eltype(src), eltype(arg))
            dst .= (get_identity(operator(src), eltype(src), eltype(src.arg)))
            dst .= operator(src).(dst, src)
        else
            arg = BroadcastedArrays.preprocess(dst, src.arg)
            arg_axes = axes(arg)
            arg_firstindices = ($((:(arg_axes[$n][1]) for n = 1:length(mask((src))))...),)
            arg_restindices = ($((:(view(arg_axes[$n], 2:lastindex(arg_axes[$n]))) for n = 1:length(mask(src)))...),)
            arg_keeps = keeps(arg)
            $(begin
                i′ = [Symbol("i′_$d") for d = 1:length(mask(src))]
                i = setindexinto([:(firstindex(dst, $d)) for d in 1:ndims(dst)], i′, mask(src))
                thunk = Expr(:block)
                for n = vcat(0, findall(d -> d isa Drop, mask(src)))
                    if n > 0
                        nest = :(dst[$(i...)] = src.op(dst[$(i...)], @inbounds getindex(arg, $(i′...))))
                    else
                        nest = :(dst[$(i...)] = @inbounds getindex(arg, $(i′...)))
                    end
                    for d = 1:length(mask(src))
                        if mask(src)[d] isa Drop
                            if d == n
                                nest = Expr(:for, :($(Symbol("i′_$d")) = arg_restindices[$d]), nest)
                            elseif d < n
                                nest = Expr(:for, :($(Symbol("i′_$d")) = arg_axes[$d]), nest)
                            else
                                nest = Expr(:block, :($(Symbol("i′_$d")) = arg_firstindices[$d]), nest)
                            end
                        else
                            nest = Expr(:for, :($(Symbol("i′_$d")) = arg_axes[$d]), nest)
                        end
                    end
                    if n > 0
                        nest = Expr(:if, :(arg_keeps[$n]), nest)
                    end
                    push!(thunk.args, nest)
                end
                thunk
            end)
            return dst
        end
    end
    #:(println($(QuoteNode(thunk))); $thunk)
    thunk
end

function Base.copyto!(dst::AbstractArray{T}, src::Broadcasted{Nothing, <:Any, Op, <:Tuple{<:Any, <:SwizzledArray{<:T, <:Any, <:Any, <:Any, Op}}}) where {T, Op}
    copyto!(dst, src.args[1])
    arr = src.args[2]
    arg = BroadcastedArrays.preprocess(dst, arr.arg)
    AXE = map(firstindex, axes(dst))
    CI  = CartesianIndices(arg)
    msk = mask(arr)
    @inbounds for i in eachindex(arg)
        i′ = setindexinto(AXE, Tuple(CI[i]), msk)
        dst[i′...] = arr.op(dst[i′...], arg[i])
    end
    return dst
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
`Swizzler` type to take advantage of special broadcast syntax. A statement like:
```
   y = Swizzler((1,), +).(x .* (Swizzler((2, 1)).x .+ 1))
```
will result in code that is essentially:
```
   y = materialize(SwizzledArray(BroadcastedArray(Broadcasted(*, SwizzledArray(x, (2, 1)), Broadcasted(+, x, 1))), (1,), +))
```
If `SwizzledArray`s are mixed with `Broadcasted`s, the result is fused into one big operation.

See also: [`swizzle!`](@ref), [`Swizzler`](@ref).

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
swizzle(A, mask, op=nooperator) = materialize(SwizzledArray(A, mask, op))

"""
    `swizzle!(dest, A, mask, op=nooperator)`

Like [`swizzle`](@ref), but store the result of `swizzle(A, mask, op)` in the
`dest` type.  Results in `materialize!(dest, SwizzledArray(...))`.

See also: [`swizzle`](@ref), [`Swizzler`](@ref).

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
swizzle!(dest, A, mask, op=nooperator) = materialize!(dest, SwizzledArray(A, mask, op))

#function Base.Broadcast.preprocess(dest, arr::SwizzledArray{T, N, Arg, mask, Op}) where {T, N, Arg, mask, Op}
#    arg = preprocess(dest, arr.arg)
#    SwizzledArray{T, N, typeof(arg), mask, Op}(arg, arr.op)
#end

"""
    `SwizzleStyle(style, ::Type{<:SwizzledArray})`

Broadcast styles are used to determine behavior of objects under broadcasting.
To customize the broadcasting behavior of a type under swizzling, one can first
define an appropriate Broadcast style for the the type, then declare how the
broadcast style should behave under broadcasting after the swizzle by
overriding the `SwizzleStyle` method.
"""
SwizzleStyle

function SwizzleStyle(::S, ::Type{A}) where {N, S <: AbstractArrayStyle{N}, A <:SwizzledArray} #TODO orthogonalize
    if @generated
        return :(return S(Val($(max(0, maximum(take(mask(A), N)))))))
    else
        return S(Val(max(0, maximum(take(mask(A), N)))))
    end
end
SwizzleStyle(::BroadcastStyle, arr) = DefaultArrayStyle{ndims(arr)}()
SwizzleStyle(::ArrayConflict, arr) = ArrayConflict()

@inline function Broadcast.BroadcastStyle(::Type{A}) where {T, N, Arg, A <: SwizzledArray{T, N, Arg}}
    if @generated
        if mask(A) == ((1:ndims(Arg))...,)
            return quote
                Base.@_inline_meta()
                return BroadcastStyle(Arg)
            end
        else
            return quote
                Base.@_inline_meta()
                return (SwizzleStyle(BroadcastStyle(Arg), A))
            end
        end
    else
        if mask(A) == ((1:ndims(Arg))...,)
            return BroadcastStyle(Arg)
        else
            return (SwizzleStyle(BroadcastStyle(Arg), A))
        end
    end
end

#=
@inline function Broadcast.broadcastable(arr::SwizzledArray{T, N, Arg}) where {T, N, Arg, A <: }
    if @generated
        if mask(arr) == ((1:length(ndims(Arg)))...)
            return quote
                Base.@_inline_meta()
                return arr.arg
            end
        else
            return quote
                Base.@_inline_meta()
                return arr
            end
        end
    else
        if mask(arr) == ((1:length(ndims(Arg)))...)
            return arr.arg
        else
            return arr
        end
    end
end
=#

@inline function Swizzle.ExtrudedArrays.keeps(arr::SwizzledArray)
    if @generated
        args = setindexinto(ntuple(d->:(false), ndims(arr)), ntuple(d->:(arg_keeps[$d]), length(mask(arr))), mask(arr))
        return quote
            arg_keeps = keeps(arr.arg)
            return ($(args...),)
        end
    else
        setindexinto(ntuple(d->false, ndims(arr)), keeps(arr.arg), mask(arr))
    end
end

function Swizzle.ExtrudedArrays.keeps(::Type{Arr}) where {Arg, Arr <: SwizzledArray{<:Any, <:Any, <:Arg}}
    setindexinto(ntuple(d->false, ndims(Arr)), keeps(Arg), mask(Arr))
end

function Swizzle.ExtrudedArrays.lift_keeps(arr::SwizzledArray{T, N, Arg, mask, Op}) where {T, N, Arg, mask, Op}
    arg = arrayify(lift_keeps(arr.arg))
    return SwizzledArray{T, N, typeof(arg), mask, Op}(arg, arr.op)
end
