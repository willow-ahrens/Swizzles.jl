using Swizzles.Properties
using Swizzles.WrapperArrays
using Swizzles.BroadcastedArrays
using Swizzles.GeneratedArrays
using Swizzles.ExtrudedArrays
using Base: checkbounds_indices, throw_boundserror, tail, dataids, unaliascopy, unalias
using Base.Iterators: reverse, repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, instantiate, broadcastable, preprocess, _broadcast_getindex, combine_eltypes

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

@inline SwizzledArray{T}(arr::SwizzledArray{S, N, Arg, mask, Op}) where {T, S, N, Arg, mask, Op} = SwizzledArray{T, N, Arg, mask, Op}(arr.arg, arr.op)
@inline SwizzledArray{T, N, Arg, mask, Op}(arr::SwizzledArray{S, N, Arg, mask, Op}) where {T, S, N, Arg, mask, Op} = SwizzledArray{T, N, Arg, mask, Op}(arr.arg, arr.op)

@inline function SwizzledArray(arg, mask, op)
    arr = SwizzledArray{Any}(arg, mask, op)
    return SwizzledArray{Properties.eltype_bound(arg)}(arr)
end

@inline SwizzledArray{T}(arg, mask, op) where {T} = SwizzledArray{T}(arg, Val(mask), op)
@inline function SwizzledArray{T}(arg, ::Val{mask}, op) where {T, mask}
    if @generated
        mask! = (take(flatten((mask, repeated(drop))), ndims(arg))...,)
        M = maximum((0, mask!...))
        #return :(return SwizzledArray{T, $M, typeof(arg), $mask!, Core.Typeof(op)}(arg, op))
        return :(return SwizzledArray{T, $M, typeof(arg), $mask!, typeof(op)}(arg, op))
    else
        mask! = (take(flatten((mask, repeated(drop))), ndims(arg))...,)
        M = maximum((0, mask!...))
        #return SwizzledArray{T, M, typeof(arg), mask!, Core.Typeof(op)}(arg, op)
        return SwizzledArray{T, M, typeof(arg), mask!, typeof(op)}(arg, op)
    end
end

mask(::Type{SwizzledArray{T, N, Arg, _mask, Op}}) where {T, N, Arg, _mask, Op} = _mask
mask(arr::S) where {S <: SwizzledArray} = mask(S)

@inline function Properties.eltype_bound(arr::SwizzledArray)
    T = Properties.eltype_bound(arr.arg)
    if eltype(mask(arr)) <: Int
        return T
    end
    T! = Union{T, Properties.return_type(arr.op, T, T)}
    if T! <: T
        return T!
    end
    T = T!
    T! = Union{T, Properties.return_type(arr.op, T, T)}
    if T! <: T
        return T!
    end
    return Any
end

struct Swizzle{T, mask, Op} <: Swizzles.Intercept
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
@inline Swizzle(mask, op::Op) where {Op} = Swizzle{nothing}(mask, op)

"""
    `Swizzle{T}(mask, op=nooperator)`

Similar to [`Swizzle`](@ref), but the eltype of the resulting swizzle is
declared to be `T`.

See also: [`Swizzle`](@ref).
"""
@inline Swizzle{T}(mask, op::Op) where {T, Op} = Swizzle{T, mask, Op}(op)

mask(::Type{Swizzle{T, _mask, Op}}) where {T, _mask, Op} = _mask
mask(sz::Sz) where {Sz <: Swizzle} = mask(Sz)

@inline (sz::Swizzle{nothing})(arg) = SwizzledArray(arrayify(arg), Val(mask(sz)), sz.op)
@inline (sz::Swizzle{T})(arg) where {T} = SwizzledArray{T}(arrayify(arg), Val(mask(sz)), sz.op)



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
Base.@propagate_inbounds function Base.copy(src::Broadcasted{DefaultArrayStyle{0}, <:Any, typeof(identity), <:Tuple{SubArray{T, <:Any, <:SwizzledArray{T, N}, <:NTuple{N}}}}) where {T, N}
    #A view of a Swizzle can be computed as a swizzle of a view (hiding the
    #complexity of dropping view indices). Therefore, we convert first.
    dst = Array{T, 0}(undef) #if you use this method for scalar getindex, this line can get pretty costly
    Base.copyto!(dst, convert(SwizzledArray, src.args[1]))
    return dst[]
end
=#

Base.@propagate_inbounds function Base.copy(src::Broadcasted{DefaultArrayStyle{0}, <:Any, typeof(identity), <:Tuple{SubArray{T, <:Any, <:SwizzledArray{T, N}, <:NTuple{N}}}}) where {T, N}
    arr = parent(src.args[1])
    if mask(arr) isa Tuple{Vararg{Int}}
        #This swizzle doesn't reduce so this is just a scalar getindex.
        arg = parent(arr)
        inds = parentindices(src.args[1])
        @inbounds return arg[parentindex(arr, inds...)...]
    else
        #A view of a Swizzle can be computed as a swizzle of a view (hiding the
        #complexity of dropping view indices). Therefore, we convert first.
        return Base.copy(Broadcasted{DefaultArrayStyle{0}}(identity, (convert(SwizzledArray, src.args[1]),)))
    end
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{Nothing, <:Any, typeof(identity), <:Tuple{SubArray{T, <:Any, <:SwizzledArray{T, N}, <:NTuple{N}}}}) where {T, N}
    #A view of a Swizzle can be computed as a swizzle of a view (hiding the
    #complexity of dropping view indices). Therefore, we convert first.
    return Base.copyto!(dst, convert(SwizzledArray, src.args[1]))
end

@inline function Base.convert(::Type{SwizzledArray}, src::SubArray{T, M, Arr, <:NTuple{N}}) where {T, N, M, Arg, Arr <: SwizzledArray{T, N, Arg}}
    arr = parent(src)
    arg = parent(arr)
    inds = parentindices(src)
    if @generated
        if M == 0
            mask′ = ((filter(d -> d isa Drop, collect(mask(Arr)))...,))
        else
            mask′ = :(_convert_remask(inds, mask(arr)...))
        end
        quote
            return SwizzledArray{eltype(src)}(SubArray(arg, parentindex(arr, inds...)), $mask′, arr.op)
        end
    else
        mask′ = _convert_remask(inds, mask(arr)...)
        return SwizzledArray{eltype(src)}(SubArray(arg, parentindex(arr, inds...)), mask′, arr.op)
    end
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

Base.@propagate_inbounds function Base.copy(src::Broadcasted{DefaultArrayStyle{0}, <:Any, typeof(identity), <:Tuple{<:SwizzledArray}}) where {T, N}
    Base.@_propagate_inbounds_meta
    arr = src.args[1]
    arg = arr.arg
    if mask(arr) isa Tuple{Vararg{Int}}
        @inbounds return arg[]
    elseif Properties.initial(arr.op, eltype(arr), eltype(arg)) != nothing
        dst = something(Properties.initial(arr.op, eltype(arr), eltype(arg)))
        arg = BroadcastedArrays.preprocess(dst, arg)
        for i in eachindex(arg)
            @inbounds dst = arr.op(dst, arg[i])
        end
        return dst
    else
        (i, inds) = peel(eachindex(arg))
        @inbounds dst = arg[i]
        for i in inds
            @inbounds dst = arr.op(dst, arg[i])
        end
        return dst
    end
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::SwizzledArray)
    #This method gets called when the destination eltype is unsuitable for
    #accumulating the swizzle. Therefore, we should allocate a suitable
    #destination and then accumulate.
    copyto!(dst, copyto!(similar(src), src))
end

@generated function Base.copyto!(dst::AbstractArray{T, N}, src::SwizzledArray{<:T, N}) where {T, N}
    quote
        Base.@_propagate_inbounds_meta
        arg = src.arg
        if mask(src) isa Tuple{Vararg{Int}}
            arg = BroadcastedArrays.preprocess(dst, src.arg)
            for i in eachindex(arg)
                #i′ = setindexinto(map(firstindex, axes(dst)), Tuple(CartesianIndices(arg)[i]), mask(src))
                i′ = childindex(dst, src, i)
                @inbounds dst[i′...] = arg[i]
            end
        elseif Properties.initial(src.op, eltype(src), eltype(arg)) != nothing
            dst .= something(Properties.initial(src.op, eltype(src), eltype(src.arg)))
            dst .= (src.op).(dst, src)
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
        end
        return dst
    end
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray{T}, src::Broadcasted{Nothing, <:Any, Op, <:Tuple{<:Any, <:SwizzledArray{<:T, <:Any, <:Any, <:Any, Op}}}) where {T, Op}
    copyto!(dst, src.args[1])
    arr = src.args[2]
    arg = BroadcastedArrays.preprocess(dst, arr.arg)
    for i in eachindex(arg)
        #i′ = setindexinto(map(firstindex, axes(dst)), Tuple(CartesianIndices(arg)[i]), mask(arr))
        i′ = childindex(dst, arr, i)
        @inbounds dst[i′...] = arr.op(dst[i′...], arg[i])
    end
    return dst
end

@inline function parentindex(arr::SubArray, i...)
    return Base.reindex(arr, Base.parentindices(arr), i)
end

@inline function parentindex(arr::SwizzledArray, i::Integer)
    return parentindex(arr, CartesianIndices(arr)[i])
end

@inline function parentindex(arr::SwizzledArray, i::CartesianIndex)
    return parentindex(arr, Tuple(i)...)
end

@inline function parentindex(arr::SwizzledArray{<:Any, 1}, i::Integer)
    return invoke(parentindex, Tuple{typeof(arr), Any}, arr, i)
end

@inline function parentindex(arr::SwizzledArray{<:Any, N, Arg}, i::Vararg{Any, N}) where {N, Arg}
    if @generated
        quote
            return ($(getindexinto([:(Base.Slice(axes(arr.arg, $d))) for d in 1:ndims(Arg)], [:(i[$d]) for d in 1:ndims(arr)], mask(arr))...),)
        end
    else
        return getindexinto(map(Base.Slice, axes(arr.arg)), i, mask(arr))
    end
end


"""
   parentindex(arr, i)

   For all wrapper arrays arr such that `arr` involves an index remapping,
   return the indices into `parent(arr)` which affect the indices `i` of `arr`.

   See also: [`swizzle`](@ref).
"""
parentindex

@inline function childindex(dst::AbstractArray{<:Any, N}, arr::SwizzledArray{<:Any, N}, i::Integer) where {N}
    return childindex(dst, arr, CartesianIndices(arr.arg)[i])
end

@inline function childindex(dst::AbstractArray{<:Any, N}, arr::SwizzledArray{<:Any, N}, i::CartesianIndex) where {N}
    return childindex(dst, arr, Tuple(i)...)
end

@inline function childindex(dst::AbstractArray{<:Any, N}, arr::SwizzledArray{<:Any, N, <:AbstractArray{<:Any, M}}, i::Vararg{Integer, M}) where {N, M}
    if @generated
        quote
            return ($(setindexinto([:(firstindex(dst, $d)) for d in 1:ndims(dst)], [:(i[$d]) for d in 1:length(mask(arr))], mask(arr))...),)
        end
    else
        return setindexinto(map(firstindex, axes(dst)), i, mask(arr))
    end
end

"""
   childindex(arr, i)

   For all wrapper arrays arr such that `arr` involves an index remapping,
   return the indices into `arr` which affect the indices `i` of `parent(arr)`.

   See also: [`swizzle`](@ref).
"""
childindex

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
swizzle(A, mask, op=nooperator) = materialize(SwizzledArray(arrayify(A), mask, op))

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
swizzle!(dest, A, mask, op=nooperator) = materialize!(dest, SwizzledArray(arrayify(A), mask, op))

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

@inline function Swizzles.ExtrudedArrays.keeps(arr::SwizzledArray)
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

function Swizzles.ExtrudedArrays.keeps(::Type{Arr}) where {Arg, Arr <: SwizzledArray{<:Any, <:Any, <:Arg}}
    setindexinto(ntuple(d->false, ndims(Arr)), keeps(Arg), mask(Arr))
end

function Swizzles.ExtrudedArrays.lift_keeps(arr::SwizzledArray{T, N, Arg, mask, Op}) where {T, N, Arg, mask, Op}
    arg = arrayify(lift_keeps(arr.arg))
    return SwizzledArray{T, N, typeof(arg), mask, Op}(arg, arr.op)
end
