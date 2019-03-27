module ShuffledArrays

using StaticArrays
using Swizzles.WrapperArrays
using Swizzles.ShallowArrays
using Swizzles.ArrayifiedArrays
using Swizzles.ChunkedUnitRanges
using Swizzles.ExtrudedArrays
using Swizzles.ProductArrays
using Swizzles
using Swizzles: SwizzledArray
using Swizzles: mask, jointuple, combinetuple, masktuple

using Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle
using Base.Broadcast: broadcast_shape, check_broadcast_shape, result_style, axistype, broadcasted
using Base.Iterators: repeated, flatten, take

export ShuffleStyle, decks
export Shuffle

"""
    decks(arr::ShuffledArray)

Return the permutations of the axes of `parent(arr)`, such that
`arr[i...] = parent(arr)[map(getindex, decks(arr), i)]
"""
function decks(arr)
    return map(Base.OneTo, size(arr))
end

"""
    decks(arr, n)

Return the permutation of the axis `n` of `parent(arr)`, such that
`arr[i...] = parent(arr)[map(getindex, decks(arr), i)]
"""
decks(arr, n) = decks(arr)[n]



"""
    unshuffle(arr)

Return an array representing the original unshuffled array.
each axis with [`decks`](@ref).
"""
function unshuffle(arr)
    return arr
end



"""
interesting observation: decks[i] must be a permutation of axes(decks[i]).
"""
struct ShuffledArray{T, N, Decks <: NTuple{<:AbstractVector, N}, IDecks <: NTuple{<:AbstractVector, N}, Arg <: AbstractArray} <: GeneratedArray{T, N}
    decks::Decks
    idecks::IDecks
    arg::Arg
end

ShuffledArray(decks, idecks, arg) = ShuffledArray{eltype(arg), ndims(arg), typeof(decks), typeof(idecks), typeof(arg)}(decks, arg)

Base.parent(arr::ShuffledArray) = arr.arg
WrapperArrays.iswrapper(arr::ShuffledArray) = true
WrapperArrays.adopt(arg, arr::ShuffledArray{T, N, Decks}) where {T, N, Decks} = ShuffledArray(arr.decks, arg)

Base.BroadcastStyle(::Type{ShuffledArray{<:Any, <:Any, <:Any, Arg}}) where {Arg} = ShuffleStyle(BroadcastStyle(Arg))

Base.IndexStyle(arr::ShuffledArray) = IndexCartesian()
Base.IndexStyle(arr::ShuffledArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}) where {T, N} = IndexStyle(arr.arg)
Base.size(arr::ShuffledArray) = size(arr.arg)
Base.size(arr::ShuffledArray, d::Int) = size(arr.arg, d)
Base.axes(arr::ShuffledArray) = axes(arr.arg)
Base.axes(arr::ShuffledArray, d::Int) = axes(arr.arg, d)
function Base.getindex(arr::ShuffledArray{T, N}, i...)::T where {T, N}
    arr.arg[ntuple(n->arr.decks[n][i[n]], Val(ndims(arr)))...]
end
(Base.getindex(arr::ShuffledArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i...)::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::ShuffledArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i::Integer)::T) where {T, N} = arr.arg[i]
function Base.setindex!(arr::ShuffledArray{T, N}, x, i...)::T where {T, N}
    arr.arg[ntuple(n->arr.decks[n][i[n]], Val(ndims(arr)))...] = x
end
(Base.setindex!(arr::ShuffledArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}}, x, i...)::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::ShuffledArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}}, x, i::Integer)::T) where {T, N} = arr.arg[i] = x



similar(bc::Broadcasted{<:ShuffleStyle{S}}) where {S} = similar(convert(Broadcasted{S()}, bc))

struct ShuffleStyle{S<:BroadcastStyle} <: BroadcastStyle end
ShuffleStyle(style::S) where {S <: BroadcastStyle} = ShuffleStyle{S}()
ShuffleStyle(style::S) where {S <: ShuffleStyle} = style
Base.Broadcast.BroadcastStyle(::ShuffleStyle{T}, ::S) where {T, S<:DefaultArrayStyle} = ShuffleStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::ShuffleStyle{T}, ::S) where {T, S<:Style{Tuple}} = ShuffleStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::ShuffleStyle{T}, ::ShuffleStyle{S}) where {T, S<:BroadcastStyle} = ShuffleStyle(result_style(T(), S()))



Base.@propagate_inbounds function Base.copy(src::Broadcasted{<:ShuffleStyle{<:Base.Broadcast.AbstractArrayStyle{0}}})
    decks(src) = () || throw(DimensionMismatchError("TODO"))
    return copy(convert(Broadcasted{S}, parent(reshuffle(src.args[0]))))
end

Base.@propagate_inbounds function Base.copy(src::Broadcasted{<:ShuffleStyle{S}}) where {S <:Base.Broadcast.AbstractArrayStyle}
    src = reshuffle(src)::ShuffledArray
    return ShuffledArray(src.decks, copy(src.arg))
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{ShuffleStyle{S}}) where {S}
    res = ShuffledArray(to_indices(dst, ntuple(n->Colon(), ndims(dst))), dst)
    copyto!(res, src)
    return res.arg
end

Base.@propagate_inbounds function Base.copyto!(dst::ShuffledArray, src::Broadcasted{ShuffleStyle{S}}) where {S}
    axes(dst) == axes(src) || throw(DimensionMismatchError("TODO"))
    src = reshuffle(src)::ShuffledArray
    if dst.decks == src.decks
        copyto!(dst.arg, src.arg)
    else
        copyto!(dst.arg, ShuffledArray(dst.idecks, src))
    end
    return dst
end


@inline decktype(p1::P, p2::P) where {P} = p1
@inline function decktype(p1::SVector{1, <:AbstractUnitRange}, p2::ChunkedUnitRange)
    return ChunkedUnitRange(axistype(p1[1], p2.arg), first(promote(length(p1[1]), p2.chunk)))
end
@inline function decktype(p1::ChunkedUnitRange, p2::SVector{1, <:AbstractUnitRange})
    return ChunkedUnitRange(axistype(p1.arg, p2[1]), first(promote(p1.chunk, length(p2[1]))))
end
@inline function decktype(p1::Vector{<:AbstractUnitRange}, p2::ChunkedUnitRange)
    return Vector{UnitRange{Int}}(p1)
end
@inline function decktype(p1::ChunkedUnitRange, p2::Vector{<:AbstractUnitRange})
    return Vector{UnitRange{Int}}(p1)
end
@inline function decktype(p1, p2)
    return Vector{Vector{Int}}(p1)
end

@inline decks(arg::Broadcasted) = combine_decks(arg.args...)

@inline function combine_decks(args...)
    combineables = map(arg -> map(tuple, decks(arg), keeps(arg)), args)
    results = combinetuple(result_deck, combineables...)
    map(first,results)
end

function result_deck((p1, k1), (p2, k2))
    if !kept(k2)
        return (decktype(p1, p2), k1 | k2)
    elseif !kept(k1)
        return (decktype(p2, p1), k1 | k2)
    else
        p1 == p2 || throw(ArgumentError("Conflicting decks declared"))
        return (decktype(p1, p2), k1 | k2)
    end
end
@inline result_deck((p1, k1)::Tuple{Any, StableKeep}, (p2, k2)::Tuple{Any, Extrude}) = (p1, StableKeep())
@inline result_deck((p1, k1)::Tuple{Any, Extrude}, (p2, k2)::Tuple{Any, StableKeep}) = (p2, StableKeep())
@inline function result_deck((p1, k1)::Tuple{Any, StableKeep}, (p2, k2)::Tuple{Any, StableKeep})
    p1 == p2 || throw(ArgumentError("Conflicting decks declared"))
    (p1, StableKeep())
end

@inline function reshuffle(src::Broadcasted)
    if shuffles don't match, materialize shuffles
    return broadcasted(broadcasted, Ref(src.f), map(reshuffle, src.args)...)
end



@inline Swizzles.childstyle(Arr::Type{<:SwizzledArray}, ::ShuffleStyle{S}) where {S} = ShuffleStyle(Swizzles.childstyle(Arr, S()))

function decks(arr::SwizzledArray)
    arg_decks = decks(arr.arg)
    arr_decks = masktuple(d->[Base.OneTo(1)], d->arg_decks[d], Val(mask(arr)))
    arg_keeps = keeps(arr.arg)
    arr_keeps = masktuple(d->Extrude(), d->arg_keeps[d], Val(mask(arr)))
    init_decks = decks(arr.init)
    init_keeps = keeps(arr.init)
    return map(first, combinetuple(result_deck, map(tuple, arr_decks, arr_keeps), map(tuple, init_decks, init_keeps)))
end

@inline function reshuffle(arr::SwizzledArray{T}) where {T}
    if deck is dropped, don't worry about shuffling it
    return Swizzle(Swizzle(arr.op, mask(arr)), mask(arr))(reshuffle(arr.init), reshuffle(arr.arg))
end



decks(arr::ArrayifiedArray) = decks(arr.arg)

reshuffle(arr::ArrayifiedArray) = broadcasted(ArrayifiedArray, reshuffle(arr.arg))



decks(arr::ExtrudedArray) = decks(arr.arg)

reshuffle(arr::ExtrudedArray) = reshuffle(arr.arg) #TODO this is sad



@inline function parse_specs_decks(arr, specs::Tuple{Vararg{Any, M}}) where {M}
    if @generated
        return :(($(ntuple(n -> n <= M ? :(parse_specs_decks(arr, $n, specs[$n])) : :([axes(arr, $n)]), ndims(arr))...),))
    else
        return ntuple(n -> n <= M ? parse_specs_decks(arr, n, specs[n]) : [axes(arr, n)], Val(ndims(arr)))
    end
end
@inline parse_specs_decks(arr, n, spec::Integer) = ChunkedUnitRange(1:size(arr, n), spec)
@inline parse_specs_decks(arr, n, spec) = spec



struct Shuffle{Decks<:Tuple} <: Swizzles.Intercept
    decks::Decks
end

@inline (shf::Shuffle)(arg) = ShuffledArray(shf.decks, arrayify(arg))

end
