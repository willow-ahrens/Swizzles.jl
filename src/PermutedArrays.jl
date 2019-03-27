module PermutedArrays

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

export PermuteStyle, perms
export Permute

"""
    perms(arr::PermutedArray)

Return the permutations of the axes of `parent(arr)`, such that
`arr[i...] = parent(arr)[map(getindex, perms(arr), i)]
"""
function perms(arr)
    return map(Base.OneTo, size(arr))
end

"""
    perms(arr, n)

Return the permutation of the axis `n` of `parent(arr)`, such that
`arr[i...] = parent(arr)[map(getindex, perms(arr), i)]
"""
perms(arr, n) = perms(arr)[n]



"""
    unPermute(arr)

Return an array representing the original unPermuted array.
each axis with [`perms`](@ref).
"""
function unPermute(arr)
    return arr
end



"""
interesting observation: perms[i] must be a permutation of axes(perms[i]).
"""
struct PermutedArray{T, N, perms <: NTuple{<:AbstractVector, N}, Iperms <: NTuple{<:AbstractVector, N}, Arg <: AbstractArray} <: GeneratedArray{T, N}
    perms::perms
    iperms::Iperms
    arg::Arg
end

PermutedArray(perms, iperms, arg) = PermutedArray{eltype(arg), ndims(arg), typeof(perms), typeof(iperms), typeof(arg)}(perms, arg)

Base.parent(arr::PermutedArray) = arr.arg
WrapperArrays.iswrapper(arr::PermutedArray) = true
WrapperArrays.adopt(arg, arr::PermutedArray{T, N, perms}) where {T, N, perms} = PermutedArray(arr.perms, arg)

Base.BroadcastStyle(::Type{PermutedArray{<:Any, <:Any, <:Any, Arg}}) where {Arg} = PermuteStyle(BroadcastStyle(Arg))

Base.IndexStyle(arr::PermutedArray) = IndexCartesian()
Base.IndexStyle(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}) where {T, N} = IndexStyle(arr.arg)
Base.size(arr::PermutedArray) = size(arr.arg)
Base.size(arr::PermutedArray, d::Int) = size(arr.arg, d)
Base.axes(arr::PermutedArray) = axes(arr.arg)
Base.axes(arr::PermutedArray, d::Int) = axes(arr.arg, d)
function Base.getindex(arr::PermutedArray{T, N}, i...)::T where {T, N}
    arr.arg[ntuple(n->arr.perms[n][i[n]], Val(ndims(arr)))...]
end
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i...)::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i::Integer)::T) where {T, N} = arr.arg[i]
function Base.setindex!(arr::PermutedArray{T, N}, x, i...)::T where {T, N}
    arr.arg[ntuple(n->arr.perms[n][i[n]], Val(ndims(arr)))...] = x
end
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}}, x, i...)::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}}, x, i::Integer)::T) where {T, N} = arr.arg[i] = x



similar(bc::Broadcasted{<:PermuteStyle{S}}) where {S} = similar(convert(Broadcasted{S()}, bc))

struct PermuteStyle{S<:BroadcastStyle} <: BroadcastStyle end
PermuteStyle(style::S) where {S <: BroadcastStyle} = PermuteStyle{S}()
PermuteStyle(style::S) where {S <: PermuteStyle} = style
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::S) where {T, S<:DefaultArrayStyle} = PermuteStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::S) where {T, S<:Style{Tuple}} = PermuteStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::PermuteStyle{S}) where {T, S<:BroadcastStyle} = PermuteStyle(result_style(T(), S()))



Base.@propagate_inbounds function Base.copy(src::Broadcasted{<:PermuteStyle{<:Base.Broadcast.AbstractArrayStyle{0}}})
    perms(src) = () || throw(DimensionMismatchError("TODO"))
    return copy(convert(Broadcasted{S}, parent(rePermute(src.args[0]))))
end

Base.@propagate_inbounds function Base.copy(src::Broadcasted{<:PermuteStyle{S}}) where {S <:Base.Broadcast.AbstractArrayStyle}
    src = rePermute(src)::PermutedArray
    return PermutedArray(src.perms, copy(src.arg))
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{PermuteStyle{S}}) where {S}
    res = PermutedArray(to_indices(dst, ntuple(n->Colon(), ndims(dst))), dst)
    copyto!(res, src)
    return res.arg
end

Base.@propagate_inbounds function Base.copyto!(dst::PermutedArray, src::Broadcasted{PermuteStyle{S}}) where {S}
    axes(dst) == axes(src) || throw(DimensionMismatchError("TODO"))
    src = rePermute(src)::PermutedArray
    if dst.perms == src.perms
        copyto!(dst.arg, src.arg)
    else
        copyto!(dst.arg, PermutedArray(dst.iperms, src))
    end
    return dst
end


@inline permtype(p1::P, p2::P) where {P} = p1
@inline function permtype(p1::SVector{1, <:AbstractUnitRange}, p2::ChunkedUnitRange)
    return ChunkedUnitRange(axistype(p1[1], p2.arg), first(promote(length(p1[1]), p2.chunk)))
end
@inline function permtype(p1::ChunkedUnitRange, p2::SVector{1, <:AbstractUnitRange})
    return ChunkedUnitRange(axistype(p1.arg, p2[1]), first(promote(p1.chunk, length(p2[1]))))
end
@inline function permtype(p1::Vector{<:AbstractUnitRange}, p2::ChunkedUnitRange)
    return Vector{UnitRange{Int}}(p1)
end
@inline function permtype(p1::ChunkedUnitRange, p2::Vector{<:AbstractUnitRange})
    return Vector{UnitRange{Int}}(p1)
end
@inline function permtype(p1, p2)
    return Vector{Vector{Int}}(p1)
end

@inline perms(arg::Broadcasted) = combine_perms(arg.args...)

@inline function combine_perms(args...)
    combineables = map(arg -> map(tuple, perms(arg), keeps(arg)), args)
    results = combinetuple(result_perm, combineables...)
    map(first,results)
end

function result_perm((p1, k1), (p2, k2))
    if !kept(k2)
        return (permtype(p1, p2), k1 | k2)
    elseif !kept(k1)
        return (permtype(p2, p1), k1 | k2)
    else
        p1 == p2 || throw(ArgumentError("Conflicting perms declared"))
        return (permtype(p1, p2), k1 | k2)
    end
end
@inline result_perm((p1, k1)::Tuple{Any, StableKeep}, (p2, k2)::Tuple{Any, Extrude}) = (p1, StableKeep())
@inline result_perm((p1, k1)::Tuple{Any, Extrude}, (p2, k2)::Tuple{Any, StableKeep}) = (p2, StableKeep())
@inline function result_perm((p1, k1)::Tuple{Any, StableKeep}, (p2, k2)::Tuple{Any, StableKeep})
    p1 == p2 || throw(ArgumentError("Conflicting perms declared"))
    (p1, StableKeep())
end

@inline function rePermute(src::Broadcasted)
    if Permutes don't match, materialize Permutes
    return broadcasted(broadcasted, Ref(src.f), map(rePermute, src.args)...)
end



@inline Swizzles.childstyle(Arr::Type{<:SwizzledArray}, ::PermuteStyle{S}) where {S} = PermuteStyle(Swizzles.childstyle(Arr, S()))

function perms(arr::SwizzledArray)
    arg_perms = perms(arr.arg)
    arr_perms = masktuple(d->[Base.OneTo(1)], d->arg_perms[d], Val(mask(arr)))
    arg_keeps = keeps(arr.arg)
    arr_keeps = masktuple(d->Extrude(), d->arg_keeps[d], Val(mask(arr)))
    init_perms = perms(arr.init)
    init_keeps = keeps(arr.init)
    return map(first, combinetuple(result_perm, map(tuple, arr_perms, arr_keeps), map(tuple, init_perms, init_keeps)))
end

@inline function rePermute(arr::SwizzledArray{T}) where {T}
    if perm is dropped, don't worry about shuffling it
    return Swizzle(Swizzle(arr.op, mask(arr)), mask(arr))(rePermute(arr.init), rePermute(arr.arg))
end



perms(arr::ArrayifiedArray) = perms(arr.arg)

rePermute(arr::ArrayifiedArray) = broadcasted(ArrayifiedArray, rePermute(arr.arg))



perms(arr::ExtrudedArray) = perms(arr.arg)

rePermute(arr::ExtrudedArray) = rePermute(arr.arg) #TODO this is sad



@inline function parse_specs_perms(arr, specs::Tuple{Vararg{Any, M}}) where {M}
    if @generated
        return :(($(ntuple(n -> n <= M ? :(parse_specs_perms(arr, $n, specs[$n])) : :([axes(arr, $n)]), ndims(arr))...),))
    else
        return ntuple(n -> n <= M ? parse_specs_perms(arr, n, specs[n]) : [axes(arr, n)], Val(ndims(arr)))
    end
end
@inline parse_specs_perms(arr, n, spec::Integer) = ChunkedUnitRange(1:size(arr, n), spec)
@inline parse_specs_perms(arr, n, spec) = spec



struct Permute{perms<:Tuple} <: Swizzles.Intercept
    perms::perms
end

@inline (shf::Permute)(arg) = PermutedArray(shf.perms, arrayify(arg))

end
