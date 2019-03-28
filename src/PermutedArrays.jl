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

export PermuteStyle, Permute

"""
    PermutationMismatch([msg])

The objects called do not have matching permutations. Optional argument `msg` is
a descriptive error string.
"""
struct PermutationMismatch <: Exception
    msg::AbstractString
end
PermutationMismatch() = PermutationMismatch("")

"""
permute an array so that
`arr[i...] = parent(arr)[map(getindex, perms(arr), i)]
interesting observation: perms[i] must be a permutation of axes(perms[i]).
"""
struct PermutedArray{T, N, Perms <: NTuple{<:AbstractVector, N}, InvPerms <: NTuple{<:AbstractVector, N}, Arg <: AbstractArray} <: GeneratedArray{T, N}
    perms::Perms
    invperms::InvPerms
    arg::Arg
end

PermutedArray(perms, iperms, arg) = PermutedArray{eltype(arg), ndims(arg), typeof(perms), typeof(iperms), typeof(arg)}(perms, arg)

Base.parent(arr::PermutedArray) = arr.arg
WrapperArrays.iswrapper(arr::PermutedArray) = true
WrapperArrays.adopt(arg, arr::PermutedArray{T, N, perms}) where {T, N, perms} = PermutedArray(arr.perms, arg)

Base.BroadcastStyle(::Type{PermutedArray{<:Any, <:Any, <:Any, Arg}}) where {Arg} = PermuteStyle(BroadcastStyle(Arg))

Base.IndexStyle(arr::PermutedArray) = IndexCartesian()
Base.IndexStyle(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}) where {T, N} = IndexStyle(arr.arg)
Base.IndexStyle(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}) where {T, N} = IndexStyle(arr.arg)
Base.size(arr::PermutedArray) = size(arr.arg)
Base.size(arr::PermutedArray, d::Int) = size(arr.arg, d)
Base.axes(arr::PermutedArray) = axes(arr.arg)
Base.axes(arr::PermutedArray, d::Int) = axes(arr.arg, d)
function Base.getindex(arr::PermutedArray{T, N}, i...)::T where {T, N}
    arr.arg[ntuple(n->arr.perms[n][i[n]], Val(ndims(arr)))...]
end
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i...)::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, i...)::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i::Integer)::T) where {T, N} = arr.arg[i]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, i::Integer)::T) where {T, N} = arr.arg[i]
function Base.setindex!(arr::PermutedArray{T, N}, x, i...)::T where {T, N}
    arr.arg[ntuple(n->arr.perms[n][i[n]], Val(ndims(arr)))...] = x
end
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}}, x, i...)::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}}, x, i...)::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}}, x, i::Integer)::T) where {T, N} = arr.arg[i] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}}, x, i::Integer)::T) where {T, N} = arr.arg[i] = x



#similar(bc::Broadcasted{<:PermuteStyle{S}}) where {S} = similar(convert(Broadcasted{S()}, bc))

struct PermuteStyle{S<:BroadcastStyle} <: BroadcastStyle end
PermuteStyle(style::S) where {S <: BroadcastStyle} = PermuteStyle{S}()
PermuteStyle(style::S) where {S <: PermuteStyle} = style
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::S) where {T, S<:DefaultArrayStyle} = PermuteStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::S) where {T, S<:Style{Tuple}} = PermuteStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::PermuteStyle{S}) where {T, S<:BroadcastStyle} = PermuteStyle(result_style(T(), S()))



Base.@propagate_inbounds function Base.copy(src::Broadcasted{<:PermuteStyle{<:Base.Broadcast.AbstractArrayStyle{0}}})
    perms(src) = () || throw(DimensionMismatchError("TODO"))
    return copy(convert(Broadcasted{S}, parent(repermute(src.args[0]))))
end

Base.@propagate_inbounds function Base.copy(src::Broadcasted{PermuteStyle{S}}) where {S <:Base.Broadcast.AbstractArrayStyle}
    try
        src = repermute(src)
        return PermutedArray(src.perms, copy(src.arg))
    catch PermutationMismatch
        return PermutedArray(src.perms, copy(src.arg))
    end
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{PermuteStyle{S}}) where {S}
    res = PermutedArray(to_indices(dst, ntuple(n->Colon(), ndims(dst))), dst)
    copyto!(res, src)
    return res.arg
end

Base.@propagate_inbounds function Base.copyto!(dst::PermutedArray, src::Broadcasted{PermuteStyle{S}}) where {S}
    axes(dst) == axes(src) || throw(DimensionMismatchError("TODO"))
    src = repermute(src)::PermutedArray
    if dst.perms == src.perms
        copyto!(dst.arg, src.arg)
    else
        copyto!(dst.arg, PermutedArray(dst.iperms, src))
    end
    return dst
end

@inline permtype(p1::P, p2::P) where {P} = p1
@inline permtype(p1::AbstractVector{Int}, p2::AbstractVector{Int}) = Array{Int}(p1)
@inline permtype(p1::Base.OneTo, p2::Base.OneTo) = Base.OneTo(convert(promote_type(eltype(p1), eltype(p2)), p1[end]))

@inline function repermute(src::Broadcasted{S}) where {S}
    args = map(repermute(arg), src.args)
    perms, iperms = combine_perms(success, args...)
    args = map(parent(args))
    return ShuffledArray(perms, iperms, ArrayfiedArray(Broadcasted{S}(src.f, axes=src.axes)

@inline function combine_perms(success, args::ShuffledArray...)
    combineables = map(arg->map(tuple, arg.perms, keeps(arg)), args)
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
@inline result_perm((p1, k1)::Tuple{Any, Keep}, (p2, k2)::Tuple{Any, Extrude}) = (p1, Keep())
@inline result_perm((p1, k1)::Tuple{Any, Extrude}, (p2, k2)::Tuple{Any, Keep}) = (p2, Keep())
@inline function result_perm((p1, k1)::Tuple{Any, Keep}, (p2, k2)::Tuple{Any, Keep})
    p1 == p2 || throw(ArgumentError("Conflicting perms declared"))
    (p1, Keep())
end

@inline function repermute(src::Broadcasted)
    return Ref(src.f), map(repermute, src.args)...)
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

@inline function repermute(arr::SwizzledArray{T}) where {T}
    if perm is dropped, don't worry about shuffling it
    return Swizzle(Swizzle(arr.op, mask(arr)), mask(arr))(repermute(arr.init), repermute(arr.arg))
end



perms(arr::ArrayifiedArray) = perms(arr.arg)

repermute(arr::ArrayifiedArray) = broadcasted(ArrayifiedArray, repermute(arr.arg))



perms(arr::ExtrudedArray) = perms(arr.arg)

repermute(arr::ExtrudedArray) = repermute(arr.arg) #TODO this is sad



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
