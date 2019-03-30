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
    isperm(v::AbstractVector) -> Bool

Return `true` if `v` is a valid permutation of `axes(v, 1)`.
"""
Base.isperm(v::AbstractVector) = Base.isperm(v, axes(v, 1))

"""
    isperm(v, u) -> Bool

Return `true` if `v` is a valid permutation of `u`.
"""
function Base.isperm(v, u)
    return sort(v) == sort(u)
end
function Base.isperm(v::UnitRange{T}, u::UnitRange{T}) where {T<:Integer}
    return v == u
end
function Base.isperm(v::UnitRange{T}, u::AbstractVector{T}) where {T<:Integer}
    return isperm(u, v)
end
function Base.isperm(v::AbstractVector{T}, u::UnitRange{T}) where {T<:Integer}
    n = length(u)
    used = falses(n)
    for x in v
        i = x - u.start
        (0 < x <= n) && (used[x] ⊻= true) || return false
    end
    true
end
function Base.isperm(v::Base.OneTo{T}, u::Base.OneTo{T}) where {T<:Integer}
    return v == u
end
function Base.isperm(v::Base.OneTo{T}, u::AbstractVector{T}) where {T<:Integer}
    return isperm(u, v)
end
function Base.isperm(v::AbstractVector{T}, u::Base.OneTo{T}) where {T<:Integer}
    n = length(u)
    used = falses(n)
    for x in v
        i = x - 1
        (0 < x <= n) && (used[x] ⊻= true) || return false
    end
    true
end



Base.invperm(v::UnitRange{<:Integer}) = v
Base.invperm(v::Base.OneTo{<:Integer}) = v



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
    return copy(convert(Broadcasted{S}, repermute(src.args[1])))
end

Base.@propagate_inbounds function Base.copy(src::Broadcasted{PermuteStyle{S}}) where {S <:Base.Broadcast.AbstractArrayStyle}
    src = repermute(src.args[1])
    return PermutedArray(src.perms, copy(src.arg))
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{PermuteStyle{S}}) where {S}
    res = PermutedArray(to_indices(dst, ntuple(n->Colon(), ndims(dst))), dst)
    copyto!(res, src)
    return res.arg
end

Base.@propagate_inbounds function Base.copyto!(dst::PermutedArray, src::Broadcasted{PermuteStyle{S}}) where {S}
    src = repermute(src.args[1])
    dst.perms == src.perms || PermutationMismatch("TODO")
    copyto!(dst, src)
    return dst
end



@inline function repermute(src::Broadcasted{S}) where {S}
    args = map(repermute(arg), src.args)
    combineables = map(arg->ziptuple(arg.perms, arg.iperms, keeps(arg)), args)
    (perms, iperms, _) = ziptuple(combinetuple(result_perm, combineables...)...)
    args = map(parent(args))
    return ShuffledArray(perms, iperms, ArrayfiedArray(Broadcasted{S}(src.f, axes=src.axes)))
end


function result_perm((p1, ip1, k1), (p2, ip2, k2))
    if !kept(k2)
        return (permtype(p1, p2), permtype(ip1, ip2), k1 | k2)
    elseif !kept(k1)
        return (permtype(p2, p1), permtype(ip2, ip1), k1 | k2)
    else
        p1 == p2 || throw(ArgumentError("Conflicting perms declared"))
        return (permtype(p1, p2), permtype(ip1, ip2), k1 | k2)
    end
end
@inline result_perm((p1, ip1, k1)::Tuple{Any, Keep}, (p2, ip2, k2)::Tuple{Any, Extrude}) = (p1, ip1, Keep())
@inline result_perm((p1, ip1, k1)::Tuple{Any, Extrude}, (p2, ip2, k2)::Tuple{Any, Keep}) = (p2, ip2, Keep())
@inline function result_perm((p1, ip1, k1)::Tuple{Any, Keep}, (p2, ip2, k2)::Tuple{Any, Keep})
    p1 == p2 || throw(ArgumentError("Conflicting perms declared"))
    (p1, ip1, Keep())
end

@inline permtype(p1::P, p2::P) where {P} = p1
@inline permtype(p1::AbstractVector{Int}, p2::AbstractVector{Int}) = Vector{Int}(p1)
@inline permtype(p1::Base.OneTo, p2::Base.OneTo) = Base.OneTo(convert(promote_type(eltype(p1), eltype(p2)), p1[end]))



@inline Swizzles.childstyle(Arr::Type{<:SwizzledArray}, ::PermuteStyle{S}) where {S} = PermuteStyle(Swizzles.childstyle(Arr, S()))

@inline function repermute(arr::SwizzledArray{T}) where {T}
    arg = repermute(arr.arg)
    init = repermute(arr.init)
    arg_perms = arr.arg.perms
    arr_perms = masktuple(d->[Base.OneTo(1)], d->arg_perms[d], Val(mask(arr)))
    arg_iperms = arr.arg.iperms
    arr_iperms = masktuple(d->[Base.OneTo(1)], d->arg_iperms[d], Val(mask(arr)))
    arg_keeps = keeps(arr.arg)
    arr_keeps = masktuple(d->Extrude(), d->arg_keeps[d], Val(mask(arr)))
    init_perms = arr.init.perms
    init_keeps = keeps(arr.init)
    (perms, iperms, _) = ziptuple(combinetuple(result_perm, ziptuple(arr_perms, arr_iperms, arr_keeps), ziptuple(init_perms, init_iperms, init_keeps))...)
    return PermutedArray(perms, iperms, Swizzle(arr.op, arr.mask...)(parent(arg), parent(init)))
end



function repermute(arr::ArrayifiedArray)
    arg = repermute(arr.arg)
    if parent(arg) isa ArrayifiedArray
        return PermutedArray(arg.perms, arg.iperms, parent(arg))
    else
        return PermutedArray(arg.perms, arg.iperms, ArrayifiedArray(parent(arg)))
    end
end



struct Permute{_Perms, _InvPerms} <: Swizzles.Intercept
    _perms::Perms
    _invperms::InvPerms
end

Permute(_perms...) = Permute(_perms, map(invperm, _perms))

@inline function(ctr::Permute{<:Tuple{Vararg{Any, _N}}, <:Tuple{Vararg{Any, _N}}})(arg::Arg) where {_N, Arg <: AbstractArray}
    if @generated
        perms = ntuple(n -> n <= _N ? :(ctr._perms[$n]) : :(axes(arg, $n)), ndims(arg))
        invperms = ntuple(n -> n <= _N ? :(ctr._invperms[$n]) : :(axes(arg, $n)), ndims(arg))
        return :(return PermutedArray(($(perms...),), ($(invperms...),), arg))
    else
        perms = ntuple(n -> n <= _N ? ctr._perms[n] : axes(arg, n), ndims(arg))
        invperms = ntuple(n -> n <= _N ? ctr._invperms[n] : axes(arg, n), ndims(arg))
        return PermutedArray(perms, invperms, arg)
    end
end

end
