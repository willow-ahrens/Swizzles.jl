module PermutedArrays

using StaticArrays
using Swizzles.WrapperArrays
using Swizzles.ShallowArrays
using Swizzles.ArrayifiedArrays
using Swizzles.GeneratedArrays
using Swizzles.ExtrudedArrays
using Swizzles
using Swizzles: SwizzledArray
using Swizzles: mask, jointuple, combinetuple, masktuple, ziptuple, nziptuple

using Base: Slice
using Base.Broadcast: BroadcastStyle, Broadcasted, Style, AbstractArrayStyle, DefaultArrayStyle
using Base.Broadcast: broadcast_shape, check_broadcast_shape, result_style, axistype, broadcasted
using Base.Iterators: repeated, flatten, take

export PermuteStyle, Permute, PermutationMismatch

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
struct PermutedArray{T, N, Perms <: Tuple{Vararg{AbstractVector, N}}, IPerms <: Tuple{Vararg{AbstractVector, N}}, Arg <: AbstractArray} <: GeneratedArray{T, N}
    perms::Perms
    iperms::IPerms
    arg::Arg
end

PermutedArray(perms, iperms, arg) = PermutedArray{eltype(arg), ndims(arg), typeof(perms), typeof(iperms), typeof(arg)}(perms, iperms, arg)

Base.parent(arr::PermutedArray) = arr.arg
WrapperArrays.iswrapper(arr::PermutedArray) = true
WrapperArrays.adopt(arg, arr::PermutedArray) = PermutedArray(arr.perms, arr.iperms, arg)

Base.Broadcast.BroadcastStyle(::Type{<:PermutedArray{<:Any, <:Any, <:Any, <:Any, Arg}}) where {Arg} = PermuteStyle(BroadcastStyle(Arg))

Base.IndexStyle(arr::PermutedArray) = IndexCartesian()
Base.IndexStyle(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}) where {T, N} = IndexStyle(arr.arg)
Base.IndexStyle(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}) where {T, N} = IndexStyle(arr.arg)
Base.size(arr::PermutedArray) = size(arr.arg)
Base.size(arr::PermutedArray, d::Int) = size(arr.arg, d)
Base.axes(arr::PermutedArray) = axes(arr.arg)
Base.axes(arr::PermutedArray, d::Int) = axes(arr.arg, d)
function Base.getindex(arr::PermutedArray{T, N}, i::Vararg{Any, N})::T where {T, N}
    arr.arg[ntuple(n->arr.perms[n][i[n]], Val(N))...]
end
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i...)::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, i...)::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i::Vararg{Any, N})::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, i::Vararg{Any, N})::T) where {T, N} = arr.arg[i...]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, i::Integer)::T) where {T, N} = arr.arg[i]
(Base.getindex(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, i::Integer)::T) where {T, N} = arr.arg[i]
function Base.setindex!(arr::PermutedArray{T, N}, x, i::Vararg{Any, N})::T where {T, N}
    arr.arg[ntuple(n->arr.iperms[n][i[n]], Val(N))...] = x
end
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, x, i::Vararg{Any, N})::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, x, i::Vararg{Any, N})::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, x, i...)::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, x, i...)::T) where {T, N} = arr.arg[i...] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.OneTo, N}}}, x, i::Integer)::T) where {T, N} = arr.arg[i] = x
(Base.setindex!(arr::PermutedArray{T, N, <:Tuple{Vararg{<:Base.Slice{<:Base.OneTo}, N}}}, x, i::Integer)::T) where {T, N} = arr.arg[i] = x



#similar(bc::Broadcasted{<:PermuteStyle{S}}) where {S} = similar(convert(Broadcasted{S()}, bc))

struct PermuteStyle{S<:BroadcastStyle} <: BroadcastStyle end
PermuteStyle(style::S) where {S <: BroadcastStyle} = PermuteStyle{S}()
PermuteStyle(style::S) where {S <: PermuteStyle} = style
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::S) where {T, S<:AbstractArrayStyle} = PermuteStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::S) where {T, S<:DefaultArrayStyle} = PermuteStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::S) where {T, S<:Style{Tuple}} = PermuteStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::PermuteStyle{T}, ::PermuteStyle{S}) where {T, S<:BroadcastStyle} = PermuteStyle(result_style(T(), S()))



Base.@propagate_inbounds function Base.copy(src::Broadcasted{PermuteStyle{S}}) where {S<:Base.Broadcast.AbstractArrayStyle{0}}
    return copy(convert(Broadcasted{S}, repermute(src).arg.arg))
end

Base.@propagate_inbounds function Base.copy(src::Broadcasted{PermuteStyle{S}}) where {S <:Base.Broadcast.AbstractArrayStyle}
    src = repermute(src)
    return PermutedArray(src.perms, src.iperms, copy(src.arg))
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{PermuteStyle{S}}) where {S}
    pdst = repermute(dst)
    psrc = repermute(src)
    pdst.perms == psrc.perms || PermutationMismatch("TODO")
    copyto!(pdst.arg, psrc.arg)
    return dst
end



@inline function repermute(src)
    perms = map(Slice, axes(src))
    iperms = map(invperm, perms)
    return PermutedArray(perms, iperms, arrayify(src))
end

@inline function repermute(src::PermutedArray)
    arg = repermute(src.arg)
    return PermutedArray(map(getindex, src.perms, arg.perms), map(getindex, src.iperms, arg.iperms), arg.arg)
end

@inline function repermute(src::Broadcasted{PermuteStyle{S}}) where {S}
    return repermute(convert(Broadcasted{S}, src))
end
@inline function repermute(src::Broadcasted{S}) where {S}
    args = map(repermute, src.args)
    combineables = map(arg->ziptuple(arg.perms, arg.iperms, keeps(arg)), args)
    (perms, iperms, _) = nziptuple(Val(3), combinetuple(result_perm, combineables...)...)
    args = map(parent, args)
    return PermutedArray(perms, iperms, ArrayifiedArray(Broadcasted{S}(src.f, args, src.axes)))
end


function result_perm((p1, ip1, k1), (p2, ip2, k2))
    if !kept(k2)
        return (permtype(p1, p2), permtype(ip1, ip2), k1 | k2)
    elseif !kept(k1)
        return (permtype(p2, p1), permtype(ip2, ip1), k1 | k2)
    else
        p1 == p2 || throw(PermutationMismatch("Conflicting permutations"))
        return (permtype(p1, p2), permtype(ip1, ip2), k1 | k2)
    end
end
@inline result_perm((p1, ip1, k1)::Tuple{Any, Keep}, (p2, ip2, k2)::Tuple{Any, Extrude}) = (p1, ip1, Keep())
@inline result_perm((p1, ip1, k1)::Tuple{Any, Extrude}, (p2, ip2, k2)::Tuple{Any, Keep}) = (p2, ip2, Keep())
@inline function result_perm((p1, ip1, k1)::Tuple{Any, Keep}, (p2, ip2, k2)::Tuple{Any, Keep})
    p1 == p2 || throw(PermutationMismatch("Conflicting permutations"))
    (p1, ip1, Keep())
end

@inline permtype(p1::P, p2::P) where {P} = p1
@inline permtype(p1::AbstractVector{Int}, p2::AbstractVector{Int}) = Vector{Int}(p1)
@inline permtype(p1::Base.OneTo, p2::Base.OneTo) = Base.OneTo(convert(promote_type(eltype(p1), eltype(p2)), p1[end]))



@inline Swizzles.childstyle(Arr::Type{<:SwizzledArray}, ::PermuteStyle{S}) where {S} = PermuteStyle(Swizzles.childstyle(Arr, S()))

function repermute(arr::SwizzledArray{T}) where {T}
    arg = repermute(arr.arg)::PermutedArray
    init = repermute(arr.init)::PermutedArray
    arr_perms = masktuple(d->Base.OneTo(1), d->arg.perms[d], Val(mask(arr)))
    arr_iperms = masktuple(d->Base.OneTo(1), d->arg.iperms[d], Val(mask(arr)))
    arg_keeps = keeps(arg)
    arr_keeps = masktuple(d->Extrude(), d->arg_keeps[d], Val(mask(arr)))
    init_perms = init.perms
    init_iperms = init.iperms
    init_keeps = keeps(init)
    combineables = (ziptuple(arr_perms, arr_iperms, arr_keeps), ziptuple(init_perms, init_iperms, init_keeps))
    (perms, iperms, _) = nziptuple(Val(3), combinetuple(result_perm, combineables...)...)
    return PermutedArray(perms, iperms, Swizzle(arr.op, Val(mask(arr)))(parent(init), parent(arg)))
end



repermute(arr::ArrayifiedArray) = repermute(arr.arg)



struct Permute{_Perms, _IPerms} <: Swizzles.Intercept
    _perms::_Perms
    _iperms::_IPerms
    @inline function Permute(_perms...)
        _iperms = map(invperm, _perms)
        return new{typeof(_perms), typeof(_iperms)}(_perms, _iperms)
    end
end


@inline function(ctr::Permute{<:Tuple{Vararg{Any, _N}}, <:Tuple{Vararg{Any, _N}}})(arg::Arg) where {_N, Arg <: AbstractArray}
    if @generated
        perms = ntuple(n -> n <= _N ? :(ctr._perms[$n]) : :(axes(arg, $n)), ndims(arg))
        iperms = ntuple(n -> n <= _N ? :(ctr._iperms[$n]) : :(axes(arg, $n)), ndims(arg))
        return :(return PermutedArray(($(perms...),), ($(iperms...),), arg))
    else
        perms = ntuple(n -> n <= _N ? ctr._perms[n] : axes(arg, n), ndims(arg))
        iperms = ntuple(n -> n <= _N ? ctr._iperms[n] : axes(arg, n), ndims(arg))
        return PermutedArray(perms, iperms, arg)
    end
end

end
