using Swizzles.Properties
using Swizzles.WrapperArrays
using Swizzles.ArrayifiedArrays
using Swizzles.GeneratedArrays
using Swizzles.ExtrudedArrays
using Base: checkbounds_indices, throw_boundserror, tail, dataids, unaliascopy, unalias
using Base.Iterators: reverse, repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, instantiate, broadcastable, preprocess, _broadcast_getindex, combine_eltypes, broadcast_shape
using Base.FastMath: add_fast, mul_fast, min_fast, max_fast
using StaticArrays



struct SwizzledArray{T, N, Op, mask, Init<:AbstractArray, Arg<:AbstractArray} <: GeneratedArray{T, N}
    op::Op
    init::Init
    arg::Arg
    function SwizzledArray{T, N, Op, mask, Init, Arg}(op::Op, init::Init, arg::Arg) where {T, N, Op, mask, Init, Arg}
        #FIXME check swizzles. also check noop axes!
        @assert T !== nothing
        @assert N == max(0, mask...)
        new(op, init, arg)
    end
end

@inline function SwizzledArray{T, N, Op, mask}(op::Op, init::Init, arg::Arg) where {T, N, Op, mask, Init, Arg}
    SwizzledArray{T, N, Op, mask, Init, Arg}(op, init, arg)
end

#eltype converter

@inline function Base.convert(::Type{SwizzledArray{T}}, arr::SwizzledArray{S, N, Op, mask, Init, Arg}) where {T, S, N, Op, mask, Init, Arg}
    return SwizzledArray{T, N, Op, mask, Init, Arg}(arr.op, arr.init, arr.arg)
end

#eltype bound

@inline function Properties.eltype_bound(arr::SwizzledArray)
    #FIXME not strictly correct
    T = eltype(arr.init)
    S = Properties.eltype_bound(arr.arg)
    if eltype(mask(arr)) <: Int
        return S
    end
    T! = Union{T, Properties.return_type(arr.op, T, S)}
    if T! <: T
        return T!
    end
    T = T!
    T! = Union{T, Properties.return_type(arr.op, T, S)}
    if T! <: T
        return T!
    end
    return Any
end



@inline mask(::Type{<:SwizzledArray{<:Any, <:Any, <:Any, _mask}}) where {_mask} = _mask
@inline mask(::SwizzledArray{<:Any, <:Any, <:Any, _mask}) where {_mask} = _mask



function Base.show(io::IO, arr::SwizzledArray)
    print(io, SwizzledArray)
    print(io, '(', arr.op, ", ", mask(arr), ", ", arr.init, ", ", arr.arg, ')')
    nothing
end

Base.parent(arr::SwizzledArray) = arr.arg
Base.parent(::Type{<:SwizzledArray{T, N, Op, mask, Init, Arg}}) where {T, N, Op, mask, Init, Arg} = Arg
WrapperArrays.iswrapper(arr::SwizzledArray) = true
function WrapperArrays.adopt(arg::Arg, arr::SwizzledArray{T, N, Op, mask, Init}) where {T, N, Op, mask, Init, Arg}
    SwizzledArray{T, N, Op, mask, Init, Arg}(arr.op, arr.init, arg)
end

Base.dataids(arr::SwizzledArray) = (dataids(arr.op), dataids(arr.init), dataids(arr.arg))
function Base.unaliascopy(arr::SwizzledArray{T, N, Op, mask}) where {T, N, Op, mask}
    op = unaliascopy(arr.op)
    init = unaliascopy(arr.init)
    arg = unaliascopy(arr.arg)
    SwizzledArray{T, N, typeof(op), mask, typeof(init), typeof(arg)}(op, init, arg)
end
function Base.unalias(dst, arr::SwizzledArray{T, N, Op, mask}) where {T, N, Op, mask}
    op = unalias(dst, arr.op)
    init = unalias(dst, arr.init)
    arg = unalias(dst, arr.arg)
    SwizzledArray{T, N, typeof(op), mask, typeof(init), typeof(arg)}(op, init, arg)
end

@inline function Base.size(arr::SwizzledArray)
    arg_size = size(arr.arg)
    imasktuple(d->1, d->arg_size[d], Val(mask(arr)))
end

@inline function Base.axes(arr::SwizzledArray)
    arg_axes = axes(arr.arg)
    imasktuple(d->Base.OneTo(1), d->arg_axes[d], Val(mask(arr)))
end

Base.@propagate_inbounds function Base.copy(src::Broadcasted{DefaultArrayStyle{0}, <:Any, typeof(identity), <:Tuple{SubArray{T, <:Any, <:SwizzledArray{T, N}, <:Tuple{Vararg{Any, N}}}}}) where {T, N}
    return Base.copy(Broadcasted{DefaultArrayStyle{0}}(identity, (convert(SwizzledArray, src.args[1]),)))
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{Nothing, <:Any, typeof(identity), <:Tuple{SubArray{T, <:Any, <:SwizzledArray{T, N}, <:Tuple{Vararg{Any, N}}}}}) where {T, N}
    #A view of a Swizzle can be computed as a swizzle of a view (hiding the
    #complexity of dropping view indices). Therefore, we convert first.
    return Base.copyto!(dst, convert(SwizzledArray, src.args[1]))
end

Base.@propagate_inbounds function Base.convert(::Type{SwizzledArray}, src::SubArray{T, M, Arr, <:Tuple{Vararg{Any, N}}}) where {T, N, M, Op, Arr <: SwizzledArray{T, N, Op}}
    arr = parent(src)
    inds = parentindices(src)
    arg = arr.arg
    init = arr.init
    if M == 0
        mask′ = _convert_dropmask(mask(arr)...)
    else
        mask′ = _convert_remask(inds, mask(arr)...)
    end
    init′ = SubArray(init, ntuple(n -> (Base.@_inline_meta; size(init, n) == 1 ? firstindex(init, n) : inds′[n]), Val(ndims(init))))
    arg′ = SubArray(arg, parentindex(arr, inds...))
    return SwizzledArray{eltype(src), M, Op, mask′}(arr.op, init′, arg′)
end

@inline _convert_dropmask(::Drop, mask...) = (drop, _convert_dropmask(mask...)...)
@inline _convert_dropmask(::Any, mask...) = _convert_dropmask(mask...)
@inline _convert_dropmask() = ()

@inline function _convert_remask(indices, d, mask...)
    if d isa Drop
        (drop, _convert_remask(indices, mask...)...)
    elseif Base.index_dimsum(indices[d]) isa Tuple{}
        _convert_remask(indices, mask...)
    else
        (length(Base.index_dimsum(indices[1:d])), _convert_remask(indices, mask...)...)
    end
end
@inline _convert_remask(indices) = ()

#Ideally, we would have written this.
#=
Base.@propagate_inbounds function Base.copy(src::Broadcasted{DefaultArrayStyle{0}, <:Any, typeof(identity), <:Tuple{Arr}}) where {Arr <: SwizzledArray}
    arr = src.args[1]
    dst = Array{eltype(arr), 0}(undef)
    copyto!(dst, Broadcasted{Nothing}(identity, (arr,))) #TRACE
    return dst[]
end
=#

#Instead, we write this:
Base.@propagate_inbounds function Base.copy(src::Broadcasted{DefaultArrayStyle{0}, <:Any, typeof(identity), <:Tuple{Arr}}) where {Arr <: SwizzledArray}
    arr = src.args[1]
    arg = arr.arg
    #FIXME we need a Property here
    if mask(arr) isa Tuple{Vararg{Int}} && eltype(arr.init) <: Nothing && arr.op isa Guard
        if length(arg) > 0
            return arg[]
        else
            return arr.init[]
        end
    else
        dst = arr.init[]
        arg = ArrayifiedArrays.preprocess(dst, arr.arg) #FIXME if dst isn't an array, does this even make sense?
        @inbounds for i in eachindex(arg)
            dst = arr.op(dst, arg[i])
        end
    end
    return dst
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{Nothing, <:Any, typeof(identity), <:Tuple{SwizzledArray}})
    #This method gets called when the destination eltype is unsuitable for
    #accumulating the swizzle. Therefore, we should allocate a suitable
    #destination and then accumulate.
    arr = src.args[1]
    arr′ = copyto!(similar(arr), arr)
    @assert ndims(dst) == ndims(arr′)
    copyto!(dst, arr′)
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray{T, N}, src::Broadcasted{Nothing, <:Any, typeof(identity), Tuple{Arr}}) where {T, N, Arr <: SwizzledArray{<:T, N}}
    arr = src.args[1]
    arg = arr.arg
    arg = ArrayifiedArrays.preprocess(dst, arr.arg)
    #FIXME we need a Property here
    if mask(arr) isa Tuple{Vararg{Int}} && eltype(arr.init) <: Nothing && arr.op isa Guard
        if length(arg) == 0
            dst .= arr.init
        else
            @inbounds for i in eachindex(arg)
                i′ = childindex(dst, arr, i)
                dst[i′...] = arg[i]
            end
        end
    else
        dst .= arr.init
        @inbounds for i in eachindex(arg)
            i′ = childindex(dst, arr, i)
            dst[i′...] = arr.op(dst[i′...], arg[i])
        end
    end
    return dst
end

Base.@propagate_inbounds function parentindex(arr::SubArray, i...)
    return Base.reindex(arr, Base.parentindices(arr), i)
end

Base.@propagate_inbounds function parentindex(arr::SwizzledArray, i::Integer)
    return parentindex(arr, CartesianIndices(arr)[i])
end

Base.@propagate_inbounds function parentindex(arr::SwizzledArray, i::CartesianIndex)
    return parentindex(arr, Tuple(i)...)
end

Base.@propagate_inbounds function parentindex(arr::SwizzledArray{<:Any, 1}, i::Integer)
    return invoke(parentindex, Tuple{typeof(arr), Any}, arr, i)
end

Base.@propagate_inbounds function parentindex(arr::SwizzledArray{<:Any, N}, i::Vararg{Any, N}) where {N}
    arg_axes = axes(arr.arg)
    masktuple(d->Base.Slice(arg_axes[d]), d->i[d], Val(mask(arr)))
end


"""
   parentindex(arr, i)

   For all wrapper arrays arr such that `arr` involves an index remapping,
   return the indices into `parent(arr)` which affect the indices `i` of `arr`.

   See also: [`swizzle`](@ref).
"""
parentindex

Base.@propagate_inbounds function childindex(dst::AbstractArray{<:Any, N}, arr::SwizzledArray{<:Any, N}, i::Integer) where {N}
    return childindex(dst, arr, CartesianIndices(arr.arg)[i])
end

Base.@propagate_inbounds function childindex(dst::AbstractArray{<:Any, N}, arr::SwizzledArray{<:Any, N}, i::CartesianIndex) where {N}
    return childindex(dst, arr, Tuple(i)...)
end

Base.@propagate_inbounds function childindex(dst::AbstractArray{<:Any, N}, arr::SwizzledArray{<:Any, N, <:Any, <:Any, <:AbstractArray, <:AbstractArray{<:Any, M}}, i::Vararg{Integer, M}) where {N, M}
    dst_axes = axes(dst)
    imasktuple(d->firstindex(dst_axes[d]), d->i[d], Val(mask(arr)))
end

"""
   childindex(arr, i)

   For all wrapper arrays arr such that `arr` involves an index remapping,
   return the indices into `arr` which affect the indices `i` of `parent(arr)`.

   See also: [`swizzle`](@ref).
"""
childindex


#function Base.Broadcast.preprocess(dest, arr::SwizzledArray{T, N, Op, mask, Arg}) where {T, N, Arg, Op, mask}
#    arg = preprocess(dest, arr.arg)
#    SwizzledArray{T, N, Op, mask, typeof(arg)}(arr.op, arg)
#end

"""
    `childstyle(::Type{<:AbstractArray}, ::BroadcastStyle)`

Broadcast styles are used to determine behavior of objects under broadcasting.
To customize the broadcasting behavior of a wrapper array, one can first declare
how the broadcast style should behave under broadcasting after the wrapper array
is applied by overriding the `childstyle` method.
"""
@inline childstyle(Arr::Type{<:AbstractArray}, ::Any) = BroadcastStyle(Arr)

@inline childstyle(Arr::Type{<:SwizzledArray}, ::DefaultArrayStyle) = DefaultArrayStyle{ndims(Arr)}()
@inline childstyle(Arr::Type{<:SwizzledArray}, ::BroadcastStyle) = DefaultArrayStyle{ndims(Arr)}()
@inline childstyle(::Type{<:SwizzledArray}, ::ArrayConflict) = ArrayConflict()
@inline childstyle(Arr::Type{<:SwizzledArray}, ::Style{Tuple}) = mask(Arr) == (1,) ? Style{Tuple}() : DefaultArrayStyle{ndims(Arr)}()

@inline function Broadcast.BroadcastStyle(Arr::Type{<:SwizzledArray})
    childstyle(Arr, BroadcastStyle(parent(Arr)))
end

@inline function Swizzles.ExtrudedArrays.keeps(arr::SwizzledArray)
    arg_keeps = keeps(arr.arg)
    imasktuple(d->Extrude(), d->arg_keeps[d], Val(mask(arr)))
end

#=
function Swizzles.ExtrudedArrays.inferkeeps(Arr::Type{<:SwizzledArray})
    arg_keeps = inferkeeps(parent(Arr))
    imasktuple(d->Extrude(), d->arg_keeps[d], Val(mask(Arr)))
end
=#

function Swizzles.ExtrudedArrays.lift_keeps(arr::SwizzledArray)
    return adopt(arrayify(lift_keeps(parent(arr))), arr)
end
