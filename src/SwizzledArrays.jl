using Swizzles.Properties
using Swizzles.WrapperArrays
using Swizzles.BroadcastedArrays
using Swizzles.GeneratedArrays
using Swizzles.ExtrudedArrays
using Base: checkbounds_indices, throw_boundserror, tail, dataids, unaliascopy, unalias
using Base.Iterators: reverse, repeated, countfrom, flatten, product, take, peel, EltypeUnknown
using Base.Broadcast: Broadcasted, BroadcastStyle, Style, DefaultArrayStyle, AbstractArrayStyle, Unknown, ArrayConflict
using Base.Broadcast: materialize, materialize!, instantiate, broadcastable, preprocess, _broadcast_getindex, combine_eltypes

"""
    `nooperator(a, b)`

An operator which does not expect to be called. It startles easily.
"""
nooperator(a, b) = throw(ArgumentError("unspecified operator"))

@inline function parse_swizzle_mask(arr, _mask::Tuple{Vararg{Union{Int, Drop}, M}}) where {M}
    return ntuple(d -> d <= M ? _mask[d] : drop, Val(ndims(arr)))
end

struct SwizzledArray{T, N, Arg<:AbstractArray, Op, mask} <: GeneratedArray{T, N}
    arg::Arg
    op::Op
    function SwizzledArray{T, N, Arg, Op, mask}(arg::Arg, op::Op) where {T, N, Arg, Op, mask}
        #FIXME check swizzles. also check noop axes!
        new(arg, op)
    end
end

@inline function SwizzledArray{T}(arr::SwizzledArray{S, N, Arg, Op, mask}) where {T, S, N, Arg, Op, mask}
    return SwizzledArray{T, N, Arg, Op, mask}(arr.arg, arr.op)
end
@inline function SwizzledArray{T, N, Arg, Op, mask}(arr::SwizzledArray{S, N, Arg, Op, mask}) where {T, S, N, Arg, Op, mask}
    return SwizzledArray{T, N, Arg, Op, mask}(arr.arg, arr.op)
end

@inline function SwizzledArray(arg, op, _mask...)
    return SwizzledArray{nothing}(arg, op, _mask...)
end
@inline function SwizzledArray{T}(arg, op, _mask...) where {T}
    return SwizzledArray{T}(arg, op, Val(_mask))
end
@inline function SwizzledArray{T}(arg, op, _mask::Tuple) where {T}
    return SwizzledArray{T}(arg, op, Val(_mask))
end
@inline function SwizzledArray{T}(arg, op, ::Val{_mask}) where {T, _mask}
    if @generated
        mask = parse_swizzle_mask(arg, _mask)
        return :(return SwizzledArray{T, $(max(0, mask...)), typeof(arg), typeof(op), $mask}(arg, op))
    else
        mask = parse_swizzle_mask(arg, _mask)
        return SwizzledArray{T, max(0, mask...), typeof(arg), typeof(op), mask}(arg, op)
    end
end
@inline function SwizzledArray{nothing, N, Arg, Op, mask}(arg::Arg, op::Op) where {N, Arg, Op, mask}
    arr = SwizzledArray{Any, N, Arg, Op, mask}(arg, op)
    return SwizzledArray{Properties.eltype_bound(arr)}(arr)
end



@inline mask(::Type{SwizzledArray{T, N, Arg, Op, _mask}}) where {T, N, Arg, Op, _mask} = _mask
@inline mask(::SwizzledArray{T, N, Arg, Op, _mask}) where {T, N, Arg, Op, _mask} = _mask

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



function Base.show(io::IO, arr::SwizzledArray)
    print(io, SwizzledArray)
    print(io, '(', arr.arg, ", ", arr.op, ", ", mask(arr), ')')
    nothing
end

Base.parent(arr::SwizzledArray) = arr.arg
Base.parent(::Type{<:SwizzledArray{<:Any, <:Any, Arg}}) where {Arg} = Arg
WrapperArrays.iswrapper(arr::SwizzledArray) = true
function WrapperArrays.adopt(arg, arr::SwizzledArray{T, N, <:Any, Op, mask}) where {T, N, Op, mask}
    SwizzledArray{T, N, typeof(arg), Op, mask}(arg, arr.op)
end

Base.dataids(arr::SwizzledArray) = (dataids(arr.arg), dataids(arr.op))
Base.unaliascopy(arr::SwizzledArray{T, N, Arg, Op, mask}) where {T, N, Arg, Op, mask} = SwizzledArray{T, N, Arg, Op, mask}(unaliascopy(arr.arg), unaliascopy(arr.op))
Base.unalias(dest, arr::SwizzledArray{T, N, Arg, Op, mask}) where {T, N, Arg, Op, mask} = SwizzledArray{T, N, Arg, Op, mask}(unalias(dest, arr.arg), unalias(dest, arr.op))

@inline function Base.size(arr::SwizzledArray)
    imasktuple(d->1, d->size(arr.arg, d), Val(mask(arr)))
end

@inline function Base.axes(arr::SwizzledArray)
    imasktuple(d->Base.OneTo(1), d->axes(arr.arg, d), Val(mask(arr)))
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
            return SwizzledArray{eltype(src)}(SubArray(arg, parentindex(arr, inds...)), arr.op, $mask′)
        end
    else
        mask′ = _convert_remask(inds, mask(arr)...)
        return SwizzledArray{eltype(src)}(SubArray(arg, parentindex(arr, inds...)), arr.op, mask′)
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

@generated function Base.copy(src::Broadcasted{DefaultArrayStyle{0}, <:Any, typeof(identity), <:Tuple{Arr}}) where {Arr <: SwizzledArray}
    return quote
        Base.@_propagate_inbounds_meta
        arr = src.args[1]
        arg = arr.arg
        if mask(arr) isa Tuple{Vararg{Int}}
            @inbounds return arg[]
        elseif Properties.initial(arr.op, eltype(arr), eltype(arg)) !== nothing
            dst = something(Properties.initial(arr.op, eltype(arr), eltype(arg)))
            arg = BroadcastedArrays.preprocess(nothing, arg)
            @inbounds for i in eachindex(arg)
                dst = arr.op(dst, arg[i])
            end
            return dst
        else
            arg = BroadcastedArrays.preprocess(nothing, arg)
            arg_axes = axes(arg)
            arg_firstindices = ($((:(arg_axes[$n][1]) for n = 1:length(mask((Arr))))...),)
            arg_restindices = ($((:(view(arg_axes[$n], 2:lastindex(arg_axes[$n]))) for n = 1:length(mask(Arr)))...),)
            arg_keeps = keeps(arg)
            @inbounds $(begin
                i′ = [Symbol("i′_$d") for d = 1:length(mask(Arr))]
                thunk = Expr(:block)
                for n = vcat(0, findall(d -> d isa Drop, mask(Arr)))
                    if n > 0
                        nest = :(dst = arr.op(dst, getindex(arg, $(i′...))))
                    else
                        nest = :(dst = getindex(arg, $(i′...)))
                    end
                    for d = 1:length(mask(Arr))
                        if mask(Arr)[d] isa Drop
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
end

Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Broadcasted{Nothing, <:Any, typeof(identity), <:Tuple{SwizzledArray}})
    #This method gets called when the destination eltype is unsuitable for
    #accumulating the swizzle. Therefore, we should allocate a suitable
    #destination and then accumulate.
    arr = src.args[1]
    copyto!(dst, copyto!(similar(arr), arr))
end

@generated function Base.copyto!(dst::AbstractArray{T, N}, src::Broadcasted{Nothing, <:Any, typeof(identity), Tuple{Arr}}) where {T, N, Arr <: SwizzledArray{<:T, N}}
    quote
        Base.@_propagate_inbounds_meta
        arr = src.args[1]
        arg = arr.arg
        if mask(arr) isa Tuple{Vararg{Int}}
            arg = BroadcastedArrays.preprocess(dst, arr.arg)
            @inbounds for i in eachindex(arg)
                i′ = childindex(dst, arr, i)
                dst[i′...] = arg[i]
            end
        elseif Properties.initial(arr.op, eltype(arr), eltype(arg)) !== nothing
            dst .= something(Properties.initial(arr.op, eltype(arr), eltype(arr.arg)))
            dst .= (arr.op).(dst, arr)
        else
            arg = BroadcastedArrays.preprocess(dst, arr.arg)
            arg_axes = axes(arg)
            arg_firstindices = ($((:(arg_axes[$n][1]) for n = 1:length(mask((Arr))))...),)
            arg_restindices = ($((:(view(arg_axes[$n], 2:lastindex(arg_axes[$n]))) for n = 1:length(mask(Arr)))...),)
            arg_keeps = keeps(arg)
            @inbounds $(begin
                i′ = [Symbol("i′_$d") for d = 1:length(mask(Arr))]
                i = imasktuple(d->:(firstindex(dst, $d)), d->i′[d], mask(Arr))
                thunk = Expr(:block)
                for n = vcat(0, findall(d -> d isa Drop, mask(Arr)))
                    if n > 0
                        nest = :(dst[$(i...)] = arr.op(dst[$(i...)], getindex(arg, $(i′...))))
                    else
                        nest = :(dst[$(i...)] = getindex(arg, $(i′...)))
                    end
                    for d = 1:length(mask(Arr))
                        if mask(Arr)[d] isa Drop
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
    @inbounds for i in eachindex(arg)
        i′ = childindex(dst, arr, i)
        dst[i′...] = arr.op(dst[i′...], arg[i])
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

Base.@propagate_inbounds function parentindex(arr::SwizzledArray{<:Any, N, Arg}, i::Vararg{Any, N}) where {N, Arg}
    masktuple(d->Base.Slice(axes(arr.arg, d)), d->i[d], Val(mask(arr)))
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

Base.@propagate_inbounds function childindex(dst::AbstractArray{<:Any, N}, arr::SwizzledArray{<:Any, N, <:AbstractArray{<:Any, M}}, i::Vararg{Integer, M}) where {N, M}
    imasktuple(d->firstindex(axes(dst, d)), d->i[d], Val(mask(arr)))
end

"""
   childindex(arr, i)

   For all wrapper arrays arr such that `arr` involves an index remapping,
   return the indices into `arr` which affect the indices `i` of `parent(arr)`.

   See also: [`swizzle`](@ref).
"""
childindex


#function Base.Broadcast.preprocess(dest, arr::SwizzledArray{T, N, Arg, Op, mask}) where {T, N, Arg, Op, mask}
#    arg = preprocess(dest, arr.arg)
#    SwizzledArray{T, N, typeof(arg), Op, mask}(arg, arr.op)
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

@inline function Broadcast.BroadcastStyle(::Type{Arr}) where {Arg, Arr <: SwizzledArray{<:Any, <:Any, Arg}}
    childstyle(Arr, BroadcastStyle(Arg))
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
    arg_keeps = keeps(arr.arg)
    imasktuple(d->false, d->arg_keeps[d], Val(mask(arr)))
end

function Swizzles.ExtrudedArrays.keeps(::Type{Arr}) where {Arg, Arr <: SwizzledArray{<:Any, <:Any, <:Arg}}
    arg_keeps = keeps(Arg)
    imasktuple(d->false, d->arg_keeps[d], Val(mask(Arr)))
end

function Swizzles.ExtrudedArrays.lift_keeps(arr::SwizzledArray{T, N, Arg, Op, mask}) where {T, N, Arg, Op, mask}
    arg = arrayify(lift_keeps(arr.arg))
    return SwizzledArray{T, N, typeof(arg), Op, mask}(arg, arr.op)
end
