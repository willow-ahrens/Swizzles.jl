module MatchStyles

struct MatchedArray{T, N, Arg} <: ShallowArray{T, N, Arg}
    arg::Arg
end
Base.parent(arr::MatchedArray) = arr.arg
WrappedArrays.iswrapper(::MatchedArray) = true
WrappedArrays.adopt(::MatchedArray, arr) = arr #don't adopt MatchedArrays

function mark_destination(dst, bc::Broadcasted{Style}) where {Style}
    return Broadcasted{Style}(bc.f, map(arg->mark_destination(dst, arg), bc.args), bc.axes)
end
function mark_destination(dst, arr::AbstractArray)
    if arr === dst
        return MatchedArray(arr)
    end
    if iswrapper(arr)
        arr = adopt(mark_destination(dst, parent(arr)), arr)
    else
        arr
    end
end
function mark_destination(dst, arr::BroadcastedArray{T, N}) where {T, N}
    arg = mark_destination(dst, arr.arg)
    x = BroadcastedArray{T, N, typeof(arg)}(arg)
end

function unmatch(dst, bc::Broadcasted{Style}) where {Style}
    return Broadcasted{Style}(bc.f, map(unmatch, bc.args), bc.axes)
end
function unmatch(arr::AbstractArray)
    if iswrapper(arr)
        arr = adopt(unmatch(parent(arr)), arr)
    else
        arr
    end
end
function unmatch(arr::BroadcastedArray{T, N}) where {T, N}
    arg = unmatch(arr.arg)
    x = BroadcastedArray{T, N, typeof(arg)}(arg)
end
unmatch(arr::MatchedArray) = parent(arr)

struct MarkDestinationStyle{S<:BroadcastStyle} <: BroadcastStyle end
MarkDestinationStyle(style::S) where {S <: BroadcastStyle} = MarkDestinationStyle{S}()
Base.Broadcast.BroadcastStyle(::S, ::MarkDestinationStyle{T}) where {S, T} = MarkDestinationStyle(combine_styles(S, T))
Base.Broadcast.BroadcastStyle(::MarkDestinationStyle{S}, ::MarkDestinationStyle{T}) where {S, T} = MarkDestinationStyle(combine_styles(S, T))

struct MatchDestinationStyle{S<:BroadcastStyle} <: BroadcastStyle end
MatchDestinationStyle(style::S) where {S <: BroadcastStyle} = MatchDestinationStyle{S}()


function Base.copyto!(dst::AbstractArray, src::Broadcasted{<:MarkDestinationStyle{S}}) where {S}
    copyto!(dst, Broadcasted{MatchDestinationStyle{S}}(bc.f, map(arg->mark_destination(dst, arg), bc.args), bc.axes))
end

function Base.copyto!(dst::AbstractArray, src::Broadcasted{<:MatchDestinationStyle{S}}) where {S}
    copyto!(dst, Broadcasted{S}(bc.f, map(unmatch, bc.args), bc.axes))
end

#in-place update pattern
function Base.copyto!(dst::AbstractArray, src::Broadcasted{<:MatchDestinationStyle{<:AbstractArrayStyle}, <:Any, F, <:Tuple{<:MatchArray}}) where {F}
    @boundscheck axes(dst) == axes(src) || throw(DimensionMismatchError("FIXME"))
    arg = src.args[1]
    @simd for I in eachindex(arg)
        @inbounds arg[I] = src.f(arg[I])
    end
end

end
