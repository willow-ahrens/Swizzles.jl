module MatchedArrays

using Swizzle.WrapperArrays
using Swizzle.ShallowArrays
using Swizzle.BroadcastedArrays

using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle
using Base.Broadcast: result_style

export MatchedArray, MatchDestinationStyle, MarkDestinationStyle

struct MatchedArray{T, N, Arg} <: ShallowArray{T, N, Arg}
    arg::Arg
end
Base.parent(arr::MatchedArray) = arr.arg
WrapperArrays.iswrapper(::MatchedArray) = true
WrapperArrays.adopt(::MatchedArray, arr) = arr #don't adopt MatchedArrays

mark_destination(dst, src) = src
function mark_destination(dst, src::Broadcasted{Style}) where {Style}
    return Broadcasted{Style}(src.f, map(arg->mark_destination(dst, arg), src.args), src.axes)
end
function mark_destination(dst, src::AbstractArray)
    if src === dst
        return MatchedArray(src)
    end
    if iswrapper(src)
        src = adopt(mark_destination(dst, parent(src)), src)
    else
        src
    end
end
function mark_destination(dst, src::BroadcastedArray{T, N}) where {T, N}
    arg = mark_destination(dst, src.arg)
    x = BroadcastedArray{T, N, typeof(arg)}(arg)
end

unmatch(src) = src
function unmatch(src::Broadcasted{Style}) where {Style}
    return Broadcasted{Style}(src.f, map(unmatch, src.args), src.axes)
end
function unmatch(src::AbstractArray)
    if iswrapper(src)
        src = adopt(unmatch(parent(src)), src)
    else
        src
    end
end
function unmatch(src::BroadcastedArray{T, N}) where {T, N}
    arg = unmatch(src.arg)
    x = BroadcastedArray{T, N, typeof(arg)}(arg)
end
unmatch(src::MatchedArray) = parent(src)

struct MarkDestinationStyle{S<:BroadcastStyle} <: BroadcastStyle end
MarkDestinationStyle(style::S) where {S <: BroadcastStyle} = MarkDestinationStyle{S}()
Base.Broadcast.BroadcastStyle(::S, ::MarkDestinationStyle{T}) where {S<:BroadcastStyle, T} = MarkDestinationStyle(result_style(S(), T()))
Base.Broadcast.BroadcastStyle(::MarkDestinationStyle{S}, ::MarkDestinationStyle{T}) where {S<:BroadcastStyle, T} = MarkDestinationStyle(result_style(S(), T()))

struct MatchDestinationStyle{S<:BroadcastStyle} <: BroadcastStyle end
MatchDestinationStyle(style::S) where {S <: BroadcastStyle} = MatchDestinationStyle{S}()

function Base.copyto!(dst::AbstractArray, src::Broadcasted{<:MarkDestinationStyle{S}}) where {S}
    copyto!(dst, Broadcasted{MatchDestinationStyle{S}}(src.f, map(arg->mark_destination(dst, arg), src.args), src.axes))
end

function Base.copyto!(dst::AbstractArray, src::Broadcasted{<:MatchDestinationStyle{S}}) where {S}
    copyto!(dst, Broadcasted{S}(src.f, map(unmatch, src.args), src.axes))
end

#in-place update pattern
function Base.copyto!(dst::AbstractArray, src::Broadcasted{<:MatchDestinationStyle{<:AbstractArrayStyle}, <:Any, F, <:Tuple{<:MatchedArray}}) where {F}
    @boundscheck axes(dst) == axes(src) || throw(DimensionMismatchError("FIXME"))
    arg = src.args[1]
    @simd for I in eachindex(arg)
        @inbounds arg[I] = src.f(arg[I])
    end
end

end
