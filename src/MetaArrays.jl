module MetaArrays

abstract type MetaArray{T, N} <: AbstractArray{T, N} end

"""
    MetaArray <: AbstractArray

An AbstractArray that is mostly implemented using Broadcast.

See also: [`parent`](@ref)
"""
abstract type MetaArray{T, N} <: AbstractArray{T, N} end

Base.copyto!(dst::MetaArray, src) = _copyto!(dst, src)
Base.copyto!(dst::MetaArray, src::AbstractArray) = _copyto!(dst, src)
Base.copyto!(dst, src::MetaArray) = _copyto!(dst, src)
Base.copyto!(dst::AbstractArray, src::MetaArray) = _copyto!(dst, src)
Base.copyto!(dst::MetaArray, src::MetaArray) = _copyto!(dst, src)
function _copyto!(dst, src)
    if axes(dst) != axes(src)
        assign!(reshape(dst, :), reshape(src, :))
    else
        assign!(dst, src)
    end
end

Base.copy!(dst::MetaArray, src) = _copy!(dst, src)
Base.copy!(dst::MetaArray, src::AbstractArray) = _copy!(dst, src)
Base.copy!(dst, src::MetaArray) = _copy!(dst, src)
Base.copy!(dst::AbstractArray, src::MetaArray) = _copy!(dst, src)
Base.copy!(dst::MetaArray, src::MetaArray) = _copy!(dst, src)
function _copy!(dst, src)
    if axes(dst) != axes(src)
        throw(ArgumentError("axes of $dst and $src do not match."))
    else
        assign!(dst, src)
    end
end

@inline Base.Broadcast.materialize(A::MetaArray) = copy(A)
@inline Base.Broadcast.materialize!(dst, A::MetaArray) = copyto!(dst, A)

struct PreprocessAssign{T} where T next::T end

struct FindDstAssign{T} where T next::T end
struct MatchDstAssign{T} where T next::T end
struct UnmatchDstAssign{T} where T next::T end

struct MatchedArray{T, N, Arg} <: WrapperArray{T, N, Arg}
    arg::Arg
end
Base.parent(arr::MatchedArray) = arr.arg
WrappedArrays.iswrapper(::MatchedArray) = true
WrappedArrays.adopt(::MatchedArray, arr) = arr #don't adopt MatchedArrays

function finddst(dst, bc::Broadcasted{Style}) where {Style}
    return Broadcasted{Style}(bc.f, map(arg->finddst(dst, arg), bc.args), bc.axes)
end
function finddst(dst, arr::AbstractArray)
    if arr === dst
        return MatchedArray(arr)
    end
    if iswrapper(arr)
        arr = adopt(finddst(dst, parent(arr)), arr)
    else
        arr
    end
end
function finddst(dst, arr::BroadcastedArray{T, N}) where {T, N}
    arg = finddst(dst, arr.arg)
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

assign_src_style(x) = FindDstAssign(MatchAssign(UnmatchAssign(PreprocessAssign(BroadcastStyle(x.arg))))

function assign!(dst::AbstractArray, src::AbstractArray)
    return assign!(dst, src, assign_src_style(src))
end
function assign!(dst::AbstractArray, src::AbstractArray, ::Any)
    if axes(dst) != axes(src)
        throw(DimensionMismatch("destination axes $axdest are not compatible with source axes $axsrc"))
    end
    src = preprocess(dst, src)
    @simd for I in eachindex(src)
        @inbounds dst[I] = op(dst[I], src[I])
    end
    return dst
end

function assign!(dst::AbstractArray, src::AbstractArray, style::FindDstAssign)
    assign!(dst, finddst(dst, src), style.next)
end

function assign!(dst::AbstractArray, src::AbstractArray, style::MatchAssign)
    assign!(dst, src, style.next)
end

function assign!(dst::AbstractArray, src::AbstractArray, style::UnmatchAssign)
    assign!(dst, unmatch(src), style.next)
end

function assign!(dst::AbstractArray, src::AbstractArray, style::PreprocessAssign)
    assign!(dst, preprocess(dst, src), style.next)
end

function assign!(dst::AbstractArray, src::BroadcastedArray{Broadcasted{<:Any, <:Any, <:Any, Tuple{<:MatchArray}}}, style::MatchAssign)
    @simd for I in eachindex(dst)
        @inbounds dst[I] = src.arg.f(dst[I])
    end
    return dst
end

#=
struct NullArray{N} <: AbstractArray{<:Any, N}
    axes::NTuple{N}
end

Base.axes(arr::NullArray) = arr.axes
Base.setindex!(arr, val, inds...) = val

Base.foreach(f, a::MetaArray) = assign!(NullArray(axes(a)), a)

=#
end
