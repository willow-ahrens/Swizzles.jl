module GeneratedArrays

abstract type GeneratedArray{T, N} <: AbstractArray{T, N} end

using Base.Broadcast: Broadcasted
using Base.Broadcast: instantiate, broadcasted

export GeneratedArray

"""
    GeneratedArray <: AbstractArray

A convenience type for defining array types that are mostly implemented using
copyto!(::Any, ::Broadcasted) and copy(::Broadcasted). Consult the [Interfaces
chapter](@ref man-interfaces-broadcasting) on broadcasting for more info about
broadcast. Many Base functions are implemented for GeneratedArrays in terms of
`copyto!` including `map`, `getindex`, and `foreach`. Note that
`copy(::GeneratedArray)` is not implemented in terms of `Broadcast`, as `copy`
is intended to be a shallow copy function on arrays.

See also: [`copy`](@ref), [`copyto!`](@ref)
"""
abstract type GeneratedArray{T, N} <: AbstractArray{T, N} end

#Beware infinite recursion!

Base.copyto!(dst, src::GeneratedArray) = copyto!(dst, Array(src))

Base.copyto!(dst::GeneratedArray, src) = _copyto!(dst, broadcastable(src))
Base.copyto!(dst::GeneratedArray, src::AbstractArray) = _copyto!(dst, src)
Base.copyto!(dst::AbstractArray, src::GeneratedArray) = _copyto!(dst, src)
Base.copyto!(dst::GeneratedArray, src::GeneratedArray) = _copyto!(dst, src)
Base.copyto!(dst::GeneratedArray, src::Broadcasted) = invoke(copyto!, Tuple{AbstractArray, typeof(src)}, dst, src)

function _copyto!(dst::AbstractArray, src)
    if axes(dst) != axes(src)
        reshape(dst, axes(src)) .= src
    else
        dst .= src
    end
end

#OVERRIDE ALL THE THINGS!

@inline Base.Broadcast.materialize(A::GeneratedArray) = identity.(A)
@inline Base.Broadcast.materialize!(dst, A::GeneratedArray) = dst .= A

#The following nonsense means that generated arrays can override getindex or they can override copyto!(view)
Base.@propagate_inbounds Base.getindex(arr::GeneratedArray, I::Integer) = _getindex(arr, I)
Base.@propagate_inbounds Base.getindex(arr::GeneratedArray, I::CartesianIndex) = _getindex(arr, I)
Base.@propagate_inbounds Base.getindex(arr::GeneratedArray, I...) = _getindex(arr, I...)

Base.@propagate_inbounds function _getindex(arr, I...)::eltype(arr)
    identity.(view(arr, I...))
end

#The following nonsense means that generated arrays can override getindex or they can override copyto!(view)
Base.@propagate_inbounds Base.setindex!(arr::GeneratedArray, v, I::Integer) = _setindex!(arr, v, I)
Base.@propagate_inbounds Base.setindex!(arr::GeneratedArray, v, I::CartesianIndex) = _setindex!(arr, v, I)
Base.@propagate_inbounds Base.setindex!(arr::GeneratedArray, v, I...) = _setindex!(arr, v, I...)

Base.@propagate_inbounds function _setindex!(arr, v, I...)
    view(arr, I...) .= v
end

Base.@propagate_inbounds function Base.map(f::F, arr::GeneratedArray, tail...) where {F}
    f.(arr, tail...)
end

Base.@propagate_inbounds function Base.foreach(f::F, arr::GeneratedArray, tail...) where {F}
    NullArray(axes(arr)) .= f.(arr, tail...)
    return nothing
end

Base.@propagate_inbounds function Base.reduce(op::Op, arr::GeneratedArray; dims=:, kwargs...) where {Op}
    if :init in keys(kwargs...)
        return Reduce(op, dims).(kwargs.init, arr)
    else
        return Reduce(op, dims).(arr)
    end
end

Base.@propagate_inbounds function Base.sum(op::Op, arr::GeneratedArray; dims=:, kwargs...) where {Op}
    if :init in keys(kwargs...)
        return Sum(dims).(kwargs.init, arr)
    else
        return Sum(dims).(arr)
    end
end

Base.@propagate_inbounds function Base.maximum(op::Op, arr::GeneratedArray; dims=:, kwargs...) where {Op}
    if :init in keys(kwargs...)
        return Max(dims).(kwargs.init, arr)
    else
        return Max(dims).(arr)
    end
end

Base.@propagate_inbounds function Base.minimum(op::Op, arr::GeneratedArray; dims=:, kwargs...) where {Op}
    if :init in keys(kwargs...)
        return Min(dims).(kwargs.init, arr)
    else
        return Min(dims).(arr)
    end
end

#=
#do overrides for wierd copyto!s, map, foreach, etc...

struct ScaledPower{T, S, E}
    value::T
    scale::S
    exponent::E
end

function Base.:^(x::ScaledPower, y)
    return x.scale^y * x.value^(y + x.exponent)
end

function incbypow(x::ScaledPower{T, S, E}, y::S)
    if y != zero(y)
        ay = abs(y)
        if x.scale < ay
            value = one(T) + x.value * (x.scale/ay)^x.exponent
            scale = ay
        else
            value = value + (ay/x.scale)^x.exponent
            scale = x.scale
        end
    end
    return ScaledPower(value, scale, x.exponent)
end

Base.LinearAlgebra.norm(x::GeneratedArray, p) = Reduce(incbypow).(x, ScaledPower(zero(eltype(x)), zero(eltype(x)), p))^(-p)

distance(x, y) = Reduce(incbypow).(x .- y, ScaledPower(zero(eltype(x)), zero(eltype(x)), 2))^(-2)

struct NullArray{N} <: AbstractArray{<:Any, N}
    axes::NTuple{N}
end

Base.axes(arr::NullArray) = arr.axes
Base.setindex!(arr, val, inds...) = val

Base.foreach(f, a::GeneratedArray) = assign!(NullArray(axes(a)), a)
function ArrayifiedArrays.assign!(dst::NullArray, MetaArray(op, arg)) #foreach
=#


end
