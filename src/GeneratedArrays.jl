module GeneratedArrays

abstract type GeneratedArray{T, N} <: AbstractArray{T, N} end

using Base.Broadcast: Broadcasted
using Base.Broadcast: instantiate, broadcasted
using LinearAlgebra

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
        return Reduce(op, dims).(Ref(kwargs.init), arr)
    else
        return Reduce(op, dims).(arr)
    end
end

Base.@propagate_inbounds function Base.sum(op::Op, arr::GeneratedArray; dims=:, kwargs...) where {Op}
    if :init in keys(kwargs...)
        return Sum(dims).(Ref(kwargs.init), arr)
    else
        return Sum(dims).(arr)
    end
end

Base.@propagate_inbounds function Base.maximum(op::Op, arr::GeneratedArray; dims=:, kwargs...) where {Op}
    if :init in keys(kwargs...)
        return Max(dims).(Ref(kwargs.init), arr)
    else
        return Max(dims).(arr)
    end
end

Base.@propagate_inbounds function Base.minimum(op::Op, arr::GeneratedArray; dims=:, kwargs...) where {Op}
    if :init in keys(kwargs...)
        return Min(dims).(Ref(kwargs.init), arr)
    else
        return Min(dims).(arr)
    end
end

Base.@propagate_inbounds function LinearAlgebra.dot(x::GeneratedArray, y::AbstractArray)
    return Sum().(x .* y)
end

Base.@propagate_inbounds function LinearAlgebra.dot(x::AbstractArray, y::GeneratedArray)
    return Sum().(x .* y)
end

Base.@propagate_inbounds function LinearAlgebra.dot(x::GeneratedArray, y::GeneratedArray)
    return Sum().(x .* y)
end



struct Square{T, S} <: Number
    arg::T
    scale::S
end

@inline square(x) = Square(sign(x)^2, norm(x))

@inline root(x::Square) = sqrt(x.arg) * x.scale

function Base.promote_type(::Type{Square{T1, S1}}, ::Type{Square{T2, S2}}) where {T1, S1, T2, S2}
    return Square{promote_type(T1, T2), promote_type(S1, S2)}
end

function Base.convert(::Type{Square{T, S}}, x::Square) where {T, S, E}
    return Square(convert(T, x.arg), convert(S, x.scale))
end

@inline function Base.:+(x::T, y::T) where {T <: Square}
    if x.scale < y.scale
        (x, y) = (y, x)
    end
    if x.scale > y.scale
        if iszero(y.scale)
            return Square(x.arg + zero(y.arg) * (one(x.scale)/one(y.scale))^1, x.scale)
        else
            return Square(x.arg + y.arg * (x.scale/y.scale)^2, x.scale)
        end
    else
        return Square(x.arg + y.arg * (one(x.scale)/one(y.scale))^1, x.scale)
    end
end

struct Power{T, S, E} <: Number
    arg::T
    scale::S
    exponent::E
end

@inline power(x, p) = Power(sign(x)^p, norm(x), p)

@inline root(x::Power) = x.arg ^ inv(x.exponent) * x.scale

function Base.promote_type(::Type{Power{T1, S1, E1}}, ::Type{Power{T2, S2, E2}}) where {T1, S1, E1, T2, S2, E2}
    return Power{promote_type(T1, T2), promote_type(S1, S2), promote_type(E1, E2)}
end

function Base.convert(::Type{Power{T, S, E}}, x::Power) where {T, S, E}
    return Power(convert(T, x.arg), convert(S, x.scale), convert(E, x.exponent))
end

@inline function Base.:+(x::T, y::T) where {T <: Power}
    x.exponent == y.exponent || ArgumentError("Cannot accurately add Powers with different exponents")
    if x.scale < y.scale
        (x, y) = (y, x)
    end
    if x.scale > y.scale
        if iszero(y.scale)
            return Power(x.arg + zero(y.arg) * (one(x.scale)/one(y.scale))^one(y.exponent), x.scale, x.exponent)
        else
            return Power(x.arg + y.arg * (x.scale/y.scale)^y.exponent, x.scale, x.exponent)
        end
    else
        return Power(x.arg + y.arg * (one(x.scale)/one(y.scale))^one(y.exponent), x.scale, x.exponent)
    end
end

end
