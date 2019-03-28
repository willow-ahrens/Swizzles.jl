module GeneratedArrays

abstract type GeneratedArray{T, N} <: AbstractArray{T, N} end

using Base.Broadcast: Broadcasted, AbstractArrayStyle, preprocess
using Base.Broadcast: instantiate, broadcasted
using LinearAlgebra
using Swizzles
using Swizzles.NullArrays

export GeneratedArray, myidentity

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

Base.copy(src::GeneratedArray) = copyto!(similar(src), src)

Base.copyto!(dst, src::GeneratedArray) = copyto!(dst, Array(src))

Base.copyto!(dst::GeneratedArray, src::AbstractArray) = _copyto!(dst, src)
Base.copyto!(dst::AbstractArray, src::GeneratedArray) = _copyto!(dst, src)
Base.copyto!(dst::GeneratedArray, src::GeneratedArray) = _copyto!(dst, src)
myidentity(x) = x
function _copyto!(dst::AbstractArray, src)
    if axes(dst) != axes(src)
        dst .= myidentity.(reshape(arrayify(src), axes(dst)))
    else
        dst .= myidentity.(src)
    end
    return dst
end



Base.@propagate_inbounds Base.Broadcast.materialize(A::GeneratedArray) = identity.(A)
Base.@propagate_inbounds Base.Broadcast.materialize!(dst, A::GeneratedArray) = dst .= A



#The following nonsense means that generated arrays can override getindex or they can override copyto!(view)
Base.@propagate_inbounds Base.getindex(arr::GeneratedArray, I::Integer)::eltype(arr) = _getindex(arr, I)
Base.@propagate_inbounds Base.getindex(arr::GeneratedArray, I::CartesianIndex)::eltype(arr) = _getindex(arr, I)
Base.@propagate_inbounds Base.getindex(arr::GeneratedArray, I...) = _getindex(arr, I...)

Base.@propagate_inbounds function _getindex(arr, I...)
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
    return _reduce(op, arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _reduce(op::Op, arr::GeneratedArray, dims, nt::NamedTuple{()}) where {Op}
    return Reduce(op, dims).(arr)
end
Base.@propagate_inbounds function _reduce(op::Op, arr::GeneratedArray, dims, nt::NamedTuple{(:init,)}) where {Op}
    return Reduce(op, dims).(nt.init, arr)
end



Base.@propagate_inbounds function Base.mapreduce(f::F, op::Op, arr::GeneratedArray; dims=:, kwargs...) where {F, Op}
    return _mapreduce(f, op, arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _mapreduce(f::F, op::Op, arr::GeneratedArray, dims, nt::NamedTuple{()}) where {F, Op}
    return Reduce(op, dims).(f.(arr))
end
Base.@propagate_inbounds function _mapreduce(f::F, op::Op, arr::GeneratedArray, dims, nt::NamedTuple{(:init,)}) where {F, Op}
    return Reduce(op, dims).(nt.init, f.(arr))
end



Base.@propagate_inbounds function Base.sum(arr::GeneratedArray; dims=:, kwargs...)
    return _sum(arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _sum(arr::GeneratedArray, dims, nt::NamedTuple{()})
    return Sum(dims).(arr)
end
Base.@propagate_inbounds function _sum(arr::GeneratedArray, dims, nt::NamedTuple{(:init,)})
    return Sum(dims).(nt.init, arr)
end



Base.@propagate_inbounds function Base.maximum(arr::GeneratedArray; dims=:, kwargs...)
    return _maximum(arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _maximum(arr::GeneratedArray, dims, nt::NamedTuple{()})
    return Reduce(max, dims).(arr)
end
Base.@propagate_inbounds function _maximum(arr::GeneratedArray, dims, nt::NamedTuple{(:init,)})
    return Reduce(max, dims).(nt.init, arr)
end



Base.@propagate_inbounds function Base.minimum(arr::GeneratedArray; dims=:, kwargs...)
    return _minimum(arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _minimum(arr::GeneratedArray, dims, nt::NamedTuple{()})
    return Reduce(min, dims).(arr)
end
Base.@propagate_inbounds function _minimum(arr::GeneratedArray, dims, nt::NamedTuple{(:init,)})
    return Reduce(min, dims).(nt.init, arr)
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



Base.@propagate_inbounds function LinearAlgebra.adjoint(x::GeneratedArray)
    return arrayify(Delay().(Beam(2, 1).(adjoint.(x))))
end

Base.@propagate_inbounds function LinearAlgebra.transpose(x::GeneratedArray)
    return arrayify(Delay().(Beam(2, 1).(transpose.(x))))
end

Base.@propagate_inbounds function Base.permutedims(x::GeneratedArray)
    return arrayify(Delay().(Beam(2, 1).(x)))
end

Base.@propagate_inbounds function Base.permutedims(x::GeneratedArray, perm)
    return arrayify(Delay().(Focus(perm...).(x)))
end



Ts = (:(GeneratedArray{<:Any, 1}), :(GeneratedArray{<:Any, 2}), :(AbstractArray{<:Any, 1}), :(AbstractArray{<:Any, 2}))
for (A, B) in Iterators.product(Ts, Ts)
    if :(GeneratedArray{<:Any, 1}) in (A, B) || :(GeneratedArray{<:Any, 2}) in (A, B)
        @eval begin
            Base.@propagate_inbounds function Base.:*(A::$A, B::$B)
                return SumOut(2).(A.*Beam(2, 3).(B))
            end
        end
    end
end

for (Y, A, B) in Iterators.product(Ts, Ts, Ts)
    if :(GeneratedArray{<:Any, 1}) in (Y, A, B) || :(GeneratedArray{<:Any, 2}) in (Y, A, B)
        @eval begin
            Base.@propagate_inbounds function LinearAlgebra.mul!(Y::$Y, A::$A, B::$B)
                return Y .= SumOut(2).(A.*Beam(2, 3).(B))
            end
        end
    end
end



struct Square{T, S} <: Number
    arg::T
    scale::S
end

@inline square(x) = Square(sign(x)^2, norm(x))

@inline root(x::Square) = sqrt(x.arg) * x.scale

@inline Base.zero(::Type{Square{T, S}}) where {T, S} = Square{T, S}(zero(T), zero(S))

function Base.promote_rule(::Type{Square{T1, S1}}, ::Type{Square{T2, S2}}) where {T1, S1, T2, S2}
    return Square{promote_type(T1, T2), promote_type(S1, S2)}
end

function Base.convert(::Type{Square{T, S}}, x::Square) where {T, S}
    return Square(convert(T, x.arg), convert(S, x.scale))
end

@inline function Base.:+(x::T, y::T) where {T <: Square}
    if x.scale < y.scale
        (x, y) = (y, x)
    end
    if x.scale > y.scale
        if iszero(y.scale)
            return Square(x.arg + zero(y.arg) * (one(y.scale)/one(x.scale))^1, x.scale)
        else
            return Square(x.arg + y.arg * (y.scale/x.scale)^2, x.scale)
        end
    else
        return Square(x.arg + y.arg * (one(y.scale)/one(x.scale))^1, x.scale)
    end
end

struct Power{T, S, E} <: Number
    arg::T
    scale::S
    exponent::E
end

@inline power(x, p) = Power(sign(x)^p, norm(x), p)

@inline root(x::Power) = x.arg ^ inv(x.exponent) * x.scale

@inline Base.zero(::Type{Power{T, S, E}}) where {T, S, E} = Power{T, S, E}(zero(T), zero(S), one(E))
@inline Base.zero(x::Power) = Power(zero(x.arg), zero(x.scale), x.exponent)

function Base.promote_rule(::Type{Power{T1, S1, E1}}, ::Type{Power{T2, S2, E2}}) where {T1, S1, E1, T2, S2, E2}
    return Power{promote_type(T1, T2), promote_type(S1, S2), promote_type(E1, E2)}
end

function Base.convert(::Type{Power{T, S, E}}, x::Power) where {T, S, E}
    return Power(convert(T, x.arg), convert(S, x.scale), convert(E, x.exponent))
end

@inline function Base.:+(x::T, y::T) where {T <: Power}
    if x.exponent != y.exponent
        if iszero(x.arg) && iszero(x.scale)
            (x, y) = (y, x)
        end
        if iszero(y.arg) && iszero(y.scale)
            y = Power(y.arg, y.scale, x.exponent)
        else
            ArgumentError("Cannot accurately add Powers with different exponents")
        end
    end
    #TODO handle negative exponent
    if x.scale < y.scale
        (x, y) = (y, x)
    end
    if x.scale > y.scale
        if iszero(y.scale)
            return Power(x.arg + zero(y.arg) * (one(y.scale)/one(x.scale))^one(y.exponent), x.scale, x.exponent)
        else
            return Power(x.arg + y.arg * (y.scale/x.scale)^y.exponent, x.scale, x.exponent)
        end
    else
        return Power(x.arg + y.arg * (one(y.scale)/one(x.scale))^one(y.exponent), x.scale, x.exponent)
    end
end

Base.@propagate_inbounds function LinearAlgebra.norm(arr::GeneratedArray; kwargs...)
    norm(arr, 2; kwargs...)
end

Base.@propagate_inbounds function LinearAlgebra.norm(arr::GeneratedArray, p::Real)
    if p == 2
        return root.(Sum().(square.(arr)))
    elseif p == 1
        return Sum().(norm.(arr))
    elseif p == Inf
        return Reduce(max).(norm.(arr))
    elseif p == 0
        return Sum().(norm.(norm.(arr), 0))
    elseif p == -Inf
        return Reduce(min).(norm.(arr))
    else
        return root.(Sum().(power.(arr, p)))
    end
end

Base.@propagate_inbounds function distance(x, y)
    return distance(x, y, 2)
end
Base.@propagate_inbounds function distance(x, y, p)
    return norm(arrayify(broadcasted(-, x, y)), p)
end

end
