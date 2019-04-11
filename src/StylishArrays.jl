module StylishArrays

abstract type StylishArray{T, N} <: AbstractArray{T, N} end

using Base.Broadcast: Broadcasted, AbstractArrayStyle, preprocess, BroadcastStyle, DefaultArrayStyle
using Base.Broadcast: instantiate
using LinearAlgebra
using Swizzles
using Swizzles.NullArrays
using Swizzles.ScalarArrays

export StylishArray, myidentity, Styled



struct Styled{Style, Arg}
    arg::Arg
end

@inline Styled{Style}(arg::Arg) where {Style, Arg} = Styled{Style, Arg}(arg)
@inline Styled(arg) = Styled{typeof(BroadcastStyle(typeof(arg)))}(arg)

myidentity(x) = x
@inline Base.similar(src::Styled{Style}, args...) where {Style} = similar(Broadcasted{Style}(identity, (src.arg,), axes(src.arg)), args...)
Base.@propagate_inbounds Base.copy(src::Styled) = copyto!(similar(src), src)
Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray, src::Styled{Style}) where {Style}
    @boundscheck axes(dst) == axes(src.arg) || error("TODO")
    copyto!(dst, Broadcasted{Style}(myidentity, (src.arg,), axes(dst)))
end



"""
    StylishArray <: AbstractArray

A convenience type for defining array types which are implemented using
`similar(::Broadcasted)`, `copyto!(::Any, ::Broadcasted)` and
`copy(::Broadcasted)`. Consult the [Interfaces chapter](@ref
man-interfaces-broadcasting) on broadcasting for more info about broadcast. Many
Base functions are implemented, ` including `map`, `getindex`, and `foreach`.
Power users who wish to avoid broadcasting semantics can also intercept
`copyto!(::Any, ::Styled)`, `copy(::Styled)`, and `similar(::Styled)`

See also: [`similar`](@ref), [`copy`](@ref), [`copyto!`](@ref)
"""
abstract type StylishArray{T, N} <: AbstractArray{T, N} end

#These two are more for intercept purposes than StylishArray purposes.
Base.@propagate_inbounds Base.Broadcast.materialize(A::StylishArray) = identity.(A)
Base.@propagate_inbounds Base.Broadcast.materialize!(dst, A::StylishArray) = dst .= A

#Beware infinite recursion!

Base.@propagate_inbounds Base.similar(src::StylishArray) = similar(Styled(src))

Base.@propagate_inbounds Base.copy(src::StylishArray) = copy(Styled(src))

Base.@propagate_inbounds Base.copyto!(dst::StylishArray, src::AbstractArray) = _copyto!(dst, src)
Base.@propagate_inbounds Base.copyto!(dst::AbstractArray, src::StylishArray) = _copyto!(dst, src)
Base.@propagate_inbounds Base.copyto!(dst::StylishArray, src::StylishArray) = _copyto!(dst, src)
Base.@propagate_inbounds function _copyto!(dst::AbstractArray, src)
    @boundscheck axes(dst) == axes(src) || error("I don't like it when copyto! has mismatched axes lololol")
    #dst .= myidentity.(reshape(arrayify(src), axes(dst)))
    copyto!(dst, Styled(src))
    return dst
end



#The following nonsense means that generated arrays can override getindex or they can override copyto!(view)
Base.@propagate_inbounds Base.getindex(arr::StylishArray, I::Integer)::eltype(arr) = _getindex(arr, I)
Base.@propagate_inbounds Base.getindex(arr::StylishArray, I...) = _getindex(arr, I...)

Base.@propagate_inbounds function _getindex(arr, I...)
    src = Styled(view(arr, I...))
    dst = copyto!(similar(src), src)
    if ndims(dst) == 0
        return dst[]
    else
        return dst
    end
end



#The following nonsense means that stylish arrays can override getindex or they can override copyto!(view)
Base.@propagate_inbounds Base.setindex!(arr::StylishArray, v, I::Integer) = _setindex!(arr, v, I)
Base.@propagate_inbounds Base.setindex!(arr::StylishArray, v, I...) = _setindex!(arr, v, I...)

Base.@propagate_inbounds function _setindex!(arr, v, I...)
    dst = view(arr, I...)
    if ndims(dst) == 0
        copyto!(dst, Styled(ScalarArray(v)))[]
    else
        copyto!(dst, Styled(v))
    end
end



Base.@propagate_inbounds function Base.map(f::F, arr::StylishArray, tail...) where {F}
    src = @_ f.(arr, tail...)
    copyto!(similar(src), src)
end



Base.@propagate_inbounds function Base.foreach(f::F, arr::StylishArray, tail...) where {F}
    src = @_ f(arr, tail...)
    copyto!(NullArray(axes(src)), src)
    return nothing
end



Base.@propagate_inbounds function Base.reduce(op::Op, arr::StylishArray; dims=:, kwargs...) where {Op}
    return _reduce(op, arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _reduce(op::Op, arr::StylishArray, dims, nt::NamedTuple{()}) where {Op}
    res = copy(Reduce(op, dims)(arr))
    return dims == Colon() ? res[] : res
end
Base.@propagate_inbounds function _reduce(op::Op, arr::StylishArray, dims, nt::NamedTuple{(:init,)}) where {Op}
    res = copy(Reduce(op, dims)(nt.init, arr))
    return dims == Colon() ? res[] : res
end



Base.@propagate_inbounds function Base.mapreduce(f::F, op::Op, arr::StylishArray; dims=:, kwargs...) where {F, Op}
    return _mapreduce(f, op, arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _mapreduce(f::F, op::Op, arr::StylishArray, dims, nt::NamedTuple{()}) where {F, Op}
    res = copy(Reduce(op, dims)(@_ f.(arr)))
    return dims == Colon() ? res[] : res
end
Base.@propagate_inbounds function _mapreduce(f::F, op::Op, arr::StylishArray, dims, nt::NamedTuple{(:init,)}) where {F, Op}
    res = copy(Reduce(op, dims)(nt.init, @_ f.(arr)))
    return dims == Colon() ? res[] : res
end



Base.@propagate_inbounds function Base.sum(arr::StylishArray; dims=:, kwargs...)
    return _sum(arr, dims, kwargs.data)
end

Base.@propagate_inbounds function _sum(arr::StylishArray, dims, nt::NamedTuple{()})
    res = copy(Sum(dims)(arr))
    return dims == Colon() ? res[] : res
end
Base.@propagate_inbounds function _sum(arr::StylishArray, dims, nt::NamedTuple{(:init,)})
    res = copy(Sum(dims)(nt.init, arr))
    return dims == Colon() ? res[] : res
end



Base.@propagate_inbounds Base.maximum(arr::StylishArray; kwargs...) = reduce(max, arr; kwargs...)



Base.@propagate_inbounds Base.minimum(arr::StylishArray; kwargs...) = reduce(min, arr; kwargs...)



Base.@propagate_inbounds function LinearAlgebra.dot(x::StylishArray, y::AbstractArray)
    res = copy(Sum()(@_ x .* y))
    return ndims(res) == 0 ? res[] : res
end

Base.@propagate_inbounds function LinearAlgebra.dot(x::AbstractArray, y::StylishArray)
    res = copy(Sum()(@_ x .* y))
    return ndims(res) == 0 ? res[] : res
end

Base.@propagate_inbounds function LinearAlgebra.dot(x::StylishArray, y::StylishArray)
    res = copy(Sum()(@_ x .* y))
    return ndims(res) == 0 ? res[] : res
end



Base.@propagate_inbounds function LinearAlgebra.adjoint(x::StylishArray)
    return Beam(2, 1)(@_ adjoint.(x))
end

Base.@propagate_inbounds function LinearAlgebra.transpose(x::StylishArray)
    return Beam(2, 1)(@_ transpose.(x))
end

Base.@propagate_inbounds function Base.permutedims(x::StylishArray)
    return Beam(2, 1)(x)
end

Base.@propagate_inbounds function Base.permutedims(x::StylishArray, perm)
    return Focus(perm...)(x)
end



Ts = (:(StylishArray{<:Any, 1}), :(StylishArray{<:Any, 2}), :(AbstractArray{<:Any, 1}), :(AbstractArray{<:Any, 2}))
for (A, B) in Iterators.product(Ts, Ts)
    if :(StylishArray{<:Any, 1}) in (A, B) || :(StylishArray{<:Any, 2}) in (A, B)
        @eval begin
            Base.@propagate_inbounds function Base.:*(A::$A, B::$B)
                res = copy(SumOut(2)(@_ A .* Beam(2, 3)(B)))
                return ndims(res) == 0 ? res[] : res
            end
        end
    end
end

for (Y, A, B) in Iterators.product(Ts, Ts, Ts)
    if :(StylishArray{<:Any, 1}) in (Y, A, B) || :(StylishArray{<:Any, 2}) in (Y, A, B)
        @eval begin
            Base.@propagate_inbounds function LinearAlgebra.mul!(Y::$Y, A::$A, B::$B)
                res = copyto!(Y, SumOut(2)(@_ A .* Beam(2, 3)(B)))
                return ndims(res) == 0 ? res[] : res
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

Base.@propagate_inbounds function LinearAlgebra.norm(arr::StylishArray; kwargs...)
    norm(arr, 2; kwargs...)
end

Base.@propagate_inbounds function LinearAlgebra.norm(arr::StylishArray, p::Real)
    if p == 2
        return root(copy(Sum()(@_ square.(arr)))[])
    elseif p == 1
        return copy(Sum()(@_ norm.(arr)))[]
    elseif p == Inf
        return copy(Reduce(max)(@_ norm.(arr)))[]
    elseif p == 0
        return copy(Sum()(@_ norm.(norm.(arr), 0)))[]
    elseif p == -Inf
        return copy(Reduce(min)(@_ norm.(arr)))[]
    else
        return root(copy(Sum()(@_ power.(arr, p)))[])
    end
end

Base.@propagate_inbounds function distance(x, y)
    return distance(x, y, 2)
end
Base.@propagate_inbounds function distance(x, y, p)
    return norm(arrayify(@_ x .- y), p)
end

end
