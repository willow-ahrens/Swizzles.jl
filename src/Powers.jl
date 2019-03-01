


#=
struct Square{T, E} <: Real
    arg::T
    exponent::E
end

power(x::Power, p) = Power(x.arg, x.exponent + p)

value(x::Power) = x.arg ^ p

function Base.promote_type(::Type{Power{T1, E1}}, ::Type{Power{T2, E2}}) where {T1, E1, T2, E2}
    return Power{promote_type{T1, E1}, promote_type{T2, E2}}
end

function Base.convert(::Type{Power{T, E}}, x::Power) where {T, E}
    return Power(convert(T, x.arg), convert(E, x.exponent))
end

struct ScaledPower{T, S, E} <: Real
    arg::T
    scale::S
    exponent::E
end

power(x::ScaledPower, p) = ScaledPower(x.arg^p, x.scale, x.exponent + p)

function scale(x::Power)
    xabs = norm(x.arg)
    ScaledPower(x.arg/xabs, xabs, x.exponent)
end

value(x::ScaledPower) = x.arg * (x.scale ^ p)

function Base.promote_type(::Type{ScaledPower{T1, S1, E1}}, ::Type{ScaledPower{T2, S2, E2}}) where {T1, S1, E1, T2, S2, E2}
    return Power{promote_type{T1, S1, E1}, promote_type{T2, S2, E2}}
end

function Base.convert(::Type{ScaledPower{T, S, E}}, x::ScaledPower) where {T, S, E}
    return Power(convert(T, x.arg), convert(S, x.scale), convert(E, x.exponent))
end
=#



#=
function Base.:+(x::T, y::T) where {T <: Power}
    return scale(x) + scale(y)
end

function Base.:+(x::T, y::T) where {T <: ScaledPower}
    return ScaledPower(x.arg + y.arg, max(x.scale, y.scale), )
end

    return ScaledPower(one(x.arg),
    rescale(Power(x.arg, p)) + Power(y.arg, p))
end

function Base.:+(x::ScaledPower, y::Power)
    if y.arg != zero(y.arg)
        ay = abs(y.arg)
        if x.scale < ay
            arg = one(T) + x.arg * (x.scale/ay)^x.exponent
            scale = ay
        else
            arg = x.arg + (ay/x.scale)^x.exponent
            scale = x.scale
        end
    end
    return ScaledPower(arg, scale, x.exponent)
end
=#



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
