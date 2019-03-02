module Powers

using LinearAlgebra
export Power, ScaledPower
export power, value, scale

struct Power{T, E} <: Number
    arg::T
    exponent::E
end

power(x, p) = Power(x, p)
power(x::Power, p) = Power(x.arg, x.exponent + p)

value(x::Power) = x.arg ^ p

function Base.promote_type(::Type{Power{T1, E1}}, ::Type{Power{T2, E2}}) where {T1, E1, T2, E2}
    return Power{promote_type{T1, E1}, promote_type{T2, E2}}
end

function Base.convert(::Type{Power{T, E}}, x::Power) where {T, E}
    return Power(convert(T, x.arg), convert(E, x.exponent))
end

struct ScaledPower{T, S, E} <: Number
    arg::T
    scale::S
    exponent::E
end

power(x::ScaledPower, p) = ScaledPower(x.arg^p, x.scale, x.exponent + p)

function rescale(x::Power)
    ScaledPower(sign(x.arg), norm(x.arg), x.exponent)
end

value(x::ScaledPower) = x.arg * (x.scale ^ p)

function Base.promote_type(::Type{ScaledPower{T1, S1, E1}}, ::Type{ScaledPower{T2, S2, E2}}) where {T1, S1, E1, T2, S2, E2}
    return ScaledPower{promote_type(T1, T2), promote_type(S1, S2), promote_type(E1, E2)}
end

function Base.convert(::Type{ScaledPower{T, S, E}}, x::ScaledPower) where {T, S, E}
    return Power(convert(T, x.arg), convert(S, x.scale), convert(E, x.exponent))
end

function Base.promote_type(::Type{ScaledPower{T1, S1, E1}}, ::Type{Power{T2, E2}}) where {T1, S, E1, T2, E2}
    return ScaledPower{promote_type{T1, T1}, S, promote_type{E2, E2}}
end

Base.convert(::Type{T}, x::Power) where {T <:ScaledPower} = convert(T, rescale(x))



function Base.:+(x::T, y::T) where {T <: Power}
    return rescale(x) + rescale(y)
end

@inline function Base.:+(x::T, y::T) where {T <: ScaledPower}
    if x.exponent < y.exponent
        (x, y) = (y, x)
    end
    diff = (x.exponent - y.exponent)
    if iszero(y.arg)
        y = ScaledPower(y.arg^one(diff), y.scale, x.exponent)
    else
        y = ScaledPower(y.arg^diff, y.scale, x.exponent)
    end
    x = ScaledPower(x.arg^one(diff), x.scale, x.exponent)

    if x.scale < y.scale
        (x, y) = (y, x)
    end
    if iszero(y.scale)
        arg = x.arg + zero(y.arg) * (x.scale/one(y.scale))^x.exponent
    else
        arg = x.arg + y.arg * (x.scale/y.scale)^x.exponent
    end

    return ScaledPower(arg, x.scale, x.exponent)
end

end
