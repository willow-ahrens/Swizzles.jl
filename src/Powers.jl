module Powers

using LinearAlgebra
export Power
export power, root

struct Power{T, S, E::Real} <: Number
    arg::T
    scale::S
    exponent::E
end

@inline power(x, p) = Power(sign(x), norm(x), p)

@inline power(x::Power, p)
    if iszero(x.arg)
        Power(zero(x.arg) ^ one(p), zero(x.scale), x.exponent * p)
    end
    Power(x.arg ^ p, x.scale, x.exponent * p)
end

@inline root(x::Power) = x.arg ^ inv(x.exponent) * x.scale

@inline function rescale(x::Power, s)
    if iszero(s)
        return Power(zero(y.arg) * (one(s)/one(x.scale))^one(x.exponent), s, x.exponent)
    else
        return Power(x.arg * (s/x.scale)^x.exponent, s, x.exponent)
    end
end

function Base.promote_type(::Type{Power{T1, S1, E1}}, ::Type{Power{T2, S2, E2}}) where {T1, S1, E1, T2, S2, E2}
    return Power{promote_type(T1, T2), promote_type(S1, S2), promote_type(E1, E2)}
end

function Base.convert(::Type{Power{T, S, E}}, x::Power) where {T, S, E}
    return Power(convert(T, x.arg), convert(S, x.scale), convert(E, x.exponent))
end

@inline function Base.:+(x::T, y::T) where {T <: Power}
    x.exponent == y.exponent || ArgumentError("Cannot accurately add Powers with different exponents")
    p = x.exponent
    if x.scale < y.scale
        (x, y) = (y, x)
    end
    s = x.scale
    y = rescale(y, s)
    return Power(x.arg + y.arg, s, p)
end

end
