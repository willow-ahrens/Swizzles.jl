module Powers

using LinearAlgebra
export Power
export power, root

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
