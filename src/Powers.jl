module Powers

using LinearAlgebra
export Power, Square
export power, root, square

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

struct Square{T, S} <: Number
    arg::T
    scale::S
end

@inline square(x) = Square(sign(x)^2, norm(x))

@inline root(x::Square) = sqrt(x.arg) * x.scale
@inline Base.sqrt(x::Square) = root(x)

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

end
