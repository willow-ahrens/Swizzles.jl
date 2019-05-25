export TypeSet
export LessThan, GreaterThan
export NotEqual

export Negative, Nonpositive
export Positive, Nonnegative
export Zero, Nonzero


abstract type Interval{T} <: SpecialSet{T} end


struct TypeSet{T} <: Interval{T} end
TypeSet(T::Type) = TypeSet{T}()
Base.in(::T, ::TypeSet{T}) where {T} = true
function intersect(::TypeSet{T}, ::TypeSet{U}) where {T, U}
    V = typeintersect(T, U)
    V === Union{} && return ∅
    TypeSet{V}()
end
intersect(::TypeSet{T}, s::SpecialSet) where {T} = eltype(s) <: T ? s : ∅
intersect(s::SpecialSet, t::TypeSet) = intersect(t, s)
Base.issubset(s::AbstractSet, ::TypeSet{T}) where {T} = eltype(s) <: T
Base.issubset(s::SpecialSet, ::TypeSet{T}) where {T} = eltype(s) <: T
Base.issubset(s::SetIntersection, ::TypeSet{T}) where {T} = eltype(s) <: T
condition(var, ::TypeSet{T}) where {T} = "$var ∈ $(setname(T))"



struct LessThan{T} <: Interval{T}
    value::T
    inclusive::Bool
end
LessThan{T}(value) where {T} = LessThan{T}(value, false)
LessThan(value) = LessThan{typeof(value)}(value, false)
Base.convert(::Type{LessThan{T}}, s::LessThan{U}) where {U,T<:U} =
    LessThan{T}(convert(T, s.value), s.inclusive)
Base.in(x::T, s::LessThan{T}) where {T} = x < s.value || (x == s.value) & s.inclusive
function intersect(a::LessThan{T}, b::LessThan{T}) where {T}
    lt = a.value == b.value ? !a.inclusive & b.inclusive : a.value < b.value
    ifelse(lt, a, b)
end
function intersect(a::LessThan{T}, b::LessThan{U}) where {T,U}
    V = typeintersect(T, U)
    V === Union{} && return ∅
    intersect(convert(LessThan{V}, a), convert(LessThan{V}, b))
end
Base.issubset(a::LessThan{T}, b::LessThan{U}) where {U,T<:U} =
    a.value == b.value ? (a.inclusive ≤ b.inclusive) : a.value < b.value
function condition(var, s::LessThan)
    sign = s.inclusive ? '≤' : '<'
    "$var $sign $(s.value)"
end


struct GreaterThan{T} <: Interval{T}
    value::T
    inclusive::Bool
end
GreaterThan{T}(value) where {T} = GreaterThan{T}(value, false)
GreaterThan(value) = GreaterThan{typeof(value)}(value, false)
Base.convert(::Type{GreaterThan{T}}, s::GreaterThan{U}) where {U,T<:U} =
    GreaterThan{T}(convert(T, s.value), s.inclusive)
Base.in(x::T, s::GreaterThan{T}) where {T} = x > s.value || (x == s.value) & s.inclusive
function intersect(a::GreaterThan{T}, b::GreaterThan{T}) where {T}
    gt = a.value == b.value ? !a.inclusive & b.inclusive : a.value > b.value
    ifelse(gt, a, b)
end
function intersect(a::GreaterThan{T}, b::GreaterThan{U}) where {T,U}
    V = typeintersect(T, U)
    V === Union{} && return ∅
    intersect(convert(GreaterThan{V}, a), convert(GreaterThan{V}, b))
end
Base.issubset(a::GreaterThan{T}, b::GreaterThan{U}) where {U,T<:U} =
    a.value == b.value ? (a.inclusive ≤ b.inclusive) : a.value > b.value
function condition(var, s::GreaterThan)
    sign = s.inclusive ? '≥' : '>'
    "$var $sign $(s.value)"
end


function intersect(a::LessThan{T}, b::GreaterThan{U}) where {T, U}
    V = typeintersect(T, U)
    V == Union{} && return ∅

    a, b = convert(LessThan{V}, a), convert(GreaterThan{V}, b)

    result = intersect(a, b)
    result == nothing && return SetIntersection(a, b)
    result
end
function intersect(a::LessThan{T}, b::GreaterThan{T}) where {T}
    gt = a.value == b.value ? !a.inclusive & b.inclusive : a.value < b.value
    gt && return ∅

    a.value == b.value && return (a.inclusive & b.inclusive) ? Set([a.value]) : ∅

    nothing
end
intersect(b::GreaterThan, a::LessThan) = intersect(a, b)


struct NotEqual{T} <: SpecialSet{Any}
    values::Set{T}
    NotEqual{T}() where {T} = throw(ArgumentError("No elements provided to NotEqual; use TypeSet{$T}"))
    NotEqual{T}(xs...) where {T} = new{T}(Set{T}(xs))
end
NotEqual(xs...) = NotEqual{typejoin(typeof.(xs)...)}(xs...)
Base.:(==)(a::NotEqual, b::NotEqual) = a.values == b.values
Base.hash(s::NotEqual, h::UInt) = hash(s.values, hash(typeof(s), h))
Base.in(x, s::NotEqual) = x ∉ s.values
intersect(a::NotEqual{T}, b::NotEqual{U}) where {T,U} = NotEqual{typejoin(T,U)}(a.values..., b.values...)
function intersect(a::NotEqual{T}, b::SpecialSet) where {T}
    keep = Set{T}()
    for value ∈ a.values
        value ∈ b && push!(keep, value)
    end
    length(keep) == length(a.values) && return nothing
    isempty(keep) && return b
    SetIntersection(NotEqual{T}(keep...), b)
end
intersect(a::NotEqual, b::SetIntersection) = invoke(intersect, Tuple{typeof(a),SpecialSet}, a, b)
intersect(a::NotEqual, b::TypeSet) = invoke(intersect, Tuple{typeof(a), SpecialSet}, a, b)
intersect(b::SpecialSet, a::NotEqual) = intersect(a, b)
intersect(b::SetIntersection, a::NotEqual) = intersect(a, b)
intersect(b::TypeSet, a::NotEqual) = invoke(intersect, Tuple{typeof(a), SpecialSet}, a, b)
Base.issubset(a::NotEqual, b::NotEqual) = b.values ⊆ a.values
Base.issubset(a::SpecialSet, b::NotEqual) = !any(in(a), b.values)
Base.issubset(a::SetIntersection, b::NotEqual) = invoke(issubset, Tuple{SetIntersection,SpecialSet}, a, b)
function condition(var, s::NotEqual)
    length(s.values) == 1 && return "$var ≠ $(first(s.values))"
    "$var ∉ {$(join(s.values, ", "))}"
end


"""
    Negative = LessThan{Number}(0)

Negative numbers. ``{x | x < 0}``
"""
const Negative = LessThan{Number}(0)

"""
    Nonpositive = LessThan{Number}(0, true)

Nonpositive numbers. ``{x | x ≤ 0}``
"""
const Nonpositive = LessThan{Number}(0, true)

"""
    Positive = GreaterThan{Number}(0)

Positive numbers. ``{x | x > 0}``
"""
const Positive = GreaterThan{Number}(0)

"""
    Nonnegative = GreaterThan{Number}(0, true)

Nonnegative numbers. ``{x | x ≥ 0}``
"""
const Nonnegative = GreaterThan{Number}(0, true)

"""
    Zero = Set{Number}([0])

Numbers equivalent to zero. ``{0}``
"""
const Zero = Set{Number}([0])

"""
    Nonzero = NotEqual(0)

Nonzero numbers. ``{x | x ≠ 0}``
"""
const Nonzero = NotEqual(0)
