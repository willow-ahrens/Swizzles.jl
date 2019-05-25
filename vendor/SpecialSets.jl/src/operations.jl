using Combinatorics: permutations


abstract type Operation <: SpecialSet{Any} end


Base.intersect(x::SpecialSet, xs::SpecialSet...) = foldl(Base.intersect, xs; init=x)
function Base.intersect(a::SpecialSet, b::SpecialSet)
    res = intersect(a, b)
    res == nothing && return _intersect(a, b)
    res
end


"""
    intersect(::SpecialSet, t::SpecialSet) -> Union{SpecialSet, Nothing}

Overloadable method to determine the intersection of two `SpecialSet`s. If no special facts
about the given sets can be deduced, return `nothing`.

!!! note

    The `intersect` function within SpecialSets should never be called directly.
    Instead, call [`Base.intersect`](@ref).
"""
function intersect(::SpecialSet, ::SpecialSet) end


"""
    SetIntersection <: SpecialSet

Represents the intersection of its contained sets. Automatically flattens nested
instances of `SetIntersection`.

!!! warning

    Requires at least two unique `SpecialSet`s to construct. The intersection of one set is
    itself and should be represented as such; the meaning of the intersection of zero sets
    is unclear.
"""
struct SetIntersection <: Operation
    sets::Set{AbstractSet}

    function SetIntersection(sets::AbstractSet...)
        data = _flatten!(Set{AbstractSet}(), sets...)

        isempty(data) && throw(ArgumentError("Unable to construct intersection with no sets"))
        length(data) == 1 && throw(ArgumentError("Intersection with one set is invalid; use $(first(data))"))
        new(data)
    end
end
Base.:(==)(a::SetIntersection, b::SetIntersection) = a.sets == b.sets
Base.get(s::SetIntersection) = s.sets
Base.eltype(s::SetIntersection) = mapfoldl(eltype, typeintersect, get(s))
Base.in(x, s::SetIntersection) = all(set -> x ∈ set, get(s))
function intersect(s::SetIntersection, t::SpecialSet)
    xs = collect(get(s))

    new = nothing
    for (i, q) ∈ enumerate(xs)
        new = intersect(q, t)
        new == nothing || (deleteat!(xs, i); break)
    end
    new == nothing && return SetIntersection(xs..., t)

    result = intersect(_intersect(xs...), new)
    result == nothing && return _intersect(xs..., new)
    result
end
intersect(t::SpecialSet, s::SetIntersection) = intersect(s, t)
intersect(a::SetIntersection, b::SetIntersection) = foldl(intersect, get(b); init=a)
function Base.issubset(a::SetIntersection, b::SetIntersection)
    l = min(length(a.sets), length(b.sets))
    for xs ∈ permutations(collect(a.sets), l), ys ∈ permutations(collect(b.sets), l)
        all(xs .⊆ ys) && return true
    end
    false
end
Base.issubset(a::SetIntersection, b::SpecialSet) = any(s -> s ⊆ b, a.sets)
Base.issubset(a::SpecialSet, b::SetIntersection) = all(s -> a ⊆ s, b.sets)
condition(var, s::SetIntersection) = join([condition(var, set) for set ∈ s.sets], ", ")


function _intersect(xs...)
    uniques = _flatten!(Set(), xs...)
    length(uniques) == 1 && return first(uniques)
    SetIntersection(uniques...)
end
function _flatten!(data, s::SetIntersection)
    for el ∈ get(s)
        push!(data, el)
    end
    data
end
_flatten!(data, x) = push!(data, x)
_flatten!(data, xs...) = foldl(_flatten!, xs; init=data)
