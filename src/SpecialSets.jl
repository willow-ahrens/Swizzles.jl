module SpecialSets

export SpecialSet
export ∅


abstract type SpecialSet{T} <: AbstractSet{T} end
Base.:(==)(a::SpecialSet, b::SpecialSet) = a === b
Base.:(==)(::AbstractSet, ::SpecialSet) = false
Base.:(==)(::SpecialSet, ::AbstractSet) = false
Base.hash(s::SpecialSet, h::UInt) = invoke(hash, Tuple{Any,UInt}, s, zero(UInt))
Base.in(x, ::SpecialSet) = false
Base.issubset(s::SpecialSet, t::SpecialSet) = s ∩ t == s
Base.issubset(::SpecialSet, t) = false
Base.issubset(s, t::SpecialSet) = all(in(t), s)


const ∅ = Set()
Base.intersect(a::AbstractSet, b::SpecialSet) = Set([x for x ∈ a if x ∈ b])
Base.intersect(b::SpecialSet, a::AbstractSet) = Base.intersect(a, b)


include("operations.jl")

include("interval.jl")
include("step.jl")

include("show.jl")

end # module
