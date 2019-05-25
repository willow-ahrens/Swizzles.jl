"""
    setname(::Type) -> String

Generate the display name for a given type using set notation.
"""
setname(T) = string(T)
setname(::Type{Real}) = "ℝ"
setname(::Type{Int}) = "ℤ"


"""
    condition(var, set::SpecialSet) -> String

Generate the display name for the condition provided by a given set.
"""
condition(var, set::SpecialSet) = "x ∈ " * sprint(Base.show_default, set)


function Base.show(io::IO, s::SpecialSet)
    x = :x
    T = eltype(s)
    els = T == Any ? "$x" : "$x ∈ $(setname(T))"
    print(io, "{$els | ", condition(x, s), "}")
end
