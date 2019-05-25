export Step, Even, Odd


"""
    Step{T} <: SpecialSet{T}

`Step{T}(m::T, a::T = zero(T))` contains all x::T such that ``x ≡ a (mod m)``, alternatively
represented as ``x = kn + a``.
"""
struct Step{T} <: SpecialSet{T}
    m::T
    a::T
    Step{T}(m, a) where {T} = new{T}(m, mod(a, m))
end
Step(m::M, a::A) where {M,A} = Step{promote_type(M,A)}(m, a)
Step{T}(m) where {T} = Step{T}(convert(T, m), zero(T))
Step(m) = Step{typeof(m)}(m, zero(m))
Base.eltype(::Step{T}) where {T} = T
Base.in(x::T, s::Step{T}) where {T} = mod(x, s.m) == s.a
function intersect(s::Step{T}, t::Step{T}) where {T<:Integer}
    res = combine_modular_equations(s, t)
    res == nothing && return ∅
    m, a = res
    Step{T}(m, a)
end
Base.issubset(x::Step{T}, y::Step{T}) where {T} = mod(x.m, y.m) == 0 && mod(x.a, y.m) == y.a
condition(var, s::Step) = "($var ≡ $(s.a) (mod $(s.m)))"

"""
    Even = Step(2, 0)

Even integers.
"""
const Even = Step(2, 0)
"""
    Odd = Step(2, 0)

Odd integers.
"""
const Odd  = Step(2, 1)


"""
    combine_modular_equations(equations::Tuple{T,T}...) -> Union{Tuple{T,T}, Nothing}

Using an adaptation of the Chinese Remainder Theorem which supports non-coprime moduli,
combine modular arithmetic equivalences into a single equivalence.

Given modular arithmetic equivalences ``x ≡ a (mod m)``,
`combine_modular_equations((m₁, a₁), (m₂, a₂), ..., (mₙ, aₙ))`
will generate the combined equivalence ``x ≡ a₀ (mod m₀)`` as `(m₀, a₀)`.

In the case that no solutions exist, `nothing` is returned.

# Examples
```jldoctest
julia> combine_modular_equations((2, 0), (3, 0))
(6, 0)

julia> combine_modular_equations((2, 0), (4, 0))
(4, 0)

julia> combine_modular_equations((2, 0), (4, 1))

julia> combine_modular_equations((2, 0), (3, 1), (7, 3))
(42, 10)
```
"""
combine_modular_equations(eqn, eqns...) = foldl(eqns; init=eqn) do (m1, a1), (m2, a2)
    g = gcd(m1, m2)
    mod(a1, g) == mod(a2, g) || return nothing
    p, q = bézout_coefficients(m1 ÷ g, m2 ÷ g)

    m = lcm(m1, m2)
    m, mod(a1*(m2÷g)*q + a2*(m1÷g)*p, m)
end
combine_modular_equations(steps::Step...) =
    combine_modular_equations(((step.m, step.a) for step ∈ steps)...)


function bézout_coefficients(a, b)
    s, s′ = 1, 0
    t, t′ = 0, 1

    while b != 0
        q = a ÷ b
        a, b = b, a % b
        s, s′ = s′, s - q * s′
        t, t′ = t′, t - q * t′
    end

    s, t
end
