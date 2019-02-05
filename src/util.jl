using StaticArrays: SVector, setindex

struct Drop end

const drop = Drop()

Base.isequal(::Drop, ::Drop) = true
Base.isequal(::Drop, ::Integer) = false
Base.isequal(::Integer, ::Drop) = false
Base.isless(::Drop, ::Integer) = true
Base.isless(::Integer, ::Drop) = false
Base.isless(::Drop, ::Drop) = false

struct Keep end

const keep = Keep()

Base.isequal(::Keep, ::Keep) = true
Base.isequal(::Keep, ::Integer) = false
Base.isequal(::Integer, ::Keep) = false
Base.isless(::Keep, ::Integer) = false
Base.isless(::Integer, ::Keep) = true
Base.isless(::Keep, ::Keep) = false

Base.isless(::Drop, ::Keep) = true
Base.isless(::Keep, ::Drop) = false

"""
    masktuple(f, g, I)
Return `R::Tuple` such that `R[j] == f(j) when `I[j] isa
Drop` and `R[j] == g(I[j])` otherwise.
# Examples
```jldoctest
julia> A = [-1; -2; -3; -4; -5];
julia> B = [11; 12; 13; 14; 15];
julia> I = (2; 4; drop; 3; 1);
julia> masktuple(i->A[i], j->B[i], I)
  (12, 14, -3, 13, 11)
```
"""
@inline function masktuple(f, g, I::Tuple{Vararg{Union{Int, Drop}}})
    ntuple(j -> I[j] isa Drop ? f(j) : g(I[j]), length(I))
end
@generated function masktuple(f, g, ::Val{I}) where {I}
    return quote
        Base.@_inline_meta
        ($(masktuple(j->:(f($j)), j->:(g($j)),I)...),)
    end
end

"""
    imasktuple(f, g, I, n)
Return `R::Tuple` of length `max(0, I...)` such that `R[I[j]] == g(j)` whenever
`!(I[j] isa Drop)`, and `R[i] = f[i]` otherwise.
# Examples
```jldoctest
julia> A = (2, 4, 0, 3, 1);
julia> imasktuple(A, (-1, -2), (drop, 3))
  (2, 4, -2, 3, 1)
```
"""
@inline function imasktuple(f, g, I::Tuple{Vararg{Union{Int, Drop}}})
    ntuple(j-> (k = findfirst(isequal(j), I)) === nothing ? f(j) : g(k), max(0, I...))
end
@generated function imasktuple(f, g, ::Val{I}) where {I}
    return quote
        Base.@_inline_meta
        ($(imasktuple(i->:(f($i)), i->:(g($i)),I)...),)
    end
end

using Base.Broadcast: broadcasted, BroadcastStyle, Broadcasted

struct Delay
  value
end

Delay() = Delay(nothing)

Base.Broadcast.broadcasted(::BroadcastStyle, ::Delay, bc) = Delay(bc)

Base.Broadcast.materialize(uw::Delay) = uw.value

abstract type Intercept end

@inline Base.Broadcast.broadcasted(style::BroadcastStyle, intr::Intercept, args...) = intr(args...)
