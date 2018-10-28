struct Drop end

const drop = Drop()

Base.isequal(::Drop, ::Drop) = true
Base.isequal(::Drop, ::Integer) = false
Base.isequal(::Integer, ::Drop) = false
Base.isless(::Drop, ::Integer) = true
Base.isless(::Integer, ::Drop) = false
Base.isless(::Drop, ::Drop) = false

"""
    getindexinto(a, B, i::Union{Integer, Drop})
If `i isa Drop`, return `a`. If `i isa Integer`, return `B[i]`.
# Examples
```jldoctest
julia> B = [2; 4; 0; 3; 1];
julia> getindexinto(-1, B, 3)
  0
julia> getindexinto(-1, B, drop)
 -1
```
"""
getindexinto(a, B, i::Union{Integer, Drop}) = _scalar_getindexinto(a, B, i)

_scalar_getindexinto(a, B, i::Integer) = B[i]
_scalar_getindexinto(a, B, i::Drop) = a

"""
    getindexinto(A, B, I)
Return a collection `R` similar to `I` such that `R[j] == A[j] when `I[j] isa
Drop` and `R[j] == B[I[j]]` otherwise.
# Examples
```jldoctest
julia> A = [-1; -2; -3; -4; -5];
julia> B = [11; 12; 13; 14; 15];
julia> I = [2; 4; drop; 3; 1];
julia> getindexinto(A, B, I)
5-element Array{Int64,1}:
  12
  14
 -3
  13
  11
julia> getindexinto(A, B, (2, 4, drop, 3, 1))
  (12, 14, -3, 13, 11)
```
"""
getindexinto(A, B, I::Union{Tuple, AbstractVector}) = _vector_getindexinto(A, B, I)

_vector_getindexinto(A, B, I::Tuple{}) = ()
_vector_getindexinto(A, B, I::Tuple) = ntuple(j -> I[j] isa Drop ? A[j] : B[I[j]], length(I))
function _vector_getindexinto(A, B, I::AbstractVector)
    R = similar(I, Any)
    for j in eachindex(I)
        i = I[j]
        R[j] = i isa Drop ? A[j] : B[i]
    end
    return map(identity, R)
end

"""
    setindexinto(A, b, i::Union{Integer, Drop})
If `i isa Drop`, return a collection `R` similar to `A` where `R[j] == A[j]`
everywhere. If `i isa Integer`, return a collection `R` similar to `A` such
that `R[i] == b` and `R[j] = A[j]` everywhere else.
# Examples
```jldoctest
julia> A = [2; 4; 0; 3; 1];
julia> setindexinto(A, -1, 3)
5-element Array{Int64,1}:
  2
  4
 -1
  3
  1
julia> setindexinto(A, -1, drop)
5-element Array{Int64,1}:
  2
  4
  0
  3
  1
julia> A = (2, 4, 0, 3, 1);
julia> setindexinto(A, -1, 2)
  (2, -1, 0, 3, 1)
julia> setindexinto(A, -1, drop)
  (2, 4, 0, 3, 1)
```
"""
setindexinto(A::Union{Tuple, AbstractVector}, b, i::Union{Integer, Drop}) = _scalar_setindexinto(A, b, i)

@inline function _scalar_setindexinto(A::Tuple, b, i::Integer)
    @boundscheck A[i]
    ntuple(j -> j == i ? b : A[j], length(A))
end
@inline _scalar_setindexinto(A::Tuple, b, i::Drop) = A
@inline _scalar_setindexinto(A::AbstractVector, b, i::Drop) = map(identity, A)
function _scalar_setindexinto(A::AbstractVector, b, i::Integer)
    R = similar(A, Any)
    copyto!(R, A)
    R[i] = b
    return map(identity, R)
end

"""
    setindexinto(A, B, I)
Return a collection `R` similar to `A` such that `R[I[j]] == B[j]` whenever
`!(I[j] isa Drop)`, and `R[i] = A[i]` otherwise.
# Examples
```jldoctest
julia> A = [2; 4; 0; 3; 1];
julia> setindexinto(A, (-1, -2), (drop, 3))
5-element Array{Int64,1}:
  2
  4
 -2
  3
  1
julia> setindexinto(A, (-1, -2), (3, 3))
5-element Array{Int64,1}:
  2
  4
 -2
  3
  1
julia> A = (2, 4, 0, 3, 1);
julia> setindexinto(A, (-1, -2), (drop, 3))
  (2, 4, -2, 3, 1)
```
"""
setindexinto(A::Union{Tuple, AbstractVector}, B, I) = _vector_setindexinto(A, B, I)

_vector_setindexinto(A::Tuple, B, I::Tuple{Vararg{Drop}}) = A
_vector_setindexinto(A::Tuple, B, I::AbstractVector{Drop}) = A
_vector_setindexinto(A::AbstractVector, B, I::Tuple{Vararg{Drop}}) = map(identity, A)
_vector_setindexinto(A::AbstractVector, B, I::AbstractVector{Drop}) = map(identity, A)
function _vector_setindexinto(A::Tuple, B, I::Tuple{Integer})
    @boundscheck A[I[1]]
    ntuple(j -> j == I[1] ? B[1] : A[j], length(A))
end
function _vector_setindexinto(A::Tuple, B, I::Tuple{Integer, Drop})
    @boundscheck A[I[1]]
    ntuple(j -> j == I[1] ? B[1] : A[j], length(A))
end
function _vector_setindexinto(A::Tuple, B, I::Tuple{Drop, Integer})
    @boundscheck A[I[2]]
    ntuple(j -> j == I[2] ? B[2] : A[j], length(A))
end
function _vector_setindexinto(A::Tuple, B, I::Tuple{Integer, Integer})
    @boundscheck A[max(I[1], I[2])]
    ntuple(j -> j == I[2] ? B[2] : (j == I[1] ? B[1] : A[j]), length(A))
end
_vector_setindexinto(A::Tuple, B, I) = (_vector_setindexinto([A...,], B, I)...,)
function _vector_setindexinto(A::AbstractVector, B, I)
    R = similar(A, Any)
    copyto!(R, A)
    for j in eachindex(I)
        i = I[j]
        if !isa(i, Drop)
            R[i] = B[j]
        end
    end
    return map(identity, R)
end



using Base.Broadcast: broadcasted, BroadcastStyle, Broadcasted

struct Unwrap
  value
end

Base.Broadcast.broadcasted(::BroadcastStyle, ::Type{Unwrap}, bc) = Unwrap(bc)

Base.Broadcast.materialize(uw::Unwrap) = uw.value
