struct Drop end

const drop = Drop()

Base.isequal(::Drop, ::Drop) = true
Base.isequal(::Drop, ::Int) = false
Base.isequal(::Int, ::Drop) = false
Base.isless(::Drop, ::Int) = true
Base.isless(::Int, ::Drop) = false
Base.isless(::Drop, ::Drop) = false

"""
    getindexinto(a, B, i::Union{Int, Drop})
If `i isa Drop`, return `a`. Otherwise, return `B[i]`.
# Examples
```jldoctest
julia> B = [2; 4; 0; 3; 1];
julia> getindexinto(-1, B, 3)
  0
julia> getindexinto(-1, B, drop)
 -1
```
"""
getindexinto

getindexinto(a, b, i::Int) = b[i]
getindexinto(a, b, i::Drop) = a

"""
    getindexinto(A, B, I)
Return a collection `R` similar to `I` such that `R[j] == B[I[j]] when `I[j] isa
Int` and `R[j] == A[j] when `I[j] isa Drop`.
`A` must have the same length as `I`.
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
getindexinto

getindexinto(a, b, i::Tuple{}) = ()
function getindexinto(a, b, i::Tuple{Vararg{<:Union{Int, Drop}}})
    @boundscheck length(a) == length(i)
    r = ntuple(j ->getindexinto(a[j], b, i[j]), length(i))
    return r
end

function getindexinto(a, b, i::AbstractVector{<:Union{Int, Drop}}) #FIXME wrong output
    @boundscheck length(a) == length(i)
    map((x, j)->getindexinto(x, b, j), a, i)
end

"""
    setindexinto(A, b, i::Union{Int, Drop})
Return a collection `R` similar to `A` such that `R[j] == A[j]` whenever `j != i`, and
`R[i] == b` if `i isa Int`.
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
setindexinto

setindexinto(a, b, i::Int) = ntuple(j -> j == i ? b : a[j], length(a))
setindexinto(a, b, i::Drop) = a

"""
    setindexinto(A, B, I)
Return a collection `R` similar to `A` equivalent to the `R` that would be produced by
  ```
    R = A
    for (b, i) in zip(B, I)
      R = setindexinto(R, b, i)
    end
  ```
`B` must have the same length as `I`.
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
setindexinto

function setindexinto(a, b, i::Tuple{Vararg{Drop}})
    @boundscheck length(b) == length(i)
    return a
end
function setindexinto(a, b, i::Tuple{Int})
    @boundscheck length(b) == 1
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Int, Drop})
    @boundscheck length(b) == 2
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Drop, Int})
    @boundscheck length(b) == 2
    setindexinto(a, b[2], i[2])
end
function setindexinto(a, b, i::Tuple{Int, Int})
    @boundscheck length(b) == 2
    ntuple(j -> j == i[2] ? b[2] : (j == i[1] ? b[1] : a[j]), length(a))
end

function setindexinto(a, b, i::Tuple{Vararg{<:Union{Int, Drop}}})
    @boundscheck length(b) == length(i)
    state = Dict(j => x for (j, x) in zip(i, b))
    ntuple(j -> haskey(state, j) ? state[j] : a[j], length(a))
end

function setindexinto(a, b, i::AbstractVector{<:Union{Int, Drop}}) #FIXME wrong output?
    @boundscheck length(b) == length(i)
    state = Dict(j => x for (j, x) in zip(i, b))
    map((x, j) -> haskey(state, j) ? state[j] : x, a, 1:length(a))
end
