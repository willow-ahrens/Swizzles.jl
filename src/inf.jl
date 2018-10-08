struct Skip end

const skip = Skip()

getindexinto(a, b, i::Int) = b[i]
getindexinto(a, b, i::Skip) = a

setindexinto(a, b, i::Int) = map((j, x) -> j == i ? b : x, enumerate(a))
setindexinto(a, b, i::Skip) = a

getindexinto(a, b, i::Tuple{}) = ()
getindexinto(a, b, i::Union{NTuple{<:Any, <:Union{Int, Skip}}, AbstractVector{<:Union{Int, Skip}}}) = map((a, j)->getindex(a, b, j), a, j)

setindexinto(a, b, i::Union{NTuple{<:Any, Skip}, AbstractVector{Skip}}) = @boundscheck length(b) == length(i)
function setindexinto(a, b, i::Tuple{Int}) =
  @boundscheck length(b) == 1
  setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Int, Skip}) =
  @boundscheck length(b) == 2
  setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Skip, Int}) =
  @boundscheck length(b) == 2
  setindexinto(a, b[2], i[2])
end
function setindexinto(a, b, i::Tuple{Int, Int}) =
  @boundscheck length(b) == 2
  map((j, x) -> j == i[2] ? b[2] : (j == i[1] ? b[1] : x), enumerate(a))
end
function setindexinto(a, b, i::Union{NTuple{<:Any, <:Union{Int, Skip}}, AbstractVector{<:Union{Int, Skip}}}) =
  @boundscheck length(b) == length(i)
  state = Dict(j => x for (j, x) in zip(i, b))
  map((j, x) -> haskey(state, j) ? state[j] : x, enumerate(a))
end


# Basic Functions for working with swizzles

"""
    is_swizzle_mask(s) -> Bool
Return `true` if `s` is a valid swizzle mask.
# Examples
```jldoctest
julia> is_swizzle_mask([1; 3])
true
julia> is_swizzle_mask([1; 1])
false
julia> is_swizzle_mask([1; 0; 0; 2])
true
julia> is_swizzle_mask([1; 0; 0; 1])
false
```
"""
is_swizzle_mask

function is_swizzle_mask(s::AbstractVector)
    return allunique(filter(!isequal(0), s)) && all(s .>= 0)
end

is_swizzle_mask(s::Tuple{}) = true
is_swizzle_mask(s::Tuple{Int}) = s[1] >= 0
is_swizzle_mask(s::Tuple{Int,Int}) = s[1] >= 0 && s[2] >= 0 && s[1] != s[2]

"""
    swizzle_elems(v, v0, w, w0, s)
Swizzle `w` into `v` according to the mask `s` and return the resulting
collection `r`.  `v` and `w` are treated as infinitely long sequences which end
with repeating values of `v0` and `w0`, respectively. `r` should be similar to
`v` and satisfy the following properties:
  `r[i] === w[s[i]]` when `1 <= s[i] <= length(w)`
  `r[i] === w0` when `s[i] > length(w)`
  `r[i] === v[i]` when `s[i] == 0` and `i <= length(v)`
  `r[i] === v0` when `s[i] == 0` and `i > length(v)`
No checking is done to determine whether `s` is a valid swizzle mask.
# Examples
```jldoctest
julia> s = [2; 4; 0; 3; 0];
julia> v = [5; 6; 7];
julia> w = [-1; -2; -3];
julia> swizzle_elems(v, 8, w, -4, s)
5-element Array{Int64,1}:
 -2
 -4
 7
 -3
 8
```
"""
swizzle_elems

function swizzle_elems(v::AbstractVector, v0, w, w0, s)
    v! = similar(v, Union{eltype(v), typeof(v0), eltype(w), typeof(w0)})
    v![1:length(v)] = v
    v[(length(v) + 1):end] = v0
    return swizzle_elems!(v!, w, w0, s)
end

@inline function swizzle_elems(v::AbstractVector, v0, w, w0, s::NTuple{N, Int}) where {N}
    w isa AbstractVector && @assert !Base.has_offset_axes(w)
    ntuple(i -> (s[i] == 0) ? (i <= length(v) ? v[i] : v0) : (s[i] <= length(w) ? w[s[i]] : w0), length(s))
end

"""
    swizzle_elems!(a, b, s)
Swizzle `b` into mutable `a` according to the mask `s` and return `a`.  The
final state of `a` should satisfy `a[i] = b[s[i]]` when `s[i] != 0` and `a[i]`
should be unmodified when `s[i] == 0` or when `i > length(s)`. No checking is
done to determine whether `s` is a valid mask. 
`maximum(s)`.
# Examples
# Examples
```jldoctest
julia> s = [2; 4; 0; 3; 1];
julia> a = [5; 5; 5; 5];
julia> b = [-1; -2; -3; -4];
julia> swizzle_elems!(a, b, s); a
4-element Array{Int64,1}:
 -2
 -4
 5
 -3
julia> a = [5; 5; 5; 5; 5; 5];
julia> swizzle_elems!(a, b, s); a
6-element Array{Int64,1}:
 -2
 -4
 5
 -3
 -1
 5
```
```
"""
function swizzle_elems!(r::AbstractVector, v, v0, s::AbstractVector{Int}) where {M, T, T0, S <: Union{T, T0}}
    @assert !Base.has_offset_axes(v)
    @boundscheck @assert 0 <= maximum(s)
    @boundscheck @assert length(s) == length(r)
    @inbounds for (si, i) in enumerate(s)
        r[i] = (si == 0 || si > length(v)) ? v0 : v[si]
    end
    r
end

"""
    invert_swizzle_mask(s)
Return the mask `t` of size `m == maximum(s)` such that
```
  swizzle_elems(s, 0, t) = [i in s ? i : 0 for i = 1:m]
```

It is helpful to think of this operation as producing a pseudoinverse of the
swizzle mask.  For example, if we have some collection `v` and compute `w =
swizzle_elems(v, nothing, s)`, then `swizzle_elems(w, nothing, t)` will
reconstruct as much of the original v as possible. Does not always detect when
it has been given an invalid swizzle mask.
# Examples
```jldoctest
julia> s = [0; 0; 3; 1];
julia> t = invert_swizzle_mask(s, 4)
3-element Array{Int64,1}:
 4
 0
 3
 0
julia> swizzle_elems(s, 0, t)
3-element Array{Int64,1}:
 1
 0
 3
 0
julia> v = [1.0, 2.0, 3.0, 4.0]
4-element Array{Int64,1}:
 1.0
 2.0
 3.0
 4.0
julia> w = swizzle_elems(v, nothing, s)
4-element Array{Union{Int64, Nothing},1}:
 nothing
 nothing
 3.0
 1.0
julia> swizzle_elems(w, nothing, t)
4-element Array{Union{Int64, Nothing},1}:
 1.0
 nothing
 3.0
 nothing
```
"""
function invert_swizzle_mask(s::AbstractVector{Int})
    @assert !Base.has_offset_axes(s)
    @assert is_swizzle_mask(s)
    m=maximum(s)
    t = fill!(similar(s, m), 0) # similar vector of zeros
    @inbounds for (i, si) in enumerate(s)
        (si == 0 || ((si >= 1) && (si <= m || t[si] == 0))) ||
            throw(ArgumentError("argument is not a swizzle mask"))
        t[si] = i
    end
    t
end
function invert_swizzle_mask(s::Tuple{})
    m=maximum(s)
    ntuple(0, m)
end
function invert_swizzle_mask(s::Tuple{Int})
    m=maximum(s)
    if s[1] == 0
        return ntuple(0, m)
    elseif s[1] >= 1
        return ntuple(i -> i == s[1] ? 1 : 0, m)
    else
        throw(ArgumentError("argument is not a swizzle mask"))
    end
end
function invert_swizzle_mask(s::Tuple{Int, Int})
    m=maximum(s)
    if s[1] == 0 && s[2] == 0
        return ntuple(0, m)
    elseif (s[1] >= 1 && s[2] == 0)
        return ntuple(i -> i == s[1] ? 1 : 0, m)
    elseif (s[2] >= 1 && s[1] == 0)
        return ntuple(i -> i == s[2] ? 2 : 0, m)
    elseif (s[1] >= 1 && s[2] >= 1 && s[1] != s[2])
        return ntuple(i -> i == s[1] ? 1 : (i == s[2] ? 2 : 0), m)
    else
        throw(ArgumentError("argument is not a swizzle mask"))
    end
end

invert_swizzle_mask(s::Tuple) = (invert_swizzle_mask([s...])...,)

"""
    strip_swizzle_mask(s)
Return the smallest equivalent swizzle mask by dropping trailing zeros.  Does
not check whether `s` is a valid swizzle mask.
"""
function strip_swizzle_mask(s::AbstractVector{Int})
    f = findlast(!isequal(0))
    return s[1:(f isa Nothing ? 0 : 1)]
end
strip_swizzle_mask(s::Tuple{}) = s
strip_swizzle_mask(s::Tuple{Int}) = s[1] == 0 ? () : s
strip_swizzle_mask(s::Tuple) = last(s) == 0 ? strip_swizzle_mask(front(s)) : s
