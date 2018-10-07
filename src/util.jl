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
    is_smallest_swizzle_mask(s) -> Bool
Return `true` if `s` is a valid swizzle mask, and the smallest of its kind.
# Examples
```jldoctest
julia> is_smallest_swizzle_mask([1; 3])
true
julia> is_swizzle_mask([1; 1])
false
julia> is_swizzle_mask([1; 0; 0; 2])
true
julia> is_swizzle_mask([1; 0; 0; 1])
false
```
"""
is_smallest_swizzle_mask(s::Union{AbstractVector, Tuple}) =
  length(s) == 0 || (last(s) != 0 && is_swizzle_mask(s))

"""
    swizzle_elems(v, w, w0, s)
Swizzle `w` into `v` according to the mask `s` and return the resulting
collection `r`.  `r` should be similar to `v` and satisfy `r[i] = b[s[i]]` when
`s[i] != 0` and `r[i] = a[i]` when `s[i] == 0` or when `i > length(s)`. `b`
must be at least as long as `maximum(s)`. No checking is done to determine
whether `s` is a valid mask.
# Examples
```jldoctest
julia> s = [2; 4; 0; 3; 1];
julia> a = [5; 5; 5; 5];
julia> b = [-1; -2; -3; -4];
julia> swizzle_elems(a, b, s)
4-element Array{Int64,1}:
 -2
 -4
 5
 -3
julia> a = [5; 5; 5; 5; 5; 5];
julia> swizzle_elems(a, b, s)
6-element Array{Int64,1}:
 -2
 -4
 5
 -3
 -1
 5
```
"""
swizzle_elems

function swizzle_elems(a, b::AbstractVector, s)
    return swizzle_elems!(similar(b, Union{eltype(a), eltype(b)}), a, b, s)
end

@inline function swizzle_elems(a, b::NTuple{N}, s) where {N}
    a isa AbstractVector && @assert !Base.has_offset_axes(a)
    ntuple(i -> (i > length(s) || s[i] == 0) ? b[i] : a[s[i]], Val(N))
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
