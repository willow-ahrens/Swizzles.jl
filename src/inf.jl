struct Sequence{T, C, R}
  cache::C
  rest::R
end

function getindex(a::Sequence, i::Int)
    @boundscheck i >= 0 || throw
    if i <= length(a.cache)
        return a.cache[i]
    else
        return a.rest[i - length(a.cache)]
    end
end

getindex(a, i::Sequence{Int}) = Sequence(a[i.cache], a[i.rest])



struct Skip end

const skip = Skip()

getindexinto(a, b, ::Skip) = a

setindexinto(a, b, ::Skip) = a

#A = B[I]

#getindexinto stops caching when all of the remaining iterator comes from either a's rest or b's rest

    a_unrest = findlast(j -> !isa(i[j], Skip), 1:length(a.cache))
    a_unrest = a_unrest == nothing ? 0 : a_unrest

getindexinto_receiver_cache(a::Sequence{<:T}, i::Sequence{<:Union{Int, Skip}}
getindexinto_sender_cache(
getindexinto_receiver_length(

function getindexinto(a::Sequence{<:T},
                      b::Sequence{<:T},
                      i::Sequence{<:Union{Int, Skip}, <:AbstractVector}) where {T}
    c_cache = similar(i.cache, T, 0)
    for i in 
    


    b_unrest = findlast(j -> !isa(i[j], Skip), 1:length(b.cache))
    a_unrest = a_unrest == nothing ? 0 : a_unrest

    b_unrest = findlast(!isequal(a.rest[0]), i.cache)
    a_unrest = a_unrest == nothing ? 0 : a_unrest
    b_unrest = 
    c_cache = similar(b.cache, T, findfirst(j -> i[j] isa Skip ? i[j] < length(b.cache) : j < length(a.cache), 1:))
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

#skip should return a and res be similar to b
axes(sz) = getindexinto(Sequence(repeated(1:1)), Sequence(axes(sz.data), repeated(1:1)), sz.dims)

#skip should return a and res be similar to a but if it is similar to b it should be similar to a so shrug
inds(sz.data) = getindexinto(Sequence((inds), repeated(don't call)), axes(sz.data), inds, sz.idims)

#A[I] = B

function setindexinto(a::Sequence{<:T}, b::Sequence{<:T}, i::Sequence{<:Union{Int, Skip}}) 
  


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
