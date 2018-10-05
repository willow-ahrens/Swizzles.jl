module SwizzledDimsArrays

export swizzledims, invswizzledims
export invswizzlemask
export SwizzledDimsArray
export beam

# Functions that should be in Base that aren't
import Base.filter

filter(f, ::Tuple{}) = ()
filter(f, args::Tuple{Any}) = f(args[1]) ? args : ()
function filter(f, args::Tuple)
    tail = filter(f, Base.tail(args))
    f(args[1]) ? (args[1], tail...) : tail
end

# Basic Functions for working with swizzles

"""
    isswizzle(s) -> Bool
Return `true` if `s` is a valid swizzle mask.
# Examples
```jldoctest
julia> isswizzle([1; 2])
true
julia> isswizzle([1; 3])
false
julia> isswizzle([1; 0; 2])
true
julia> isswizzle([1; 0; 3])
false
```
"""
function isswizzle(s)
    return allunique(filter(!isequal(0), s)) && all(s .>= 0)
end

isswizzle(s::Tuple{}) = true
isswizzle(s::Tuple{Int}) = s[1] >= 0
isswizzle(s::Tuple{Int,Int}) = s[1] >= 0 && s[2] >= 0 && s[1] != s[2]

"""
    swizzle(v, v0, s)
Swizzle `v` according to the mask `s` and return the resulting collection `r`.
`r` should be similar to `s` and satisfy `r[i] = v[s[i]]` when `s[i] != 0` and
`r[i] = v0` when `s[i] == 0`. `v` must be at least as long as `maximum(s)`. No
checking is done to determine whether `s` is a valid mask.
# Examples
```jldoctest
julia> s = [2; 4; 0; 3; 1];
julia> v = [3; 3; 3; 3];
julia> swizzle(v, 4, s)
5-element Array{Int64,1}:
 3
 3
 4
 3
 3
```
"""
function swizzle(v::Union{AbstractVector{T}, NTuple{M, T}}, v0::T0, s::AbstractVector{Int}) where {M, T, T0}
    return swizzle!(similar(s, Union{T, T0}), v, v0, s)
end

@inline function swizzle(v, v0, s::NTuple{N, Int}) where {N}
    v isa AbstractVector && @assert !Base.has_offset_axes(v)
    @boundscheck @assert maximum(s) <= length(v)
    ntuple(i -> s[i] == 0 ? v0 : v[s[i]], Val(N))
end

"""
    swizzle!(r, v, v0, s)
Swizzle `v` into mutable `r` according to the mask `s` and return `r`.  The
final state of `r` should satisfy `r[i] = v[s[i]]` when `s[i] != 0` and `r[i] = v0`
when `s[i] == 0`. `r` and `s` should have the same length. No checking is done
to determine whether `s` is a valid mask. `v` must be at least as long as
`maximum(s)`.
# Examples
```jldoctest
julia> s = [2; 4; 0; 3; 1];
julia> v = [3; 3; 3; 3];
julia> r = [1; 1; 1; 1; 1];
julia> swizzle!(r, v, 4, s)
5-element Array{Int64,1}:
 3
 3
 4
 3
 3
```
"""
function swizzle!(r::AbstractVector, v, v0, s::AbstractVector{Int}) where {M, T, T0, S <: Union{T, T0}}
    @assert !Base.has_offset_axes(v)
    @boundscheck @assert 0 <= maximum(s) <= length(v)
    @boundscheck @assert length(s) == length(r)
    @inbounds for (si, i) in enumerate(s)
        r[i] = si == 0 ? v0 : v[si]
    end
    r
end

"""
    invswizzlemask(s, m = maximum(s))
Return the mask `t` of size `m >= maximum(s)` such that
```
  swizzle(s, 0, t) = [i in s ? i : 0 for i = 1:m]
```

It is helpful to think of this operation as producing a pseudoinverse of the
swizzle mask.  For example, if we have some collection `v` and compute `w =
swizzle(v, nothing, s)`, then `swizzle(w, nothing, t)` will reconstruct as much
of the original v as possible.
# Examples
```jldoctest
julia> s = [0; 0; 3; 1];
julia> t = invswizzlemask(s, 4)
3-element Array{Int64,1}:
 4
 0
 3
 0
julia> swizzle(s, 0, t)
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
julia> w = swizzle(v, nothing, s)
4-element Array{Union{Int64, Nothing},1}:
 nothing
 nothing
 3.0
 1.0
julia> swizzle(w, nothing, t)
4-element Array{Union{Int64, Nothing},1}:
 1.0
 nothing
 3.0
 nothing
```
"""
function invswizzlemask(s::AbstractVector{Int}, m=maximum(s))
    @assert !Base.has_offset_axes(s)
    0 <= maximum(s) || throw(ArgumentError("argument is not a swizzle"))
    maximum(s) <= m || throw(ArgumentError("swizzle mask size is too small"))
    t = fill!(similar(s, m), 0) # similar vector of zeros
    @inbounds for (i, si) in enumerate(s)
        (si == 0 || ((si >= 1) && t[si] == 0)) ||
            throw(ArgumentError("argument is not a swizzle"))
        t[si] = i
    end
    t
end
function invswizzlemask(s::Tuple{}, m=maximum(s))
    maximum(s) <= m || throw(ArgumentError("swizzle mask size is too small"))
    ntuple(0, m)
end
function invswizzlemask(s::Tuple{Int}, m=maximum(s))
    maximum(s) <= m || throw(ArgumentError("swizzle mask size is too small"))
    if s[1] == 0
        return ntuple(0, m)
    elseif s[1] >= 1
        return ntuple(i -> i == s[1] ? 1 : 0, m)
    else
        throw(ArgumentError("argument is not a swizzle"))
    end
end
function invswizzlemask(s::Tuple{Int, Int}, m=maximum(s))
    maximum(s) <= m || throw(ArgumentError("swizzle mask size is too small"))
    if s[1] == 0 && s[2] == 0
        return ntuple(0, m)
    elseif (s[1] >= 1 && s[2] == 0)
        return ntuple(i -> i == s[1] ? 1 : 0, m)
    elseif (s[2] >= 1 && s[1] == 0)
        return ntuple(i -> i == s[2] ? 2 : 0, m)
    elseif (s[1] >= 1 && s[2] >= 1 && s[1] != s[2])
        return ntuple(i -> i == s[1] ? 1 : (i == s[2] ? 2 : 0), m)
    else
        throw(ArgumentError("argument is not a swizzle"))
    end
end

invswizzlemask(s::Tuple, m=maximum(s)) = (invswizzlemask([s...], m)...,)

# The SwizzledDimsArray!

struct SwizzledDimsArray{T,N,smask,ismask,A<:AbstractArray} <: AbstractArray{T,N}
    parent::A

    function SwizzledDimsArray{T,N,smask,ismask,A}(data::A) where {T,N,M,smask,ismask,A<:AbstractArray{T, M}}
        (isa(smask, NTuple{N,Int})) || throw(ArgumentError("swizzle mask $smask must be NTuple{$N,Int}"))
        (isa(ismask, NTuple{M,Int})) || throw(ArgumentError("inverse swizzle mask $ismask must be NTuple{$M,Int} since parent is AbstractArray{$T, $M}"))
        maximum(smask) <= M || throw(ArgumentError("swizzle mask $smask is not a valid swizzle mask for dimensions 1:$M"))
        for d in findall(isequal(0), ismask)
            axes(data, d) == 1:1 || throw(ArgumentError("cannot drop dim $d with axis $(axes(data, d))"))
        end
        isswizzle(smask) || throw(ArgumentError("$smask is not a valid swizzle of dimensions 1:$M"))
        [swizzle(smask, 0, ismask)...] == [d in smask ? d : 0 for d = 1:M] || throw(ArgumentError("$smask and $ismask must be inverses"))
        new(data)
    end
end

"""
    SwizzledDimsArray(A, s) -> B
Given an AbstractArray `A` and a swizzle mask `s`, create a view `B` such that
the dimensions appear to be swizzled. If B has at least as many dimensions as
`A`, a swizzle is a mapping from the dimensions of B to the dimensions of `A`,
where the value `0` means that a dimension of size 1 is inserted. Not all
dimensions of `A` need to be appear in `B`. If a dimension of `A` is not
included in `s`, the corresponding axis must be `1:1`. Similar to
`swizzledims`, except that no copying occurs (`B` shares storage with `A`).
See also: [`swizzle`](@ref), [`swizzledims`](@ref) and
[`PermutedDimsArray`](@ref).
# Examples
```jldoctest
julia> A = rand(3,1,4);
julia> B = SwizzledDimsArray(A, (0,3,0,1));
julia> size(B)
(1, 4, 1, 3)
julia> B[1,3,1,2] == A[2,1,3]
true
```
"""
function SwizzledDimsArray(data::AbstractArray{T,M}, smask::NTuple{N, Int}) where {T,N,M}
    maximum(smask) <= M || throw(ArgumentError("swizzle mask $smask is not a valid swizzle mask for dimensions 1:$M"))
    ismask = invswizzlemask(smask, M)
    SwizzledDimsArray{T,N,(smask...,),(ismask...,),typeof(data)}(data)
end

"""
    beam(A, dims...) -> B
Given an AbstractArray `A`, create a view `B` such that the dimension `i`
appears as dimension `dims[i]` in B, filling in dimensions of size `1` as
necessary. A zero value may be used to indicate that a dimension of `A` is
of size `1` and will not be used in the view.
See also: [`SwizzledDimsArray`](@ref).
# Examples
```jldoctest
julia> A = rand(3,1,4);
julia> B = beam(A, (4, 0, 3))
julia> size(B)
(1, 1, 4, 3)
julia> B[1,1,2,3] == A[3,1,2]
true
```
"""
function beam(data::AbstractArray{T, M}, dims::Vararg{Int, M}) where {T, M}
  all(map(d -> d >= 0, dims)) || throw(ArgumentError("dims must each be at least 0"))
  allunique(filter(!isequal(0), dims)) || throw(ArgumentError("nonzero dims may not be repeated"))
  SwizzledDimsArray(data, invswizzlemask(dims))
end

Base.parent(A::SwizzledDimsArray) = A.parent
Base.size(A::SwizzledDimsArray{T,N,smask}) where {T,N,smask} = println(swizzle(size(parent(A)), 1, smask))
Base.axes(A::SwizzledDimsArray{T,N,smask}) where {T,N,smask} = swizzle(axes(parent(A)), Base.OneTo(1), smask)

Base.unsafe_convert(::Type{Ptr{T}}, A::SwizzledDimsArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(A))

# It's OK to return a pointer to the first element, and indeed quite
# useful for wrapping C routines that require a different storage
# order than used by Julia. But for an array with unconventional
# storage order, a linear offset is ambiguous---is it a memory offset
# or a linear index?
Base.pointer(A::SwizzledDimsArray, i::Integer) = throw(ArgumentError("pointer(A, i) is deliberately unsupported for SwizzledDimsArray"))

function Base.strides(A::SwizzledDimsArray{T,N,smask}) where {T,N,smask}
    s = strides(parent(A))
    return swizzle(s, 1, smask) #FIXME
end

@inline function Base.getindex(A::SwizzledDimsArray{T,N,smask,ismask}, I::Vararg{Int,N}) where {T,N,smask,ismask}
    @boundscheck checkbounds(A, I...)
    @inbounds val = getindex(A.parent, swizzle(I, 1, ismask)...)
    val
end
@inline function Base.setindex!(A::SwizzledDimsArray{T,N,smask,ismask}, val, I::Vararg{Int,N}) where {T,N,smask,ismask}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.parent, val, swizzle(I, 1, ismask)...)
    val
end

"""
    swizzledims(A::AbstractArray, s)
Given an AbstractArray `A` and a swizzle mask `s`, create copy `B` of `A` where
the dimensions have been swizzled. If B has at least as many dimensions as `A`,
a swizzle is a mapping from the dimensions of B to the dimensions of `A`, where
the value `0` means that a dimension of size 1 is inserted. Not all dimensions
of `A` need to be appear in `B`. If a dimension of `A` is not included in `s`,
the corresponding axis must be `1:1`.
See also: [`SwizzledDimsArray`](@ref) and [`permutedims`](@ref).
# Examples
```jldoctest
julia> A = reshape(Vector(1:8), (2,2,2))
2×2×2 Array{Int64,3}:
[:, :, 1] =
 1  3
 2  4
[:, :, 2] =
 5  7
 6  8
julia> swizzledims(A, [3, 2, 1])
2×2×2 Array{Int64,3}:
[:, :, 1] =
 1  3
 5  7
[:, :, 2] =
 2  4
 6  8
```
"""
function swizzledims(A::AbstractArray, smask)
    dest = similar(A, swizzle(size(A), 1, smask))
    swizzledims!(dest, A, smask)
end

"""
    swizzledims!(dest, src, s)
Swizzle the dimensions of array `src` according to `s` and store the result in
the array `dest`.  `s` is a mapping from the dimensions of `dest` to the
dimensions of `src`, where the value `0` means that a dimension of size 1 is
inserted. Not all dimensions of `src` need to be appear in `dest`. If a
dimension of `src` is not included in `s`, the corresponding axis must be
`1:1`.  The preallocated array `dest` should have `size(dest) ==
swizzle(size(src), 1:1, s)` `dest` is completely overwritten. No in-place
swizzle is supported and unexpected results will happen if `src` and `dest`
have overlapping memory regions.  See also [`swizzle`](@ref),
[`swizzledims`](@ref) and [`permutedims!`](@ref).
"""
function swizzledims!(dest::AbstractArray{T, N}, src::AbstractArray{S, M}, smask) where {T, N, S, M}
    (isa(smask, NTuple{N,Int})) || throw(ArgumentError("swizzle mask $smask must be NTuple{$N,Int}"))
    maximum(smask) <= M || throw(ArgumentError("swizzle mask $smask is not a valid swizzle mask for dimensions 1:$M"))
    ismask = invswizzlemask(smask)
    for d in findall(isequal(0), ismask)
        axes(src, d) == 1:1 || throw(ArgumentError("cannot drop dim $d with axis $(axes(data, d))"))
    end
    isswizzle(smask) || throw(ArgumentError("$smask is not a valid swizzle of dimensions 1:$M"))
    perm = sortperm(filter(!isequal(0), [smask...]))
    permutedims!(view(dest, [d == 0 ? 1 : Colon() for d in smask]...), view(src, [d == 0 ? 1 : Colon() for d in ismask]...), perm)
end

function Base.copyto!(dest::SwizzledDimsArray{T,N,smask,ismask}, src::AbstractArray{S,N}) where {T,S,N,smask,ismask}
    axes(dest) == axes(src)
    perm = sortperm(filter(!isequal(0), [ismask...]))
    permutedims!(view(dest.parent, [d == 0 ? 1 : Colon() for d in ismask]...), view(src, [d == 0 ? 1 : Colon() for d in smask]...), perm)
end

function Base.showarg(io::IO, A::SwizzledDimsArray{T,N,swizzle}, toplevel) where {T,N,swizzle}
    print(io, "SwizzledDimsArray(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", swizzle, ')')
    toplevel && print(io, " with eltype ", eltype(A))
end

end # module
