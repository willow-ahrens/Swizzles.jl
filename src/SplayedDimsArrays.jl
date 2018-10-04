module SplayedDimsArrays

export splaydims, invsplaydims
export invsplay
export SplayedDimsArray
export cast

# Basic functions for working with splays

_filter_zeros(v) = filter(a -> a != 0, v)
_filter_zeros(v::Tuple) = (_filter_zeros([v...])...,)

"""
    issplay(v) -> Bool
Return `true` if `v` is a valid splay.
# Examples
```jldoctest
julia> issplay([1; 2])
true
julia> issplay([1; 3])
false
julia> issplay([1; 0; 2])
true
julia> issplay([1; 0; 3])
false
```
"""
function issplay(v)
    return isperm(_filter_zeros(v))
end

issplay(v::Tuple{}) = true
issplay(v::Tuple{Int}) = v[1] >= 0
issplay(v::Tuple{Int,Int}) = ((v[1] == 1) & (v[2] == 2)) | ((v[1] == 2) & (v[2] == 1)) | ((v[1] == 1) & (v[2] == 0)) | ((v[1] == 0) & (v[2] == 1)) | ((v[1] == 0) & (v[2] == 0))

"""
    invsplay(v)
Return the inverse splay of `v`.
A splay is a permutation with some "filler" positions represented by zeros
If the largest integer in `v` is `n` , then `v[invsplay(v)] == 1:n`.
# Examples
```jldoctest
julia> v = [2; 4; 0; 3; 1];
julia> invsplay(v)
4-element Array{Int64,1}:
 5
 1
 4
 2
julia> v[invsplay(v)]
4-element Array{Int64,1}:
 1
 2
 3
 4
```
"""
function invsplay(a::AbstractVector)
    @assert !has_offset_axes(a)
    b = zero(a) # similar vector of zeros
    n = length(a)
    @inbounds for (i, j) in enumerate(a)
        (j == 0 || ((1 <= j <= n) && b[j] == 0)) ||
            throw(ArgumentError("argument is not a splay"))
        if j != 0
            b[j] = i
        end
    end
    b
end

invsplay(p::Tuple{}) = p
function invsplay(p::Tuple{Int})
    if p[1] == 0
        return ()
    elseif p[1] == 1
        return ntuple(d -> d == n, p)
    else
        throw(ArgumentError("argument is not a splay"))
    end
end
function invsplay(p::Tuple{Int, Int})
    if p[1] == 0 && p[2] == 0
        return ()
    elseif (p[1] == 1 && p[2] == 0) || (p[1] == 0 && p[2] == 1)
        return (1,)
    elseif (p[1] == 1 && p[2] == 2) || (p[1] == 2 && p[2] == 1)
        return p
    else
        throw(ArgumentError("argument is not a splay"))
    end
end

invsplay(a::Tuple) = (invsplay([a...])...,)

"""
    invinvsplay(v)
Return the smallest splay which, when inverted, produces `v`.
A splay is a permutation with some "filler" positions represented by zeros
If `v` is of length `n` , then `invinvsplay(v)[v] == 1:n`.
# Examples
```jldoctest
julia> v = [5; 1; 4; 2];
julia> invinvsplay(v)
5-element Array{Int64,1}:
 2
 4
 0
 3
 1
julia> invinvsplay(v)[v]
4-element Array{Int64,1}:
 1
 2
 3
 4
```
"""
function invinvsplay(a::AbstractVector)
    @assert !has_offset_axes(a)
    b = zero(a) # similar vector of zeros
    n = maximum(a)
    @inbounds for (i, j) in enumerate(a)
        (j != 0 && ((1 <= j) && b[j] == 0)) ||
            throw(ArgumentError("argument is not an inverse splay"))
        b[j] = i
    end
    b
end
invinvsplay(p::Tuple{}) = p
function invinvsplay(p::Tuple{Int})
    if p[1] == 0
        return ()
    elseif p[1] >= 1
        return ntuple(i -> i == p[1], p[1])
    else
        throw(ArgumentError("argument is not a splay"))
    end
end

invinvsplay(a::Tuple) = (invinvsplay([a...])...,)

# The SplayedDimsArray!

# Some day we will want storage-order-aware iteration, so put splay in the parameters
struct SplayedDimsArray{T,N,splay,isplay,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::AA

    function SplayedDimsArray{T,N,splay,isplay,AA}(data::AA) where {T,N,M,splay,isplay,AA<:AbstractArray{T, M}}
        (isa(splay, NTuple{N,Int})) || error("splay must be NTuple{$N,Int}")
        (isa(isplay, NTuple{M,Int})) || error("isplay must be NTuple{$M,Int} since parent is AbstractArray{$T, $M}")
        count(d -> d != 0, splay) == M || throw(ArgumentError(string(splay, " is not a valid splay of dimensions 1:", M)))
        issplay(splay) || throw(ArgumentError(string(splay, " is not a valid splay of dimensions 1:", M)))
        isperm((filter(d -> d != 0, splay))) || throw(ArgumentError(string(splay, " is not a valid splay of dimensions 0:", N)))
        all(map(d->isplay[splay[d]]==d, 1:M)) || throw(ArgumentError(string(splay, " and ", isplay, " must be inverses")))
        new(data)
    end
end

"""
    SplayedDimsArray(A, splay) -> B
Given an AbstractArray `A`, create a view `B` such that the dimensions appear
to be splayed. If B has at least as many dimensions as A, a splay is a mapping
from the dimensions of B to the dimensions of A, where the value `0` means that
a dimension of size 1 is inserted. B should have at least as many dimensions as
A. Similar to `splaydims`, except that no copying occurs (`B` shares storage
with `A`).  See also: [`splaydims`](@ref) and [`PermutedDimsArray`](@ref).
# Examples
```jldoctest
julia> A = rand(3,5,4);
julia> B = SplayedDimsArray(A, (3,0,1,2,0,0));
julia> size(B)
(4, 1, 3, 5, 1, 1)
julia> B[3,1,4,2,1,1] == A[4,1,2,3,1,1]
true
```
"""
function SplayedDimsArray(data::AbstractArray{T,M}, splay::NTuple{Int, N}) where {T,N,M}
    count(d -> d != 0, splay) == M || throw(ArgumentError(string(splay, " is not a valid splay of dimensions 1:", M)))
    isplay = invsplay(splay)
    SplayedDimsArray{T,N,M,(splay...,),(isplay...,),typeof(data)}(data)
end

"""
    cast(A, dims...) -> B
Given an AbstractArray `A`, create a view `B` such that the dimension `i`
appears as dimension `dims[i]` in B, filling in dimensions of size `1` as
necessary.
See also: [`SplayedDimsArray`](@ref).
# Examples
```jldoctest
julia> A = rand(3,5,4);
julia> B = cast(A, 1, 4, 3)
julia> size(B)
(3, 1, 4, 5)
julia> B[2,1,4,3] == A[2,3,4]
true
```
"""
function cast(data::AbstractArray{T, N}, dims::Vararg{Int, N})
  all(map(d -> d >= 1, dims)) || throw(ArgumentError("dims must each be at least 1"))
  allunique(dims) || throw(ArgumentError("dims must be unique"))
  SplayedDimsArray(data, invinvsplay(dims))
end

Base.parent(A::SplayedDimsArray) = A.parent
Base.size(A::SplayedDimsArray{T,N,splay}) where {T,N,splay} = gensplay(size(parent(A)), 1, splay)
Base.axes(A::SplayedDimsArray{T,N,splay}) where {T,N,splay} = gensplay(axes(parent(A)), 1:1, splay)

Base.unsafe_convert(::Type{Ptr{T}}, A::SplayedDimsArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(A))

# It's OK to return a pointer to the first element, and indeed quite
# useful for wrapping C routines that require a different storage
# order than used by Julia. But for an array with unconventional
# storage order, a linear offset is ambiguous---is it a memory offset
# or a linear index?
Base.pointer(A::SplayedDimsArray, i::Integer) = throw(ArgumentError("pointer(A, i) is deliberately unsupported for SplayedDimsArray"))

function Base.strides(A::SplayedDimsArray{T,N,splay}) where {T,N,splay}
    s = strides(parent(A))
    return gensplay(s, 1, splay)
end

@inline function Base.getindex(A::SplayedDimsArray{T,N,splay,isplay}, I::Vararg{Int,N}) where {T,N,splay,isplay}
    @boundscheck checkbounds(A, I...)
    @inbounds val = getindex(A.parent, gensplay(I, 0, isplay)...)
    val
end
@inline function Base.setindex!(A::SplayedDimsArray{T,N,splay,isplay}, val, I::Vararg{Int,N}) where {T,N,splay,isplay}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.parent, val, gensplay(I, 0, isplay)...)
    val
end

@inline gensplay(I::NTuple{N,Any}, I0, splay::Dims{M}) where {N,M} = ntuple(d -> splay[d] == 0 ? I0 : I[splay[d]], Val(M))
@inline gensplay(I, I0, splay::AbstractVector{Int}) = gensplay(I, (splay...,))

"""
    splaydims(A::AbstractArray, splay)
Return a copy of array `A` where the dimensions have been rearranged according
to `splay`, where `splay` is a vector containing a permutation of the
dimensions of `A` with optional `0` elements that represent dimensions of size
`1`.
See also: [`SplayedDimsArray`](@ref) and [`permutedims`](@ref).
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
julia> splaydims(A, [3, 2, 1])
2×2×2 Array{Int64,3}:
[:, :, 1] =
 1  3
 5  7
[:, :, 2] =
 2  4
 6  8
```
"""
function splaydims(A::AbstractArray, splay)
    dest = similar(A, gensplay(axes(A), 1:1, splay))
    splaydims!(dest, A, splay)
end

"""
    splaydims!(dest, src, splay)
Splay the dimensions of array `src` and store the result in the array `dest`.
`splay` is a vector specifying a permutation of length `ndims(src)` with
optional zeros filled in to represent dimensions that must be one. The
preallocated array `dest` should have `size(dest)[i] == size(src)[splay[i]]`
whenever `splay[i] != 0` and `size(dest)[i] == 0` whenever `splay[i] == 0`.
`dest` is completely overwritten. No in-place splay is supported and unexpected
results will happen if `src` and `dest` have overlapping memory regions.  See
also [`splaydims`](@ref) and [`permutedims!`](@ref).
"""
function splaydims!(dest, src::AbstractArray, splay)
    permutedims!(view(dest, gensplay(axes(A), 1, splay)...), src, filter(d -> d != 0, splay))
end

function Base.copyto!(dest::SplayedDimsArray{T,N,splay}, src::AbstractArray{T,N}) where {T,N}
    permutedims!(dest.parent, src, invperm(filter(d -> d != 0, splay)))
end
Base.copyto!(dest::SplayedDimsArray, src::AbstractArray) = _copy!(dest, src)

function Base.showarg(io::IO, A::SplayedDimsArray{T,N,splay}, toplevel) where {T,N,splay}
    print(io, "SplayedDimsArray(")
    Base.showarg(io, parent(A), false)
    print(io, ", ", splay, ')')
    toplevel && print(io, " with eltype ", eltype(A))
end

end # module
