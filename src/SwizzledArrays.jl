# The Swizzled!

struct Swizzled{N,s,M,si,A,F}
    data::A
    op::F

    function Swizzled{N,s,M,si,A,F}(data::A, op::F) where {N,s,M,si,A,F}
        (isa(s, NTuple{N,Int})) || throw(ArgumentError("swizzle mask $s must be NTuple{$N,Int}"))
        (isa(si, NTuple{M,Int})) || throw(ArgumentError("inverse swizzle mask $si must be NTuple{$M,Int}"))
        is_swizzle_mask(s) || throw(ArgumentError("$s is not a valid swizzle mask"))
        is_swizzle_mask(si) || throw(ArgumentError("$si is not a valid swizzle mask"))
        [swizzle_elems(s, 0, si)...] == [d in s ? d : 0 for d = 1:M] || throw(ArgumentError("$s and $si must be inverses"))
        new(data)
    end
end

unspecified_op(a, b) = throw(ArgumentError("Unspecified reduction operator"))

"""
    Swizzled(A, s, op=unspecified_op) -> B
Given a broadcastable `A` and a swizzle mask `s`, create a view `B` such that
the dimensions appear to be swizzled. A swizzle is a mapping from the
dimensions of `B` to the dimensions of `A`, where the value `0` means that a
dimension of size 1 is inserted. Not all dimensions of `A` need to be appear in
`B`. If a dimension of `A` is not included in `s`, the corresponding axis will
be reduced using `op`. If no `op` is specified, then the dimension is assumed
to have size `1`. Similar to `swizzle`, except that no copying occurs (`B`
shares storage with `A`).  See also: [`swizzle`](@ref), [`swizzle`](@ref) and
[`PermutedDimsArray`](@ref).
# Examples
```jldoctest
julia> A = rand(3,1,4);
julia> B = Swizzled(A, (0,3,0,1));
julia> size(B)
(1, 4, 1, 3)
julia> B[1,3,1,2] == A[2,1,3]
true
```
"""
Swizzled

Swizzled(data, s::AbstractVector{Int}, op) = Swizzled(data, (s...,))
function Swizzled(data, s::NTuple{N, Int}, op) where {N}
    si = inverse_swizzle_mask(s)
    Swizzled{N,(s...,),length(si),(si...,),typeof(data), typeof(op)}(data, op)
end

Base.data(sz::Swizzled) = sz.data
Base.size(sz::Swizzled) = map(size, broadcast_axes(sz))
Base.axes(sz::Swizzled{N,s}) where {N,s} = swizzle_elems(broadcast_axes(sz.data), Base.OneTo(1), s)

@inline function Base.getindex(sz::Swizzled{N,s,M,si}, inds::Vararg{Int,N}) where {T,N,s,M,si}
    data_axes = broadcast_axes(sz.data)
    data_inds = swizzle_elems(inds, 1, si)
    @boundscheck begin
      for 
      checkindex(Bool, ind, 
    end
    @boundscheck checkbounds_indices(Bool, data_axes, swizzle_elems(inds, 1, si)) || throw_boundserror(sz, I)
    iter = eachindex(bc′)
    for 
    ElType = combine_eltypes(bc.f, bc.args)
    if Base.isconcretetype(ElType)
        # We can trust it and defer to the simpler `copyto!`
        return copyto!(similar(bc, ElType), bc)
    end
    # When ElType is not concrete, use narrowing. Use the first output
    # value to determine the starting output eltype; copyto_nonleaf!
    # will widen `dest` as needed to accommodate later values.
    bc′ = preprocess(nothing, bc)
    y = iterate(iter)
    if y === nothing
        # if empty, take the ElType at face value
        return similar(bc′, ElType)
    end
    # Initialize using the first value
    I, state = y
    @inbounds val = bc′[I]
    dest = similar(bc′, typeof(val))
    @inbounds dest[I] = val
    # Now handle the remaining values
return copyto_nonleaf!(dest, bc′, iter, state, 1)
    @inbounds val = getindex(A.data, swizzle_elems(I, Colon(), si)...)
    val
end

@inline function Base.setindex!(A::Swizzled{T,N,s,si,<:AbstractArray}, val, I::Vararg{Int,N}) where {T,N,s,si}
    @boundscheck checkbounds_indices(Bool, sz.data, swizzle_elems(I, 1, si)...) || throw_boundserror(sz, I)
    @boundscheck checkbounds_indices(Bool, A, I...) || throw_boundserror(A, I)
    @inbounds setindex!(A.data, val, swizzle_elems(I, 1, si)...)
    val
end

"""
    swizzle(A::AbstractArray, s)
Given an AbstractArray `A` and a swizzle mask `s`, create copy `B` of `A` where
the dimensions have been swizzled. If B has at least as many dimensions as `A`,
a swizzle is a mapping from the dimensions of B to the dimensions of `A`, where
the value `0` means that a dimension of size 1 is inserted. Not all dimensions
of `A` need to be appear in `B`. If a dimension of `A` is not included in `s`,
the corresponding axis must be `1:1`.
See also: [`Swizzled`](@ref) and [`permutedims`](@ref).
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
julia> swizzle(A, [3, 2, 1])
2×2×2 Array{Int64,3}:
[:, :, 1] =
 1  3
 5  7
[:, :, 2] =
 2  4
 6  8
```
"""
function swizzle(A::AbstractArray, s)
    dest = similar(A, swizzle_elems(size(A), 1, s))
    swizzle!(dest, A, s)
end

"""
    beam(A, dims...) -> B
Given a broadcastable `A`, create a view `B` such that the dimension `i`
appears as dimension `dims[i]` in B, filling in dimensions of size `1` as
necessary. A zero value may be used to indicate that a dimension of `A` is
of size `1` and will not be used in the view.
See also: [`Swizzled`](@ref).
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
  Swizzled(data, inverse_swizzle_mask(dims))
end

"""
    swizzle!(dest, src, s)
Swizzle the dimensions of array `src` according to `s` and store the result in
the array `dest`.  `s` is a mapping from the dimensions of `dest` to the
dimensions of `src`, where the value `0` means that a dimension of size 1 is
inserted. Not all dimensions of `src` need to be appear in `dest`. If a
dimension of `src` is not included in `s`, the corresponding axis must be
`1:1`.  The preallocated array `dest` should have `size(dest) ==
swizzle_elems(size(src), 1:1, s)` `dest` is completely overwritten. No in-place
swizzle is supported and unexpected results will happen if `src` and `dest`
have overlapping memory regions.  See also [`swizzle_elems`](@ref),
[`swizzle`](@ref) and [`permutedims!`](@ref).
"""
function swizzle!(dest::AbstractArray{T, N}, src::AbstractArray{S, M}, s) where {T, N, S, M}
    (isa(s, NTuple{N,Int})) || throw(ArgumentError("swizzle mask $s must be NTuple{$N,Int}"))
    maximum(s) <= M || throw(ArgumentError("swizzle mask $s is not a valid swizzle mask for dimensions 1:$M"))
    si = inverse_swizzle_mask(s)
    for d in findall(isequal(0), si)
        axes(src, d) == 1:1 || throw(ArgumentError("cannot drop dim $d with axis $(axes(data, d))"))
    end
    is_swizzle_mask(s) || throw(ArgumentError("$s is not a valid swizzle of dimensions 1:$M"))
    perm = sortperm(filter(!isequal(0), [s...]))
    permutedims!(view(dest, [d == 0 ? 1 : Colon() for d in s]...), view(src, [d == 0 ? 1 : Colon() for d in si]...), perm)
end

function Base.copyto!(dest::Swizzled{T,N,s,si}, src::AbstractArray{S,N}) where {T,S,N,s,si}
    axes(dest) == axes(src)
    perm = sortperm(filter(!isequal(0), [si...]))
    permutedims!(view(dest.data, [d == 0 ? 1 : Colon() for d in si]...), view(src, [d == 0 ? 1 : Colon() for d in s]...), perm)
end

function Base.showarg(io::IO, A::Swizzled{T,N,swizzle}, toplevel) where {T,N,swizzle}
    print(io, "Swizzled(")
    Base.showarg(io, data(A), false)
    print(io, ", ", swizzle, ')')
    toplevel && print(io, " with eltype ", eltype(A))
end
