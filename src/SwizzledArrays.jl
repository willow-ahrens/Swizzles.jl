export swizzle, swizzle!
export Swizzler, Reduce, Sum, Max, Min, Beam
export SwizzleTo, ReduceTo, SumTo, MaxTo, MinTo, BeamTo

### An operator which does not expect to be called.

"""
    `nooperator(a, b)`

An operator which does not expect to be called. It startles easily.
"""
nooperator(a, b) = throw(ArgumentError("unspecified operator"))

struct SwizzledArray{T, N, Arg, mask, imask, Op} <: AbstractArray{T, N}
    arg::Arg
    op::Op
    function SwizzledArray{T, N, Arg, mask, imask, Op}(arg::Arg, op::Op) where {T, N, Arg, mask, imask, Op}
        #FIXME check swizzles. also check noop axes!
        new(arg, op)
    end
end



@inline function SwizzledArray(sz::SwizzledArray)
    if eltype(mask(sz)) isa Int
        return SwizzledArray{eltype(sz.arg)}(sz)
    end
    T = eltype(sz.arg)
    T! = Union{T, Base._return_type(sz.op, (T, T))}
    if T! isa T
        return T!
    end
    T = T!
    T! = Union{T, Base._return_type(sz.op, (T, T))}
    if T! isa T
        return T!
    end
    return SwizzledArray{T}(sz)
end

@inline SwizzledArray{T}(sz::SwizzledArray{S, N, Arg, mask, imask, Op}) where {T, S, N, Arg, mask, imask, Op} = SwizzledArray{T, N, Arg, mask, imask, Op}(arg, op)

@inline SwizzledArray(arg, mask, op) = SwizzledArray(_SwizzledArray(Any, arg, Val(mask), op))

@inline SwizzledArray{T}(arg::Arg, mask, op) = _SwizzledArray(T, arg, Val(mask), op)

@inline function _SwizzledArray(::Type{T}, arg::Arg, ::Val{mask}, op::Op) where {T, N, mask, Op, Arg <: AbstractArray{T, N}}
    if @generated
        mask! = (take(flatten((mask, repeated(drop))), N)...,)
        imask = setindexinto(ntuple(d->drop, maximum((0, mask!...))), 1:length(mask!), mask!)
        return :(return SwizzledArray{T, N, Arg, $mask!, $imask, Op}(arg, op))
    else
        mask! = (take(flatten((mask, repeated(drop))), N)...,)
        imask = setindexinto(ntuple(d->drop, maximum((0, mask!...))), 1:length(mask!), mask!)
        return SwizzledArray{T, N, typeof(arg), Val(mask!), imask, typeof(op)}(arg, op)
    end
end

mask(::Type{SwizzledArray{Arg, _mask, _imask, Op}}) where {Arg, _mask, _imask, Op} = _mask
mask(sz::S) where {S <: SwizzledArray} = mask(S)
imask(::Type{SwizzledArray{Arg, _mask, _imask, Op}}) where {Arg, _mask, _imask, Op} = _imask
imask(sz::S) where {S <: SwizzledArray} = imask(S)

struct Swizzler{mask, Op} <: WrappedArrayConstructor
    op::Op
end

"""
    `Swizzler(mask, op=nooperator)`

Produce an object `s` such that when `s` is broadcasted as a function over an
argument `arg`, the result is a lazy view of the result of `swizzle(arg, mask,
op)`.

See also: [`swizzle`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Swizzler((1,), +).(A)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> Swizzler((), +).(A)
55
julia> Swizzler((2,)).(parse.(Int, ["1", "2"]))
1x2-element Array{Int64,1}:
 1 2
```
"""
@inline Swizzler(mask, op::Op) where {Op} = Swizzler{mask, Op}(op)

mask(::Type{Swizzler{_mask, Op}}) where {_mask, Op} = _mask
mask(sz::S) where {S <: Swizzler} = mask(S)

@inline (sz::Swizzler)(arg::AbstractArray) = SwizzledArray(_SwizzledArray(Any, arg, Val(mask(sz)), sz.op))



function Base.show(io::IO, sz::SwizzledArray)
    print(io, SwizzledArray)
    print(io, '(', sz.arg, ", ", mask(sz), ", ", sz.op, ')')
    nothing
end

@inline function Base.axes(sz::SwizzledArray)
    if @generated
        args = getindexinto(ntuple(d->:(Base.OneTo(1)), length(imask(sz))), ntuple(d->:(bc_axes[$d]), length(mask(sz))), imask(sz))
        return quote
            return ($(args...),)
        end
    else
        getindexinto(ntuple(d->Base.OneTo(1), length(imask(sz))), axes(sz.arg), imask(sz))
    end
end



Base.@propagate_inbounds function _swizzle_getindex(sz::SwizzledArray, I::Tuple{Vararg{Int}})
    @boundscheck checkbounds_indices(Bool, axes(sz), I) || throw_boundserror(sz, I)
    if @generated
        args = getindexinto(ntuple(d->:(bc_axes[$d]), length(mask(sz))), ntuple(d->:(I[$d]), length(imask(sz))), mask(sz))
        quote
            bc_axes = broadcast_axes(sz.arg)
            arg_I = ($(args...),)
        end
    else
        arg_I = getindexinto(broadcast_axes(sz.arg), I, mask(sz))
    end
    if sz.op isa typeof(nooperator)
        return @inbounds getindex(sz.arg, map(first, arg_I)...)
    else
        (i, inds) = peel(product(arg_I...))
        res = @inbounds getindex(sz.arg, i...)
        for i in inds
            res = sz.op(res, @inbounds getindex(sz.arg, i...))
        end
        return res
    end
end

Base.@propagate_inbounds Base.getindex(sz::SwizzledArray, I::Int) = _swizzle_getindex(sz, (I,))
Base.@propagate_inbounds Base.getindex(sz::SwizzledArray, I::CartesianIndex) = _swizzle_getindex(sz, Tuple(I))
Base.@propagate_inbounds Base.getindex(sz::SwizzledArray, I::Int...) = _swizzle_getindex(sz, I)
Base.@propagate_inbounds Base.getindex(sz::SwizzledArray) = _swizzle_getindex(sz, ())

"""
    `swizzle(A, mask, op=nooperator)`

Create a new object `B` such that the dimension `i` of `A` is mapped to
dimension `mask[i]` of `B`. If `mask[i]` is an instance of the singleton type
`Drop`, the dimension is reduced over using `op`. `mask` may be any (possibly
infinite) iterable over elements of type `Int` and `Drop`. The integers in
`mask` must be unique, and if `mask` is not long enough, additional `Drop`s are
added to the end.
The resulting container type from `copy(B)` is established by the following rules:
 - If all elements of `mask` are `Drop`, it returns an unwrapped scalar.
 - All other combinations of arguments default to returning an `Array`, but
   custom container types can define their own implementation rules to
   customize the result when they appear as an argument.
The swizzle operation is represented with a special lazy `SwizzledArray` type.
`swizzle` results in `copy(SwizzledArray(...))`.  The swizzle operation can use the
`Swizzler` type to take advantage of special broadcast syntax. A statement like:
```
   y = Swizzler((1,), +).(x .* (Swizzler((2, 1)).x .+ 1))
```
will result in code that is essentially:
```
   y = copy(SwizzledArray(BroadcastedArray(Broadcasted(*, SwizzledArray(x, (2, 1)), Broadcasted(+, x, 1))), (1,), +))
```
If `SwizzledArray`s are mixed with `Broadcasted`s, the result is fused into one big operation.

See also: [`swizzle!`](@ref), [`Swizzler`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> swizzle(A, (1,), +)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> swizzle(A, (), +)
55
julia> swizzle(parse.(Int, ["1", "2"]), (2,))
1x2-element Array{Int64,1}:
 1 2
```
"""
swizzle(A, mask, op=nooperator) = copy(SwizzledArray(A, mask, op))

"""
    `swizzle!(dest, A, mask, op=nooperator)`

Like [`swizzle`](@ref), but store the result of `swizzle(A, mask, op)` in the
`dest` array.  Results in `copyto!(dest, SwizzledArray(...))`.

See also: [`swizzle`](@ref), [`Swizzler`](@ref).

# Examples
```jldoctest
julia> B = [1; 2; 3; 4; 5]
5x1-element Array{Int64,1}:
 1
 2
 3
 4
 5
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> swizzle!(B, A, (1,), +)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
julia> B
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
```
"""
swizzle!(dest, A, mask, op=nooperator) = copyto!(dest, SwizzledArray(A, mask, op))

@inline Base.copy(sz::SwizzledArray) = copy(instantiate(Broadcasted(identity, (sz,))))

@inline Base.copyto!(dest, sz::SwizzledArray) = copyto!(dest, instantiate(Broadcasted(identity, (sz,))))

@inline Base.copyto!(dest::AbstractArray, sz::SwizzledArray) = copyto!(dest, instantiate(Broadcasted(identity, (sz,))))

#function Base.Broadcast.preprocess(dest, sz::SwizzledArray{Arg, mask, imask, Op}) where {Arg, mask, imask, Op}
#    arg = preprocess(dest, sz.arg)
#    SwizzledArray{typeof(arg), mask, imask, Op}(arg, sz.op)
#end

#=
"""
    `SwizzleStyle(style, ::Type{<:SwizzledArray})`

Broadcast styles are used to determine behavior of objects under broadcasting.
To customize the broadcasting behavior of a type under swizzling, one can first
define an appropriate Broadcast style for the the type, then declare how the
broadcast style should behave under broadcasting after the swizzle by
overriding the `SwizzleStyle` method.
"""
SwizzleStyle #FIXME only define on ur stuff

SwizzleStyle(::Style{Tuple}, Sz) = first(mask(Sz)) == 1 ? Style{Tuple}() : DefaultArrayStyle(Val(max(0, first(mask(Sz)))))
Broadcast.longest_tuple(::Nothing, t::Tuple{<:SwizzledArray{<:Any, (1,)},Vararg{Any}}) = longest_tuple(longest_tuple(nothing, (t[1].arg,)), tail(t))
SwizzleStyle(style::A, Sz) where {A <: AbstractArrayStyle{0}} = A(Val(0))
function SwizzleStyle(style::A, ::Type{Sz}) where {N, A <: AbstractArrayStyle{N}, Sz}
    if @generated
        return :(return A(Val($(max(0, maximum(take(mask(Sz), N)))))))
    else
        return A(Val(max(0, maximum(take(mask(Sz), N)))))
    end
end
SwizzleStyle(style::AbstractArrayStyle{Any}, Sz) = style
SwizzleStyle(::BroadcastStyle, Sz) = Unknown()
SwizzleStyle(::ArrayConflict, Sz) = ArrayConflict() #FIXME

@inline Broadcast.BroadcastStyle(Sz::Type{SwizzledArray{Arg, mask, imask, Op}}) where {Arg, mask, imask, Op} = SwizzleStyle(BroadcastStyle(Arg), Sz)
pain and suffering
=#

#=
function _SwizzleStyle(style, sz)
    
end
=#
