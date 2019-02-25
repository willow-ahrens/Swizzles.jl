module Swizzles

export Drop, drop
export masktuple, imasktuple #maybe dont export...
export ArrayifiedArray, arrayify
export SwizzledArray, mask
export swizzle, swizzle!
export Swizzle, Reduce, Sum, Max, Min, Beam
export SwizzleTo, ReduceTo, SumTo, MaxTo, MinTo, BeamTo
export Delay, Intercept

include("base.jl")
include("util.jl")
include("properties.jl")
include("Antennae.jl")

include("WrapperArrays.jl")
include("GeneratedArrays.jl")
include("ArrayifiedArrays.jl")
include("ShallowArrays.jl")
include("ExtrudedArrays.jl")
include("SwizzledArrays.jl")



struct Swizzle{T, Op, _mask} <: Swizzles.Intercept
    op::Op
end

"""
    `Swizzle(op, mask...)`
    `Swizzle(op, mask)`

Create an operator which maps `A` to a lazy array `B` such that the dimension
`i` of `A` is mapped to dimension `mask[i]` of `B`. If `mask[i]` is an instance
of the singleton type `Drop`, the dimension is reduced over using `op`. `mask`
may be any splatted or unsplatted `Tuple` of `Int` and `Drop`. The integers in
`mask` must be unique, and if `mask` is not long enough, additional `Drop`s are
added to the end.
The resulting container type from `materialize(B)` is established by the following rules:
 - If all elements of `mask` are `Drop`, it returns an unwrapped scalar.
 - All other combinations of arguments default to returning an `Array`, but
   custom container types can define their own implementation rules to
   customize the result when they appear as an argument.
The swizzle operation is represented with a special lazy `SwizzledArray` type. A
`Swizzle` will take advantage of special broadcast syntax. Broadcasting a
`Swizzle` over an array `A` will instead apply the `Swizzle` to `A`. Thus, a
statement like:
```
   y = 3.0 .+ Swizzle(+, 1).(x .* 2))
```
will result in code that is essentially:
```
   y = materialize(Broadcasted(+, (3.0,
     SwizzledArray(ArrayifiedArray(Broadcasted(*, (x, 2))), +, (1,)))))
```
If `SwizzledArray`s are mixed with `Broadcasted`s, the result is fused into one
big happy operation. Use `BroadcastStyles` to customize the behavior for your
array types.

See also: [`Swizzle{T}`](@ref)

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Swizzle(+, 1).(A)
5-element Array{Int64,1}:
  3
  7
 11
 15
 19
julia> Swizzle(+).(A)
55
julia> Swizzle(+, 2).(parse.(Int, ["1", "2"]))
1x2-element Array{Int64,1}:
 1 2
```
"""
@inline Swizzle(op, _mask...) = Swizzle{nothing}(op, _mask...)

"""
    `Swizzle{T}(op, mask...)`
    `Swizzle{T}(op, mask)`

Similar to [`Swizzle`](@ref), but the eltype of the result (and all intermediate
reduction results) is declared to be `T`.

See also: [`Swizzle`](@ref).
"""
@inline Swizzle{T}(op::Op, _mask...) where {T, Op} = Swizzle{T, Op, _mask}(op)
@inline Swizzle{T}(op::Op, _mask::Tuple) where {T, Op} = Swizzle{T, Op, _mask}(op)
@inline Swizzle{T}(op::Op, ::Val{_mask}) where {T, Op, _mask} = Swizzle{T, Op, _mask}(op)



"""
    `nooperator(a, b)`

An operator which does not expect to be called. It startles easily.
"""
nooperator(a, b) = throw(ArgumentError("unspecified operator"))

struct Guard{Op}
    op::Op
end

(op::Guard)(x::Nothing, y) = y
(op::Guard)(x, y) = op.op(x, y)
@inline Properties.return_type(g::Guard, T, S) = Properties.return_type(g.op, T, S)
@inline Properties.return_type(g::Guard, ::Type{Union{Nothing, T}}, S) where {T} = Properties.return_type(g.op, T, S)
@inline Properties.return_type(g::Guard, ::Type{Nothing}, S) = S

@inline Properties.initial(::Guard, ::Any) = Some(nothing)



@inline function Properties.initial(ctr::Swizzle{<:Any, Op}, arg) where {Op}
    init = Properties.initial(ctr.op, eltype(arg))
    if init === nothing
        return nothing
    end
    return Ref(something(init))
end

@inline (ctr::Swizzle)(arg) = ctr(arrayify(arg))
@inline function (ctr::Swizzle{T, Op, _mask})(arg::AbstractArray) where {T, Op, _mask}
    init = Properties.initial_value(ctr, arg)
    if init === nothing
        return Swizzle{T, Guard{Op}, _mask}(Guard(ctr.op))(Ref(nothing), arg)
    end
    return ctr(init, arg)
end
@inline (ctr::Swizzle)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function (ctr::Swizzle{nothing, Op, _mask})(init::AbstractArray, arg::AbstractArray) where {Op, _mask}
    arr = Swizzle{Any, Op, _mask}(ctr.op)(init, arg)
    return Swizzle{Properties.eltype_bound(arr), Op, _mask}(ctr.op)(init, arg)
end
@inline function parse_swizzle_mask(arr, _mask::Tuple{Vararg{Union{Int, Drop}, M}}) where {M}
    return ntuple(d -> d <= M ? _mask[d] : drop, Val(ndims(arr)))
end
@inline function(ctr::Swizzle{T, Op, _mask})(init::Init, arg::Arg) where {T, Op, _mask, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_swizzle_mask(Arg, _mask)
        return :(return SwizzledArray{T, $(max(0, mask...)), Op, $mask, Arg, Init}(ctr.op, arg, init))
    else
        mask = parse_swizzle_mask(arg, _mask)
        return SwizzledArray{T, max(0, mask...), Op, mask, Arg, Init}(ctr.op, arg, init)
    end
end



struct Beam{T}
"""
    `Beam{T}(mask...)`
    `Beam{T}(mask)`

Similar to [`Beam`](@ref), but the eltype is declared to be `T`.

See also: [`Beam`](@ref).
"""
    @inline Beam{T}(_mask...) where {T} = Swizzle{T}(Guard(nooperator), _mask...)
end

"""
    `Beam(mask...)`
    `Beam(mask)`

Create an operator which maps `A` to a lazy array `B` such that the dimension
`i` of `A` is mapped to dimension `mask[i]` of `B`. If dimension `i` of `A` is
known to have size `1`, it may be dropped by setting `mask[i] = drop`.

See also: [`Swizzle`](@ref), [`Beam{T}`](@ref).

# Examples
```jldoctest
julia> A = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> Beam(drop, 3).(A)
1×1×5 Array{Int64,3}:
[:, :, 1] =
 1
[:, :, 2] =
 2
[:, :, 3] =
 3
[:, :, 4] =
 4
[:, :, 5] =
 5
```
"""
@inline Beam(_mask...) = Beam{nothing}(_mask...)



struct SwizzleTo{T, Op, _imask} <: Swizzles.Intercept
    op::Op
end

"""
    `SwizzleTo(op, imask...)`
    `SwizzleTo(op, imask)`

Similar to [`Swizzle`](@ref), but the mask is "inverted". Creates an operator
which maps `A` to a lazy array `B` such that the dimension `imask[i]` of `A` is mapped
to dimension `i` of `B`. If `imask[i]` is an instance of the singleton type
`Drop`, a dimension of size `1` is inserted in that position. Dimensions which
do not appear in `imask` are reduced over using `op`.

See also: [`Swizzle`](@ref), [`SwizzleTo{T}`](@ref)

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> SwizzleTo(+, drop, 1).(A)
1x5 Array{Int64,2}:
 3  7  11  15  19
julia> SwizzleTo(+, drop, drop, 2).(A)
1×1×2 Array{Int64,3}:
[:, :, 1] =
 25
[:, :, 2] =
 30
```
"""
@inline SwizzleTo(op, _imask...) = SwizzleTo{nothing}(op, _imask...)

"""
    `SwizzleTo{T}(op, imask...)`
    `SwizzleTo{T}(op, imask)`

Similar to [`SwizzleTo`](@ref), but the eltype of the result (and all
intermediate reduction results) is declared to be `T`.

See also: [`SwizzleTo`](@ref).
"""
@inline SwizzleTo{T}(op::Op, _imask...) where {T, Op} = SwizzleTo{T, Op, _imask}(op)
@inline SwizzleTo{T}(op::Op, _imask::Tuple) where {T, Op} = SwizzleTo{T, Op, _imask}(op)
@inline SwizzleTo{T}(op::Op, ::Val{_imask}) where {T, Op, _imask} = SwizzleTo{T, Op, _imask}(op)

@inline (ctr::SwizzleTo)(arg) = ctr(arrayify(arg))
@inline (ctr::SwizzleTo)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function(ctr::SwizzleTo{T, Op, _imask})(arg::Arg) where {T, Op, _imask, Arg <: AbstractArray}
    if @generated
        mask = parse_swizzle_mask(Arg, imasktuple(d->drop, identity, _imask))
        return :(return Swizzle{T, Op, $mask}(ctr.op)(arg))
    else
        mask = parse_swizzle_mask(arg, imasktuple(d->drop, identity, _imask))
        return Swizzle{T}(ctr.op, mask)(arg)
    end
end
@inline function(ctr::SwizzleTo{T, Op, _imask})(init::Init, arg::Arg) where {T, Op, _imask, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_swizzle_mask(Arg, imasktuple(d->drop, identity, _imask))
        return :(return Swizzle{T, Op, $mask}(ctr.op)(init, arg))
    else
        mask = parse_swizzle_mask(arg, imasktuple(d->drop, identity, _imask))
        return Swizzle{T}(ctr.op, mask)(init, arg)
    end
end



struct BeamTo{T}
"""
    `BeamTo{T}(mask...)`
    `BeamTo{T}(mask)`

Similar to [`BeamTo`](@ref), but the eltype is declared to be `T`.

See also: [`BeamTo`](@ref).
"""
    @inline BeamTo{T}(_imask...) where {T} = SwizzleTo{T}(Guard(nooperator), _imask...)
end

"""
    `BeamTo(imask...)`
    `BeamTo(imask)`

Create an operator which maps `A` to a lazy array `B` such that the dimension
`imask[i]` of `A` is mapped to dimension `i` of `B`. To insert a dimension of
size `1` at dimension `i` of `B`, set `imask[i] = drop`. Dimensions of `A` which
do not appear in `imask` are assumed to have size `1`.

See also: [`Beam`](@ref), [`BeamTo{T}`](@ref).

# Examples
```jldoctest
julia> A = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> BeamTo(drop, drop, 2).(A)
1×1×5 Array{Int64,3}:
[:, :, 1] =
 1
[:, :, 2] =
 2
[:, :, 3] =
 3
[:, :, 4] =
 4
[:, :, 5] =
 5
```
"""
@inline BeamTo(_imask...) = BeamTo{nothing}(_imask...)



struct Reduce{T, Op, dims} <: Swizzles.Intercept
    op::Op
end

"""
    `Reduce(op, dims...)`
    `Reduce(op, dims)`

Create an operator which maps `A` to a lazy array `B` such that the dimensions
`dims` of `A` are reduced over using `op`, collapsing remaining dimensions
downward. If `dims` is empty, all dimensions are reduced over.

See also: [`Swizzle`](@ref), [`Reduce{T}`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Reduce(+, 2).(A)
5×1 Array{Int64,2}:
 3
 7
 11
 15
 19
```
"""
@inline Reduce(op, dims...) = Reduce{nothing}(op, dims...)

"""
    `Reduce{T}(op, dims...)`
    `Reduce{T}(op, dims)`

Similar to [`Reduce`](@ref), but the eltype of the result (and all intermediate
reduction results) is declared to be `T`.

See also: [`Reduce`](@ref).
"""
@inline Reduce{T}(op::Op, dims...) where {T, Op} = Reduce{T, Op, dims}(op)
@inline Reduce{T}(op::Op, dims::Tuple) where {T, Op} = Reduce{T, Op, dims}(op)
@inline Reduce{T}(op::Op, ::Val{dims}) where {T, Op, dims} = Reduce{T, Op, dims}(op)

@inline (ctr::Reduce)(arg) = ctr(arrayify(arg))
@inline function parse_reduce_mask(arr, dims::Tuple{Vararg{Int}}) where {M, N}
    c = 0
    return ntuple(d -> d in dims ? drop : c += 1, Val(ndims(arr)))
end
@inline function parse_reduce_mask(arr, dims::Tuple{}) where {M, N}
    return ntuple(d -> drop, Val(ndims(arr)))
end
@inline function(ctr::Reduce{T, Op, dims})(arg::Arg) where {T, Op, dims, Arg <: AbstractArray}
    if @generated
        mask = parse_reduce_mask(Arg, dims)
        return :(return Swizzle{T, Op, $mask}(ctr.op)(arg))
    else
        mask = parse_reduce_mask(arg, dims)
        return Swizzle{T}(ctr.op, mask)(arg)
    end
end
@inline function(ctr::Reduce{T, Op, dims})(init::Init, arg::Arg) where {T, Op, dims, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_reduce_mask(Arg, dims)
        return :(return Swizzle{T, Op, $mask}(ctr.op)(init, arg))
    else
        mask = parse_reduce_mask(arg, dims)
        return Swizzle{T}(ctr.op, mask)(init, arg)
    end
end



struct Sum{T}
"""
    `Sum{T}(dims...)`
    `Sum{T}(dims)`

Similar to [`Sum`](@ref), but the eltype of the result (and all intermediate
reduction results) is declared to be `T`.

See also: [`Sum`](@ref).
"""
    @inline Sum{T}(dims...) where {T} = Reduce{T}(Base.FastMath.add_fast, dims...)
end

"""
    `Sum(dims...)`
    `Sum(dims)`

Similar to [`Reduce`](@ref), but `op` is set to `+`.

See also: [`Reduce`](@ref), [`Sum{T}`](@ref).
"""
@inline Sum(dims...) = Sum{nothing}(dims...)



struct Max{T}
"""
    `Max{T}(dims...)`
    `Max{T}(dims)`

Similar to [`Max`](@ref), but the eltype of the result (and all intermediate
reduction results) is declared to be `T`.

See also: [`Max`](@ref).
"""
    @inline Max{T}(dims...) where {T} = Reduce{T}(max, dims...)
end

"""
    `Max(dims...)`
    `Max(dims)`

Similar to [`Reduce`](@ref), but `op` is set to `max`.

See also: [`Reduce`](@ref), [`Max{T}`](@ref).
"""
@inline Max(dims...) = Max{nothing}(dims...)



struct Min{T}
"""
    `Min{T}(dims...)`
    `Min{T}(dims)`

Similar to [`Min`](@ref), but the eltype of the result (and all intermediate
reduction results) is declared to be `T`.

See also: [`Min`](@ref).
"""
    @inline Min{T}(dims...) where {T} = Reduce{T}(min, dims...)
end

"""
    `Min(dims...)`
    `Min(dims)`

Similar to [`Reduce`](@ref), but `op` is set to `min`.

See also: [`Reduce`](@ref), [`Min{T}`](@ref).
"""
@inline Min(dims...) = Min{nothing}(dims...)

end
