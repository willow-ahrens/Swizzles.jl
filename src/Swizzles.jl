module Swizzles

export Nil, nil
export arrayify
export Guard
export Swizzle, Yoink
export Beam, Yeet
export Reduce, Sum
export Drop, SumOut
export Delay, Intercept

include("base.jl")
include("util.jl")
include("properties.jl")

include("BoxArrays.jl")
include("WrapperArrays.jl")
include("NullArrays.jl")
include("GeneratedArrays.jl")
include("ArrayifiedArrays.jl")
include("ShallowArrays.jl")
include("ExtrudedArrays.jl")
include("SwizzledArrays.jl")



struct Guard{Op}
    op::Op
end

(op::Guard)(x::Nothing, y) = y
(op::Guard)(x, y) = op.op(x, y)
@inline Properties.return_type(g::Guard, T, S) = Properties.return_type(g.op, T, S)
@inline Properties.return_type(g::Guard, ::Type{Union{Nothing, T}}, S) where {T} = Properties.return_type(g.op, T, S)
@inline Properties.return_type(g::Guard, ::Type{Nothing}, S) = S

@inline Properties.initial(::Guard, ::Any) = Some(nothing)




struct Swizzle{T, Op, mask} <: Swizzles.Intercept
    op::Op
end

"""
    `Swizzle(op, mask...)`
    `Swizzle(op, mask)`

Create an operator, `S`, which creates lazily reduced arrays. `S(A)` should
produce an object which represents the reduction of `A` into a result array `R`.
`S(Z, A)` represents the reduction of `A` into a result array `R` which has been
initialized as `R .= Z`. Dimension `i` of `R` corresponds to dimension `mask[i]`
of `A` (if it exists). Dimensions of `A` which do not appear in `mask` are
reduced out. If `Z` is unspecified, the `initial` function is used
to create a suitable initial value. If no such initial value is found, the
initial value is `nothing` and `op` is wrapped in a `Guard`.

`Swizzles` can be materialized with `copy` or `copyto!`, and will eagerly fuse
themselves into special broadcast syntax. Broadcasting a `Swizzle` over a
broadcast expression `bc` will instead apply the `Swizzle` directy to `bc`
without materializing `bc` first. Thus, the code:
```
   y = 3.0 .+ Swizzle(+, 1).(x .* 2))
```
will result in one big happy fused operation:
```
   using Base.Broadcast: materialize, broadcasted
   y = materialize(broadcasted(+, 3.0, Swizzle(+, 1)(broadcasted(*, x, 2)))
```

See also: [`Swizzle{T}`](@ref), [`Guard`](@ref)

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
julia> Swizzle(+, nil, 2).(parse.(Int, ["1", "2"]))
1x2-element Array{Int64,1}:
 1 2
```
"""
@inline Swizzle(op, _mask...) = Swizzle{nothing}(op, _mask...)

"""
    `Swizzle{T}(op, mask...)`
    `Swizzle{T}(op, mask)`

Similar to [`Swizzle`](@ref), but the eltype of the result (and all intermediate
reduction results) is asserted to be `T`.

See also: [`Swizzle`](@ref), [`eltype`](@ref).
"""
@inline Swizzle{T}(op::Op, _mask...) where {T, Op} = Swizzle{T, Op, _mask}(op)
@inline Swizzle{T}(op::Op, _mask::Tuple) where {T, Op} = Swizzle{T, Op, _mask}(op)
@inline Swizzle{T}(op::Op, ::Val{_mask}) where {T, Op, _mask} = Swizzle{T, Op, _mask}(op)

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
@inline function(ctr::Swizzle{T, Op, _mask})(init::Init, arg::Arg) where {T, Op, _mask, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = map(d -> d > ndims(arg) ? nil : d, _mask)
        return :(return SwizzledArray{T, $(length(mask)), Op, $mask, Init, Arg}(ctr.op, init, arg))
    else
        mask = map(d -> d > ndims(arg) ? nil : d, _mask)
        return SwizzledArray{T, length(mask), Op, mask, Init, Arg}(ctr.op, init, arg)
    end
end
@inline function (ctr::Swizzle{nothing, Op, _mask})(init::AbstractArray, arg::AbstractArray) where {Op, _mask}
    arr = Swizzle{Any, Op, _mask}(ctr.op)(init, arg)
    return Swizzle{Properties.eltype_bound(arr), Op, _mask}(ctr.op)(init, arg)
end



@inline Properties.initial(::Nothing, ::Any) = Some(nothing)

struct Yoink{T}
"""
    `Yoink{T}(mask...)`
    `Yoink{T}(mask)`

Similar to [`Yoink`](@ref), but the eltype is declared to be `T`.

See also: [`Yoink`](@ref).
"""
    @inline Yoink{T}(_mask...) where {T} = Swizzle{T}(nothing, _mask...)
end

"""
    `Yoink(mask...)`
    `Yoink(mask)`

Similar to `Swizzle`, but the resulting operator `Y` creates lazy
transformations which do not reduce. Dimensions of the argument which do not
appear in `mask` are asserted to have length `1`.

See also: [`Swizzle`](@ref), [`Yoink{T}`](@ref), [`permutedims`](@ref).

# Examples
```jldoctest
julia> A = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> Yoink(nil, nil, 2).(A)
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
@inline Yoink(_mask...) = Yoink{nothing}(_mask...)



struct Beam{T, Op, _imask} <: Swizzles.Intercept
    op::Op
end

"""
    `Beam(op, imask...)`
    `Beam(op, imask)`

Similar to [`Swizzle`](@ref), but the mask is "inverted". Creates an operator
which lazily reduces the input `A`. Dimension `i` of `A` is associated with
dimension `imask[i]` of the result. If `imask[i]` is `nil` (an instance of the
singleton type `Nil`) this dimension of `A` is reduced over. If
`i > length(imask)`, `imask[i]` is assumed to be `nil`.

See also: [`Swizzle`](@ref), [`Beam{T}`](@ref)

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Beam(+, 2).(A)
1x5 Array{Int64,2}:
 3  7  11  15  19
julia> Beam(+, nil, 3).(A)
1×1×2 Array{Int64,3}:
[:, :, 1] =
 25
[:, :, 2] =
 30
```
"""
@inline Beam(op, _imask...) = Beam{nothing}(op, _imask...)

"""
    `Beam{T}(op, imask...)`
    `Beam{T}(op, imask)`

Similar to [`Beam`](@ref), but the eltype of the result (and all
intermediate reduction results) is declared to be `T`.

See also: [`Beam`](@ref).
"""
@inline Beam{T}(op::Op, _imask...) where {T, Op} = Beam{T, Op, _imask}(op)
@inline Beam{T}(op::Op, _imask::Tuple) where {T, Op} = Beam{T, Op, _imask}(op)
@inline Beam{T}(op::Op, ::Val{_imask}) where {T, Op, _imask} = Beam{T, Op, _imask}(op)

@inline function parse_beam_mask(arr, _imask::Tuple{Vararg{Union{Int, Nil}}})
    N = max(0, _imask[1:min(ndims(arr), length(_imask))]...)
    return imasktuple(d->nil, identity, _imask, N)
end
@inline (ctr::Beam)(arg) = ctr(arrayify(arg))
@inline (ctr::Beam)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function(ctr::Beam{T, Op, _imask})(arg::Arg) where {T, Op, _imask, Arg <: AbstractArray}
    if @generated
        mask = parse_beam_mask(Arg, _imask)
        return :(return Swizzle{T, Op, $mask}(ctr.op)(arg))
    else
        mask = parse_beam_mask(arg, _imask)
        return Swizzle{T}(ctr.op, mask)(arg)
    end
end
@inline function(ctr::Beam{T, Op, _imask})(init::Init, arg::Arg) where {T, Op, _imask, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_beam_mask(Arg, _imask)
        return :(return Swizzle{T, Op, $mask}(ctr.op)(init, arg))
    else
        mask = parse_beam_mask(arg, _imask)
        return Swizzle{T}(ctr.op, mask)(init, arg)
    end
end



struct Yeet{T}
"""
    `Yeet{T}(mask...)`
    `Yeet{T}(mask)`

Similar to [`Yeet`](@ref), but the eltype is declared to be `T`.

See also: [`Yeet`](@ref).
"""
    @inline Yeet{T}(_imask...) where {T} = Beam{T}(nothing, _imask...)
end

"""
    `Yeet(imask...)`
    `Yeet(imask)`

Similar to `Beam`, but the resulting operator `Y` creates lazy
transformations which do not reduce. Dimensions of the argument which do not
have integer destinations in `imask` are asserted to have length `1`.

See also: [`Beam`](@ref), [`Yeet{T}`](@ref).


# Examples
```jldoctest
julia> A = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> Yeet(nil, 3).(A)
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
@inline Yeet(_imask...) = Yeet{nothing}(_imask...)



struct Drop{T, Op, dims} <: Swizzles.Intercept
    op::Op
end

"""
    `Drop(op, dims...)`
    `Drop(op, dims)`

Similar to `Swizzle`, but the resulting operator reduces only over the
dimensions listed in `dims`, dropping those dimensions. `Drop(op)` produces an
operator which reduces over all dimensions.

See also: [`Swizzle`](@ref), [`Drop{T}`](@ref), [`reduce`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Drop(+, 2).(A)
3-element Array{Int64,1}:
 3
 7
 11
 15
 19
```
"""
@inline Drop(op, dims...) = Drop{nothing}(op, dims...)

"""
    `Drop{T}(op, dims...)`
    `Drop{T}(op, dims)`

Similar to [`Drop`](@ref), but the eltype of the result (and all intermediate
reduction results) is declared to be `T`.

See also: [`Drop`](@ref).
"""
@inline Drop{T}(op::Op, dims...) where {T, Op} = Drop{T, Op, dims}(op)
@inline Drop{T}(op) where {T} = Drop{T}(op, Colon())
@inline Drop{T}(op::Op, dims::Colon) where {T, Op} = Drop{T, Op, dims}(op)
@inline Drop{T}(op::Op, dims::Tuple) where {T, Op} = Drop{T, Op, dims}(op)
@inline Drop{T}(op::Op, ::Val{dims}) where {T, Op, dims} = Drop{T, Op, dims}(op)

@inline (ctr::Drop)(arg) = ctr(arrayify(arg))
@inline function parse_drop_mask(arr, dims::Tuple{Vararg{Int}})
    return (setdiff(1:ndims(arr), dims)...,)
end
@inline parse_drop_mask(arr, ::Colon) = ()
@inline function(ctr::Drop{T, Op, dims})(arg::Arg) where {T, Op, dims, Arg <: AbstractArray}
    if @generated
        mask = parse_drop_mask(Arg, dims)
        return :(return Swizzle{T, Op, $mask}(ctr.op)(arg))
    else
        mask = parse_drop_mask(arg, dims)
        return Swizzle{T}(ctr.op, mask)(arg)
    end
end
@inline (ctr::Drop)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function(ctr::Drop{T, Op, dims})(init::Init, arg::Arg) where {T, Op, dims, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_drop_mask(Arg, dims)
        return :(return Swizzle{T, Op, $mask}(ctr.op)(init, arg))
    else
        mask = parse_drop_mask(arg, dims)
        return Swizzle{T}(ctr.op, mask)(init, arg)
    end
end

struct SumOut{T}
"""
    `SumOut{T}(dims...)`
    `SumOut{T}(dims)`

Similar to [`SumOut`](@ref), but the eltype of the result (and all intermediate
reduction results) is declared to be `T`.

See also: [`SumOut`](@ref).
"""
    @inline SumOut{T}(dims...) where {T} = Drop{T}(Base.FastMath.add_fast, dims...)
end

"""
    `SumOut(dims...)`
    `SumOut(dims)`

Similar to [`Drop`](@ref), but `op` is set to `+`.

See also: [`Drop`](@ref), [`SumOut{T}`](@ref).
"""
@inline SumOut(dims...) = SumOut{nothing}(dims...)



struct Reduce{T, Op, dims} <: Swizzles.Intercept
    op::Op
end

"""
    `Reduce(op, dims...)`
    `Reduce(op, dims)`

Similar to `Swizzle`, but the resulting operator reduces only over the
dimensions listed in `dims`, reduceping those dimensions. `Reduce(op)` produces an
operator which reduces over all dimensions.

See also: [`Swizzle`](@ref), [`Reduce{T}`](@ref), [`reduce`](@ref).

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
3-element Array{Int64,1}:
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
@inline Reduce{T}(op) where {T} = Reduce{T}(op, Colon())
@inline Reduce{T}(op::Op, dims::Colon) where {T, Op} = Reduce{T, Op, dims}(op)
@inline Reduce{T}(op::Op, dims::Tuple) where {T, Op} = Reduce{T, Op, dims}(op)
@inline Reduce{T}(op::Op, ::Val{dims}) where {T, Op, dims} = Reduce{T, Op, dims}(op)

@inline (ctr::Reduce)(arg) = ctr(arrayify(arg))
@inline function parse_reduce_mask(arr, dims::Tuple{Vararg{Int}})
    return ntuple(d -> d in dims ? nil : d, Val(ndims(arr)))
end
@inline parse_reduce_mask(arr, ::Colon) = ()
@inline function(ctr::Reduce{T, Op, dims})(arg::Arg) where {T, Op, dims, Arg <: AbstractArray}
    if @generated
        mask = parse_reduce_mask(Arg, dims)
        return :(return Swizzle{T, Op, $mask}(ctr.op)(arg))
    else
        mask = parse_reduce_mask(arg, dims)
        return Swizzle{T}(ctr.op, mask)(arg)
    end
end
@inline (ctr::Reduce)(init, arg) = ctr(arrayify(init), arrayify(arg))
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



end
