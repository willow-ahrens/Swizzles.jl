module Swizzles

export loop
export Nil, nil
export arrayify
export Guard
export Swizzle, Focus
export Pour, Beam
export Reduce, Sum
export Drop, SumOut
export Delay, Intercept

include("base.jl")
include("util.jl")
include("properties.jl")

include("loop.jl")
include("BoxArrays.jl")
include("WrapperArrays.jl")
include("NullArrays.jl")
include("GeneratedArrays.jl")
include("ArrayifiedArrays.jl")
include("ShallowArrays.jl")
include("ExtrudedArrays.jl")
include("SwizzledArrays.jl")




struct Swizzle{Op, mask} <: Swizzles.Intercept
    op::Op
end

"""
    `Swizzle(op, mask...)`
    `Swizzle(op, mask)`

Create a function, `S`, which creates lazily reduced arrays using the operator
op. `S(A)` should produce an object which represents the reduction of `A` into a
result array `R`. `S(Z, A)` represents the reduction of `A` into a result array
`R` which has been initialized as `R .= Z`. Dimension `i` of `R` corresponds to
dimension `mask[i]` of `A` (if it exists). Dimensions of `A` which do not appear
in `mask` are reduced out. If `Z` is unspecified, the `initial` function is used
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

See also: [`reduce`](@ref), [`Guard`](@ref)

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
@inline Swizzle(op::Op, _mask...) where {Op} = Swizzle{Op, _mask}(op)
@inline Swizzle(op::Op, _mask::Tuple) where {Op} = Swizzle{Op, _mask}(op)
@inline Swizzle(op::Op, ::Val{_mask}) where {Op, _mask} = Swizzle{Op, _mask}(op)

@inline function Properties.initial(ctr::Swizzle{<:Any, Op}, arg) where {Op}
    init = Properties.initial(ctr.op, eltype(arg))
    if init === nothing
        return nothing
    end
    return Ref(something(init))
end

@inline (ctr::Swizzle)(arg) = ctr(arrayify(arg))
@inline function (ctr::Swizzle{Op, _mask})(arg::AbstractArray) where {Op, _mask}
    init = Properties.initial_value(ctr, arg)
    if init === nothing
        return Swizzle{Guard{Op}, _mask}(Guard(ctr.op))(Ref(nothing), arg)
    end
    return ctr(init, arg)
end
@inline function (ctr::Swizzle{Nothing, _mask})(arg::AbstractArray) where {_mask}
    return Swizzle{Nothing, _mask}(nothing)(Ref(nothing), arg)
end
@inline (ctr::Swizzle)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function(ctr::Swizzle{Op, _mask})(init::Init, arg::Arg) where {Op, _mask, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = map(d -> d > ndims(arg) ? nil : d, _mask)
        return quote
            arr = SwizzledArray{Any, $(length(mask)), Op, $mask, Init, Arg}(ctr.op, init, arg)
            return convert(SwizzledArray{Properties.eltype_bound(arr)}, arr)
        end
    else
        mask = map(d -> d > ndims(arg) ? nil : d, _mask)
        arr = SwizzledArray{Any, length(mask), Op, mask, Init, Arg}(ctr.op, init, arg)
        return convert(SwizzledArray{Properties.eltype_bound(arr)}, arr)
    end
end

"""
    `Focus(mask...)`
    `Focus(mask)`

Similar to `Swizzle`, but the resulting operator `Y` creates lazy
transformations which do not reduce. Dimensions of the argument which do not
appear in `mask` are asserted to have length `1`.

See also: [`Swizzle`](@ref), [`permutedims`](@ref).

# Examples
```jldoctest
julia> A = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> Focus(nil, nil, 2).(A)
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
@inline Focus(_mask...) = Swizzle(nothing, _mask...)



struct Pour{Op, _imask} <: Swizzles.Intercept
    op::Op
end

"""
    `Pour(op, imask...)`
    `Pour(op, imask)`

Similar to [`Swizzle`](@ref), but the mask is "inverted". Creates an operator
which lazily reduces the input `A`. Dimension `i` of `A` is associated with
dimension `imask[i]` of the result. If `imask[i]` is `nil` (an instance of the
singleton type `Nil`) this dimension of `A` is reduced over. If
`i > length(imask)`, `imask[i]` is assumed to be `nil`.

See also: [`Swizzle`](@ref)

# Examples
```jldoctest
julia> A = [1 2; 3 4; 5 6; 7 8; 9 10]
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
julia> Pour(+, 2).(A)
1x5 Array{Int64,2}:
 3  7  11  15  19
julia> Pour(+, nil, 3).(A)
1×1×2 Array{Int64,3}:
[:, :, 1] =
 25
[:, :, 2] =
 30
```
"""
@inline Pour(op::Op, _imask...) where {Op} = Pour{Op, _imask}(op)
@inline Pour(op::Op, _imask::Tuple) where {Op} = Pour{Op, _imask}(op)
@inline Pour(op::Op, ::Val{_imask}) where {Op, _imask} = Pour{Op, _imask}(op)

@inline function parse_beam_mask(arr, _imask::Tuple{Vararg{Union{Int, Nil}}})
    N = max(0, _imask[1:min(ndims(arr), length(_imask))]...)
    return imasktuple(d->nil, identity, _imask, N)
end
@inline (ctr::Pour)(arg) = ctr(arrayify(arg))
@inline (ctr::Pour)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function(ctr::Pour{Op, _imask})(arg::Arg) where {Op, _imask, Arg <: AbstractArray}
    if @generated
        mask = parse_beam_mask(Arg, _imask)
        return :(return Swizzle{Op, $mask}(ctr.op)(arg))
    else
        mask = parse_beam_mask(arg, _imask)
        return Swizzle(ctr.op, mask)(arg)
    end
end
@inline function(ctr::Pour{Op, _imask})(init::Init, arg::Arg) where {Op, _imask, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_beam_mask(Arg, _imask)
        return :(return Swizzle{Op, $mask}(ctr.op)(init, arg))
    else
        mask = parse_beam_mask(arg, _imask)
        return Swizzle(ctr.op, mask)(init, arg)
    end
end



"""
    `Beam(imask...)`
    `Beam(imask)`

Similar to `Pour`, but the resulting operator `Y` creates lazy
transformations which do not reduce. Dimensions of the argument which do not
have integer destinations in `imask` are asserted to have length `1`.

See also: [`Pour`](@ref).


# Examples
```jldoctest
julia> A = [1 2 3 4 5]
1×5 Array{Int64,2}:
 1  2  3  4  5
julia> Beam(nil, 3).(A)
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
@inline Beam(_imask...) = Pour(nothing, _imask...)



struct Drop{Op, dims} <: Swizzles.Intercept
    op::Op
end

"""
    `Drop(op, dims...)`
    `Drop(op, dims)`

Similar to `Swizzle`, but the resulting operator reduces only over the
dimensions listed in `dims`, dropping those dimensions. `Drop(op)` produces an
operator which reduces over all dimensions.

See also: [`Swizzle`](@ref), [`reduce`](@ref), [`dropdims`](@ref).

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
@inline Drop(op::Op, dims...) where {Op} = Drop{Op, dims}(op)
@inline Drop(op) = Drop(op, Colon())
@inline Drop(op::Op, dims::Colon) where {Op} = Drop{Op, dims}(op)
@inline Drop(op::Op, dims::Tuple) where {Op} = Drop{Op, dims}(op)
@inline Drop(op::Op, ::Val{dims}) where {Op, dims} = Drop{Op, dims}(op)

@inline (ctr::Drop)(arg) = ctr(arrayify(arg))
@inline function parse_drop_mask(arr, dims::Tuple{Vararg{Int}})
    return (setdiff(1:ndims(arr), dims)...,)
end
@inline parse_drop_mask(arr, ::Colon) = ()
@inline function(ctr::Drop{Op, dims})(arg::Arg) where {Op, dims, Arg <: AbstractArray}
    if @generated
        mask = parse_drop_mask(Arg, dims)
        return :(return Swizzle{Op, $mask}(ctr.op)(arg))
    else
        mask = parse_drop_mask(arg, dims)
        return Swizzle(ctr.op, mask)(arg)
    end
end
@inline (ctr::Drop)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function(ctr::Drop{Op, dims})(init::Init, arg::Arg) where {Op, dims, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_drop_mask(Arg, dims)
        return :(return Swizzle{Op, $mask}(ctr.op)(init, arg))
    else
        mask = parse_drop_mask(arg, dims)
        return Swizzle(ctr.op, mask)(init, arg)
    end
end

"""
    `SumOut(dims...)`
    `SumOut(dims)`

Similar to [`Drop`](@ref), but `op` is set to `+`.

See also: [`Drop`](@ref).
"""
@inline SumOut(dims...) = Drop(Base.FastMath.add_fast, dims...)



struct Reduce{Op, dims} <: Swizzles.Intercept
    op::Op
end

"""
    `Reduce(op, dims...)`
    `Reduce(op, dims)`

Similar to `Swizzle`, but the resulting operator reduces only over the
dimensions listed in `dims`, reduceping those dimensions. `Reduce(op)` produces an
operator which reduces over all dimensions.

See also: [`Swizzle`](@ref), [`reduce`](@ref).

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
@inline Reduce(op::Op, dims...) where {Op} = Reduce{Op, dims}(op)
@inline Reduce(op) = Reduce(op, Colon())
@inline Reduce(op::Op, dims::Colon) where {Op} = Reduce{Op, dims}(op)
@inline Reduce(op::Op, dims::Tuple) where {Op} = Reduce{Op, dims}(op)
@inline Reduce(op::Op, ::Val{dims}) where {Op, dims} = Reduce{Op, dims}(op)

@inline (ctr::Reduce)(arg) = ctr(arrayify(arg))
@inline function parse_reduce_mask(arr, dims::Tuple{Vararg{Int}})
    return ntuple(d -> d in dims ? nil : d, Val(ndims(arr)))
end
@inline parse_reduce_mask(arr, ::Colon) = ()
@inline function(ctr::Reduce{Op, dims})(arg::Arg) where {Op, dims, Arg <: AbstractArray}
    if @generated
        mask = parse_reduce_mask(Arg, dims)
        return :(return Swizzle{Op, $mask}(ctr.op)(arg))
    else
        mask = parse_reduce_mask(arg, dims)
        return Swizzle(ctr.op, mask)(arg)
    end
end
@inline (ctr::Reduce)(init, arg) = ctr(arrayify(init), arrayify(arg))
@inline function(ctr::Reduce{Op, dims})(init::Init, arg::Arg) where {Op, dims, Arg <: AbstractArray, Init <: AbstractArray}
    if @generated
        mask = parse_reduce_mask(Arg, dims)
        return :(return Swizzle{Op, $mask}(ctr.op)(init, arg))
    else
        mask = parse_reduce_mask(arg, dims)
        return Swizzle(ctr.op, mask)(init, arg)
    end
end



"""
    `Sum(dims...)`
    `Sum(dims)`

Similar to [`Reduce`](@ref), but `op` is set to `+`.

See also: [`Reduce`](@ref).
"""
@inline Sum(dims...) = Reduce(Base.FastMath.add_fast, dims...)



end
