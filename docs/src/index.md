# Swizzles.jl

## Overview

`Swizzles` is an intermediate representation for array operations written in Julia. `Swizzles` functions harmoniously with Julia's `broadcast` facilities, so that users can use (mostly) familiar syntax to build their own tensor kernels in a high-level language.

For developers, `Swizzles` is an experimental approach to generating efficient fused multidimensional broadcast and reduction operations. Swizzles provides the lazy `SwizzledArray` type, which represents a lazy simultaneous reduction and transposition, and the `GeneratedArray` abstract type, which users can override to take advantage of a fused trait-based implementation of Julia's `AbstractArray` built on lazy array types like `Broadcasted`, `SubArray`, and `SwizzledArray`.

## What is a Swizzle?

`Swizzle(op, mask...)(A)` produces a lazy array `R` where the `n`^{th} dimension of `R` corresponds to dimension `mask[n]` of `A`, and other dimensions have been reduced out using the reduction operator `op`.

```julia
julia> using Swizzles

julia> A = reshape(1:10, 2, 5)
2×5 reshape(::UnitRange{Int64}, 2, 5) with eltype Int64:
 1  3  5  7   9
 2  4  6  8  10

julia> Swizzle(max, 2)(A)
5-element Swizzles.SwizzledArray{Int64,1,typeof(max),(2,),Swizzles.ArrayifiedArrays.ArrayifiedArray{Int64,0,Base.RefValue{Int64}},Base.ReshapedArray{Int64,2,UnitRange{Int64},Tuple{}}}:
  2
  4
  6
  8
 10
```

In the above example, we see that `Swizzle(max, 2)(A)[i] == maximum(A[:, i])`.

`Swizzles` overrides broadcast rules to apply Swizzles to an entire array instead of applying the `Swizzle` pointwise, so we also get

```
julia> Swizzle(max, 2).(A)
5-element Array{Int64,1}:
  2
  4
  6
  8
 10
```

Notice that since we broadcasted, Julia has automagically called `materialize` on the result and we are given a materialized array.

The dimensions in Swizzle masks don't have to occur in any particular order, so `Swizzle(max, 2, 1)` should transpose our array.
```
julia> Swizzle(max, 2, 1).(A)
5×2 Array{Int64,2}:
 1   2
 3   4
 5   6
 7   8
 9  10
```

When we're only permuting dimensions (transposition), we can pass `nothing` as an operator. In the previous example, we could have instead written `Swizzle(max, 2, 1).(A)`.

It can be useful to add dimensions of size `1`. We use `nil`, an instance of the singleton type `Nil`, to denote such dimensions in the masks. Thus, we have that `Swizzle(max, nil, 2).(A) == maximum(A, dims=1)`.

```
julia> Swizzle(max, nil, 2).(A)
1×5 Array{Int64,2}:
 2  4  6  8  10
```

#Swizzle Friends!


`Swizzle` has many friends. `Focus(mask...)` is equivalent to `Swizzle(nothing, mask...)`, a shorthand for swizzling when we are only permuting dimensions.

`Pour` is equivalent to `Swizzle` with an inverse mask. `Pour(op, imask...)(A)` maps dimension `d` of `A` to dimension `imask[d]` of the output, whereas `Swizzle(op, mask...)(A)` maps dimension `d` of the output to dimension `mask[d]` of `A`. We can think of `Pour` as *scattering* dimensions into the result, whereas `Swizzle` *gathers* these dimensions into the result.

We can still use `nil` in `Pour`, except now if `imask[d]` is `nil`, then dimension `d` is reduced. If `imask` is shorter than the number of dimensions of `A`, `imask` is padded with extra `nil` elements. If `imask` is longer than the number of dimensions of `A`, trailing elements of the inverse mask are dropped. The output of `Pour` is an array with a number of dimensions equal to the maximum of `imask`.

`Beam(mask...)` is equivalent to `Pour(nothing, imask...)` a shorthand for when we are only permuting dimensions. Thus, we have the gather/scatter pairs `Swizzle`/`Pour` and `Focus`/`Beam`. Consider the following exciting examples:

```
julia> A = reshape(1:10, 2, 5)
2×5 reshape(::UnitRange{Int64}, 2, 5) with eltype Int64:
 1  3  5  7   9
 2  4  6  8  10

julia> Swizzle(max, 1).(A)
2-element Array{Int64,1}:
  9
 10

julia> Pour(max, 1, nil).(A)
2-element Array{Int64,1}:
  9
 10

julia> Pour(max, 3, nil).(A)
1×1×2 Array{Int64,3}:
[:, :, 1] =
 9

[:, :, 2] =
 10

julia> Swizzle(max, nil, nil, 1).(A)
1×1×2 Array{Int64,3}:
[:, :, 1] =
 9

[:, :, 2] =
 10

julia> Focus(2, nil, 1).(A)
5×1×2 Array{Int64,3}:
[:, :, 1] =
 1
 3
 5
 7
 9

[:, :, 2] =
  2
  4
  6
  8
 10

julia> Beam(3, 1).(A)
5×1×2 Array{Int64,3}:
[:, :, 1] =
 1
 3
 5
 7
 9

[:, :, 2] =
  2
  4
  6
  8
 10
```

In addition to the inverse masks, `Swizzles` also have friends which mimic the behavior of Julia's `reduce` function. `Reduce(op, dims...).(A)` will produce a result with the dimensions of `A` which occur in `dims` having been reduced. If no dimensions are specified, all dimensions of `A` are reduced. Thus,

```
julia> Reduce(max, 1).(A)
1×5 Array{Int64,2}:
 2  4  6  8  10
```

If we want the reduced dimensions to disappear and collapse remaining dimensions downward (similar to Julia's `dropdims` function), we can use the `Drop(op, dims...)` operator.

```
julia> Drop(max, 1).(A)
5-element Array{Int64,1}:
  2
  4
  6
  8
 10
```

Although they have different semantics, all of the friends of Swizzles eventually call `Swizzle` with the appropriate mask and are implemented under the hood using `SwizzledArray`.

## Fusion

`SwizzledArrays` can fuse easily with 



 example
