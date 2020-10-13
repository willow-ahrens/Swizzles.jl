# Swizzles

<!---
[![Travis](https://travis-ci.org/peterahrens/Swizzles.jl.svg?branch=master)](https://travis-ci.org/peterahrens/Swizzles.jl)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva/branch/master?svg=true)](https://ci.appveyor.com/project/peterahrens/swizzles-jl/branch/master)
[![Coveralls](https://coveralls.io/repos/peterahrens/Swizzles.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/peterahrens/Swizzles.jl?branch=master)
[![Codecov](http://codecov.io/github/peterahrens/Swizzles.jl/coverage.svg?branch=master)](http://codecov.io/github/peterahrens/Swizzles.jl?branch=master)
[![pipeline status](https://gitlab.com/peterahrens/Swizzles.jl/badges/master/pipeline.svg)](https://gitlab.com/peterahrens/Swizzles.jl/commits/master)
[![coverage report](https://gitlab.com/peterahrens/Swizzles.jl/badges/master/coverage.svg)](https://gitlab.com/peterahrens/Swizzles.jl/commits/master)
-->

Swizzles are Julia operators that allow the user to fuse reduction and transposition operations into broadcast expressions. Swizzles are lazily evaluated, creating a language of Julia objects to represent tensor contractions and related operations. Swizzles were created as a good-faith attempt to implement tensor algebra using the abstractions and programming patterns of base Julia (broadcasting in particular). We hope that the results of our exploration may help inform future implementation decisions and redesigns of the Julia standard library. This project is no longer actively developed; Peter is partitioning graphs now. Swizzles.jl has been tested on Julia 1.5.

## What Is?

`Swizzle(op, mask...)(init, arg)` creates an operator which initializes `dst` using `init`, then reduces `arg` into `dst` such that dimension `d` of `dst` corresponds to dimension `mask[d]` of `arg`. This operator is represented lazily using the array type `SwizzledArray`. Thus,

```julia-repl
julia> using Swizzles

julia> A = [1 2 3; 4 5 6; 7 8 9]
3×3 Array{Int64,2}:
 1  2  3
 4  5  6
 7  8  9

julia> Swizzle(+, 2).(A)
3-element Array{Int64,1}:
 12
 15
 18

julia> Swizzle(+).(A)
45
```

We can use an instance of the singleton type `Nil` to insert a size 1 dimension into `dst`.

```julia-repl
julia> Swizzle(+, nil, 2).(A)
1×3 Array{Int64,2}:
 12  15  18
```

It is convenient to represent transpositions (`Swizzle`s which do not initialize or reduce) using the operator `Beam(imask...)`. `Beam(imask...)(arg)` produces an output array `dst` such that dimension `imask[d]` of `dst` corresponds to dimension `d` of `arg`.

```julia-repl
julia> B = [12; 15; 18]
3-element Array{Int64,1}:
 12
 15
 18

julia> Beam(2).(B)
1×3 Array{Int64,2}:
 12  15  18
```

Notice the similarity between index notation

```
A[i,j] = ∑ B[i,k,l] * D[l,j] * C[k,j]
```

and the Swizzles.jl representation

```
A .= Swizzle(+, 2, 1).(Beam(+, 2, 3, 4).(B) .* Beam(4, 1).(D) .* Beam(3, 1).(C))
```

## Why?

Swizzles.jl was created to provide a trait-based dispatch mechanism for tensor
kernel selection and array implementation. Swizzles uses `BroadcastStyles` and
`eachindex` to help select implementations, and provides an alternative abstract
array supertype, `StylishArray`, for new array types to target. The language
of `Swizzle` and `Broadcast` provides a high-level intermediate representation
for tensor operations.

## More Examples!

```julia-repl
julia> using Swizzles, LinearAlgebra

julia> x = rand(7); y = rand(7);

julia> Swizzle(+).(x .* y) ≈ dot(x, y)
true

julia> Swizzle(+).(abs.(x)) ≈ norm(x, 1)
true

julia> A = rand(5, 7); B = rand(7, 8);

julia> Swizzle(+, nil, 2).(A) ≈ sum(A, dims=1)
true

julia> Swizzle(+, 2).(A) ≈ sum(A, dims=1)[1,:]
true

julia> Beam(2, 1).(A) ≈ transpose(A)
true

julia> Beam(1, 4).(A) ≈ reshape(A, 5, 1, 1, 7)
true

julia> Swizzle(+, 1, 3).(A.*Beam(2, 3).(B)) ≈ A * B
true
```
