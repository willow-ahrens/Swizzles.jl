# Swizzles

[![Travis](https://travis-ci.org/peterahrens/Swizzles.jl.svg?branch=master)](https://travis-ci.org/peterahrens/Swizzles.jl)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/32r7s2skrgm9ubva/branch/master?svg=true)](https://ci.appveyor.com/project/peterahrens/swizzles-jl/branch/master)
[![Coveralls](https://coveralls.io/repos/peterahrens/Swizzles.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/peterahrens/Swizzles.jl?branch=master)
[![Codecov](http://codecov.io/github/peterahrens/Swizzles.jl/coverage.svg?branch=master)](http://codecov.io/github/peterahrens/Swizzles.jl?branch=master)

Swizzles are Julia datatypes that allow the user to fuse reduction and transposition operations into broadcast expressions. Swizzle operations are lazily evaluated together with broadcast expressions, and provide a familar interface for reducing and permuting.

## Syntax

Let's start with a few examples!

```julia-repl
julia> using Swizzles, LinearAlgebra

julia> A = rand(5, 7); B = rand(7, 8);

julia> Sum(2).(A.*Beam(2, 3).(B)) ≈ A * B
true

julia> Beam(2, 1).(A) ≈ transpose(A)
true

julia> x = rand(7); y = rand(7);

julia> Sum().(x .* y) ≈ dot(x, y)
true

julia> sqrt.(Sum().(x.^2)) ≈ norm(x, 2)
true

julia> Sum().(abs.(x)) ≈ norm(x, 1)
true

```
