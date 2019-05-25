# SpecialSets.jl

[![Travis Build Status](https://travis-ci.com/HarrisonGrodin/SpecialSets.jl.svg?branch=master)](https://travis-ci.com/HarrisonGrodin/SpecialSets.jl)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/lh7y4fktg41s2l6t?svg=true)](https://ci.appveyor.com/project/HarrisonGrodin/specialsets-jl)
[![Coverage Status](https://coveralls.io/repos/github/HarrisonGrodin/SpecialSets.jl/badge.svg?branch=master)](https://coveralls.io/github/HarrisonGrodin/SpecialSets.jl?branch=master)

**SpecialSets** provides implementations of sets commonly used in mathematics, as well as the logic for cleanly combining such sets.

## Examples
```julia
julia> LessThan(3)
{x ∈ ℤ | x < 3}

julia> LessThan(3, true)
{x ∈ ℤ | x ≤ 3}

julia> LessThan(3, true) ∩ GreaterThan(3, true)
Set([3])

julia> LessThan(3, true) ∩ TypeSet(String)
Set(Any[])

julia> GreaterThan(5, true) ∩ NotEqual(12)
{x ∈ ℤ | x ≠ 12, x ≥ 5}

julia> GreaterThan(5, true) ∩ NotEqual(3)
{x ∈ ℤ | x ≥ 5}

julia> Even ∩ Step(3)
{x ∈ ℤ | (x ≡ 0 (mod 6))}

julia> Even ∩ Step(3, 1)
{x ∈ ℤ | (x ≡ 4 (mod 6))}

julia> Even ∩ Odd
Set(Any[])

julia> Even ∩ LessThan(0)
{x ∈ ℤ | x < 0, (x ≡ 0 (mod 2))}

julia> Even ∩ Set(1:5) ∩ NotEqual(4)
Set([2])
```
