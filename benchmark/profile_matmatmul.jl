using Profile
using Cthulhu
using Swizzles
using InteractiveUtils

f(A, B) = Sum(2).(A .* Beam(2, 3).(B))

A = B = rand(64,64)

f(A, B)
@profile begin
    for i = 1:1000
        f(A, B)
    end
end

Profile.print()
println()
display(@code_typed(f(A, B)))
println()
