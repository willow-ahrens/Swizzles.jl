using SplayedDimsArrays
using Test

A = rand(5)
B = rand(4)
C = rand(1, 5)
D = rand(1, 4)
E = rand(1, 4, 5)
F = rand(5, 4, 5)

println(cast(A, 3).*cast(B,1))
