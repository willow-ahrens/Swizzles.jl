using BenchmarkTools
using Swizzles
using Base.Iterators

suite["transpose"] = BenchmarkGroup()

for (N, M) in product(2 .^ (2:2:6), 2 .^ (2:2:6))
    mytranspose = suite["transpose"][(N=N, M=M)] = BenchmarkGroup()
    A = rand(N, M)
    B = rand(M, N)
    mytranspose["naive"] = @benchmarkable begin
        A = $A
        B = $B
        for j = axes(A, 1)
            for i = axes(A, 2)
                B[i, j] += A[j, i]
            end
        end
        return B
    end
    mytranspose["default"] = @benchmarkable copyto!($B, permutedims($A))
    mytranspose["swizzle"] = @benchmarkable $B.=Beam(2, 1).($A)
end
