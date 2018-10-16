using BenchmarkTools
using Swizzles

suite["matmatmul"] = BenchmarkGroup()

for (N, M, K) in zip(2 .^ (2:2:4), 2 .^ (2:2:4), 2 .^ (2:2:4))
    mymatmul = suite["matmatmul"][(N=N, M=M, K=K)] = BenchmarkGroup()
    A = rand(N, K)
    B = rand(K, M)
    mymatmul["naive"] = @benchmarkable begin
        A = $A
        B = $B
        C = zeros(size(A, 1), size(B, 2))
        for j = axes(B, 1)
            for k = axes(B, 1)
                for i = axes(A, 1)
                    C[i, j] += A[i, k] * B[k, j]
                end
            end
        end
        return C
    end
    mymatmul["default"] = @benchmarkable $A * $B
    mymatmul["swizzle"] = @benchmarkable Sum(2).($A .* Beam(2, 3).($B))
end
