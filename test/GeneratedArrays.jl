using Swizzles.GeneratedArrays
using Swizzles.GeneratedArrays: square, root, power
using LinearAlgebra

@testset "GeneratedArrays" begin
    @testset "Squares" begin
        a = sqrt(prevfloat(typemax(Float64)))
        x = square(a)
        @test root(x) ≈ a
        @test root(x + x + x + x) ≈ 2 * a
        @test root(square(0) + x) ≈ a
        @test root(x + square(0)) ≈ a
        @test root(square(0) + square(0)) ≈ 0
        y = square(Inf)
        z = square(-Inf)
        @test root(y) == Inf
        @test root(y + z) == Inf
    end

    @testset "Powers" begin
        a = sqrt(prevfloat(typemax(Float64)))
        x = power(a, 2)
        @test root(x) ≈ a
        @test root(x + x + x + x) ≈ 2 * a
        @test root(power(0, 2) + x) ≈ a
        @test root(x + power(0, 2)) ≈ a
        @test root(power(0, 2) + power(0, 2)) ≈ 0
        y = power(Inf, 2)
        z = power(-Inf, 2)
        @test root(y) == Inf
        @test root(y + z) == Inf
    end

    @testset "Base" begin
        struct Test1{T, N, Arr<:AbstractArray{T, N}} <: GeneratedArray{T, N}
            parent::Arr
        end
        Base.size(arr::Test1) = size(arr.parent)
        Base.axes(arr::Test1) = axes(arr.parent)
        Base.getindex(arr::Test1, i...) = arr.parent[i...]
        Base.getindex(arr::Test1, i::Int) = arr.parent[i]
        Base.getindex(arr::Test1, i::CartesianIndex) = arr.parent[i]
        Base.setindex!(arr::Test1, x, i...) = arr.parent[i...] = x
        Base.setindex!(arr::Test1, x, i::Int) = arr.parent[i] = x
        Base.setindex!(arr::Test1, x, i::CartesianIndex) = arr.parent[i] = x


        A = [1 2 3; 4 5 6; 7 8 9]
        @test Test1(A)[1] == A[1]
        @test Test1(A)[1:3] == A[1:3]
        @test Test1(A)[2:3, 1:2] == A[2:3, 1:2]
        @test Test1(A)[2, 1:3] == A[2, 1:3]

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 2 3; 4 5 6; 7 8 9]
        Test1(A)[1] = -1
        B[1] = -1
        @test A == B

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 2 3; 4 5 6; 7 8 9]
        Test1(A)[1:3] = [-1 -1 -1]
        B[1:3] = [-1 -1 -1]
        @test A == B

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 2 3; 4 5 6; 7 8 9]
        Test1(A)[2:3, 1:2] = [-1 -1; -1 -1]
        B[2:3, 1:2] = [-1 -1; -1 -1]
        @test A == B

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 2 3; 4 5 6; 7 8 9]
        Test1(A)[2, 1:3] = [-1 -1 -1]
        B[2, 1:3] = [-1 -1 -1]
        @test A == B

        A = [1 2 3; 4 5 6; 7 8 9]
        @test map(x -> x == 2, Test1(A)) == map(x -> x == 2, A)

        A = [1 2 3; 4 5 6; 7 8 9]
        @test map((a, b) -> a + b == 4, Test1(A), Test1(A)) == map((a, b) -> a + b == 4, A, A)

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 2 3; 4 5 6; 7 8 9]
        C = [1 2 3; 4 5 6; 7 8 9]
        @test foreach(i -> B[i] = -i, Test1(A)) == foreach(i -> C[i] = -i, A)
        @test B == C

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 2 3; 4 5 6; 7 8 9]
        C = [1 2 3; 4 5 6; 7 8 9]
        @test foreach((a, b) -> B[a] = b, Test1(A), -A) == foreach((a, b) -> C[a] = b, A, -A)
        @test B == C

        A = [1 2 3; 4 5 6; 7 8 9]
        B = [1 2 3; 4 5 6; 7 8 9]
        C = [1 2 3; 4 5 6; 7 8 9]
        @test foreach((a, b) -> B[a] = b, Test1(A), -A) == foreach((a, b) -> C[a] = b, A, -A)
        @test B == C

        A = [1 2 3; 4 5 6; 7 8 9]

        @test reduce(+, Test1(A)) == reduce(+, A)
        @test reduce(+, Test1(A), dims=2) == reduce(+, A, dims=2)
        @test reduce(+, Test1(A), dims=(1, 2)) == reduce(+, A, dims=(1, 2))
        @test reduce(+, Test1(A), dims=:) == reduce(+, A, dims=:)
        @test reduce(+, Test1(A), dims=()) == reduce(+, A, dims=())
        @test reduce(+, Test1(A), init=10) == reduce(+, A, init=10)

        @test sum(Test1(A)) == sum(A)
        @test sum(Test1(A), dims=2) == sum(A, dims=2)
        @test sum(Test1(A), dims=(1, 2)) == sum(A, dims=(1, 2))
        @test sum(Test1(A), dims=:) == sum(A, dims=:)
        @test sum(Test1(A), dims=()) == sum(A, dims=())
        @test sum(Test1(A), init=10) == reduce(+, A, init=10)
    end

    @testset "LinearAlgebra" begin
        for (X, Y) in ((rand(3, 4), rand(3,4)), (rand(3), rand(3)))
            @test dot(Test1(X), Test1(Y)) ≈ dot(X, Y)
        end

        big = prevfloat(typemax(Float64))/16
        small = nextfloat(zero(Float64))

        for ps in ((-Inf,), (-3.0,), (-1,), (-0.5,), (0,), (0.5,), (1,), (2,), (3.0,), (Inf), ())
            for X in ([1, 2], [big, big], [small, small], rand(3, 3), rand(3))
                @test norm(Test1(X), ps...) ≈ norm(X, ps...)
            end
        end
    end
end
