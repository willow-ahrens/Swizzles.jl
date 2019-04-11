using Swizzles.StylishArrays
using Swizzles.StylishArrays: square, root, power
using LinearAlgebra

@testset "StylishArrays" begin
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

    struct TestStylishArray{T, N, Arr<:AbstractArray{T, N}} <: StylishArray{T, N}
        parent::Arr
    end
    Base.size(arr::TestStylishArray) = size(arr.parent)
    Base.axes(arr::TestStylishArray) = axes(arr.parent)
    Base.getindex(arr::TestStylishArray, i...) = arr.parent[i...]
    Base.getindex(arr::TestStylishArray, i::Int) = arr.parent[i]
    Base.getindex(arr::TestStylishArray, i::CartesianIndex) = arr.parent[i]
    Base.setindex!(arr::TestStylishArray, x, i...) = arr.parent[i...] = x
    Base.setindex!(arr::TestStylishArray, x, i::Int) = arr.parent[i] = x
    Base.setindex!(arr::TestStylishArray, x, i::CartesianIndex) = arr.parent[i] = x

    @testset "Base" begin

        @testset "copyto!" begin
            for A in ([1 2 3; 4 5 6; 7 8 9], TestStylishArray([1 2 3; 4 5 6; 7 8 9]))
                B = TestStylishArray(zeros(3,3))
                @test A == copyto!(B, A)
                @test A == B
                B = TestStylishArray(zeros(9))
                #@test reshape(A, :) == copyto!(B, A) #FIXME
                #@test reshape(A, :) == B #FIXME
            end

            A = TestStylishArray([1 2 3; 4 5 6; 7 8 9])
            B = zeros(3,3)
            @test A == copyto!(B, A)
            @test A == B
            B = zeros(9)
            #@test reshape(A, :) == copyto!(B, A) #FIXME
            #@test reshape(A, :) == B #FIXME
        end

        @testset "getindex and setindex!" begin
            A = [1 2 3; 4 5 6; 7 8 9]
            @test TestStylishArray(A)[1] == A[1]
            @test TestStylishArray(A)[1:3] == A[1:3]
            @test TestStylishArray(A)[2:3, 1:2] == A[2:3, 1:2]
            @test TestStylishArray(A)[2, 1:3] == A[2, 1:3]

            A = [1 2 3; 4 5 6; 7 8 9]
            B = [1 2 3; 4 5 6; 7 8 9]
            TestStylishArray(A)[1] = -1
            B[1] = -1
            @test A == B

            A = [1 2 3; 4 5 6; 7 8 9]
            B = [1 2 3; 4 5 6; 7 8 9]
            TestStylishArray(A)[1:3] = [-1 -1 -1]
            B[1:3] = [-1 -1 -1]
            @test A == B

            A = [1 2 3; 4 5 6; 7 8 9]
            B = [1 2 3; 4 5 6; 7 8 9]
            TestStylishArray(A)[2:3, 1:2] = [-1 -1; -1 -1]
            B[2:3, 1:2] = [-1 -1; -1 -1]
            @test A == B

            A = [1 2 3; 4 5 6; 7 8 9]
            B = [1 2 3; 4 5 6; 7 8 9]
            TestStylishArray(A)[2, 1:3] = [-1 -1 -1]
            B[2, 1:3] = [-1 -1 -1]
            @test A == B
        end

        @testset "map" begin
            A = [1 2 3; 4 5 6; 7 8 9]
            @test map(x -> x == 2, TestStylishArray(A)) == map(x -> x == 2, A)

            A = [1 2 3; 4 5 6; 7 8 9]
            @test map((a, b) -> a + b == 4, TestStylishArray(A), TestStylishArray(A)) == map((a, b) -> a + b == 4, A, A)
        end

        @testset "foreach" begin
            A = [1 2 3; 4 5 6; 7 8 9]
            B = [1 2 3; 4 5 6; 7 8 9]
            C = [1 2 3; 4 5 6; 7 8 9]
            @test foreach(i -> B[i] = -i, TestStylishArray(A)) == foreach(i -> C[i] = -i, A)
            @test B == C

            A = [1 2 3; 4 5 6; 7 8 9]
            B = [1 2 3; 4 5 6; 7 8 9]
            C = [1 2 3; 4 5 6; 7 8 9]
            @test foreach((a, b) -> B[a] = b, TestStylishArray(A), -A) == foreach((a, b) -> C[a] = b, A, -A)
            @test B == C

            A = [1 2 3; 4 5 6; 7 8 9]
            B = [1 2 3; 4 5 6; 7 8 9]
            C = [1 2 3; 4 5 6; 7 8 9]
            @test foreach((a, b) -> B[a] = b, TestStylishArray(A), -A) == foreach((a, b) -> C[a] = b, A, -A)
            @test B == C
        end

        @testset "reductions" begin
            A = [1 2 3; 4 5 6; 7 8 9]

            @test mapreduce(-, +, TestStylishArray(A)) == mapreduce(-, +, A)
            @test mapreduce(-, +, TestStylishArray(A), dims=2) == mapreduce(-, +, A, dims=2)
            @test mapreduce(-, +, TestStylishArray(A), dims=(1, 2)) == mapreduce(-, +, A, dims=(1, 2))
            @test mapreduce(-, +, TestStylishArray(A), dims=:) == mapreduce(-, +, A, dims=:)
            @test mapreduce(-, +, TestStylishArray(A), dims=()) == mapreduce(-, +, A, dims=())
            @test mapreduce(-, +, TestStylishArray(A), init=10) == mapreduce(-, +, A, init=10)

            @test reduce(+, TestStylishArray(A)) == reduce(+, A)
            @test reduce(+, TestStylishArray(A), dims=2) == reduce(+, A, dims=2)
            @test reduce(+, TestStylishArray(A), dims=(1, 2)) == reduce(+, A, dims=(1, 2))
            @test reduce(+, TestStylishArray(A), dims=:) == reduce(+, A, dims=:)
            @test reduce(+, TestStylishArray(A), dims=()) == reduce(+, A, dims=())
            @test reduce(+, TestStylishArray(A), init=10) == reduce(+, A, init=10)

            @test sum(TestStylishArray(A)) == sum(A)
            @test sum(TestStylishArray(A), dims=2) == sum(A, dims=2)
            @test sum(TestStylishArray(A), dims=(1, 2)) == sum(A, dims=(1, 2))
            @test sum(TestStylishArray(A), dims=:) == sum(A, dims=:)
            @test sum(TestStylishArray(A), dims=()) == sum(A, dims=())
            @test sum(TestStylishArray(A), init=10) == reduce(+, A, init=10)
        end
    end

    @testset "LinearAlgebra" begin
        @testset "norm" begin
            for (X, Y) in ((rand(3, 4), rand(3,4)), (rand(3), rand(3)))
                @test dot(TestStylishArray(X), TestStylishArray(Y)) ≈ dot(X, Y)
            end

            big = prevfloat(typemax(Float64))/16
            small = nextfloat(zero(Float64))

            for ps in ((-Inf,), (-3.0,), (-1,), (-0.5,), (0,), (0.5,), (1,), (2,), (3.0,), (Inf), ())
                for X in ([1, 2], [big, big], [small, small], rand(3, 3), rand(3))
                    @test norm(TestStylishArray(X), ps...) ≈ norm(X, ps...)
                end
            end
        end

        @testset "mul!" begin
            A = rand(3,3)
            u = rand(3,1)
            v = rand(1,3)
            w = rand(3)

            for (a, b) in Iterators.product((A, u, v, w), (A, u, v, w))
                valid = false
                y = nothing
                try
                    y = a * b
                    valid = true
                catch MethodError
                end
                if valid
                    @test TestStylishArray(a) * b ≈ a * b
                    @test a * TestStylishArray(b) ≈ a * b
                    @test TestStylishArray(a) * TestStylishArray(b) ≈ a * b

                    @test mul!(identity.(y), TestStylishArray(a), b) ≈ mul!(identity.(y), a, b)
                    @test mul!(identity.(y), a, TestStylishArray(b)) ≈ mul!(identity.(y), a, b)
                    @test mul!(identity.(y), TestStylishArray(a), TestStylishArray(b)) ≈ mul!(identity.(y), a, b)
                    @test mul!(TestStylishArray(identity.(y)), a, b) ≈ mul!(identity.(y), a, b)
                    @test mul!(TestStylishArray(identity.(y)), TestStylishArray(a), b) ≈ mul!(identity.(y), a, b)
                    @test mul!(TestStylishArray(identity.(y)), a, TestStylishArray(b)) ≈ mul!(identity.(y), a, b)
                    @test mul!(TestStylishArray(identity.(y)), TestStylishArray(a), TestStylishArray(b)) ≈ mul!(identity.(y), a, b)
                end
            end
        end

        @testset "transpose and adjoint" begin
            A = rand(Complex{Float64}, 3,3)
            u = rand(Complex{Float64}, 3,1)
            v = rand(Complex{Float64}, 1,3)
            w = rand(Complex{Float64}, 3)

            for (a) in (A, u, v, w)
                @test transpose(TestStylishArray(a)) == transpose(a)
                @test transpose(TestStylishArray(a)) isa Swizzles.SwizzledArray
                @test adjoint(TestStylishArray(a)) == adjoint(a)
                @test adjoint(TestStylishArray(a)) isa Swizzles.SwizzledArray
            end
        end

        @testset "permutedims" begin
            A = rand(3,3)
            u = rand(3,1)
            v = rand(1,3)
            w = rand(3)

            for a in (A, u, v, w)
                @test permutedims(TestStylishArray(a)) == permutedims(a)
                @test permutedims(TestStylishArray(a)) isa Swizzles.SwizzledArray
            end

            A = rand(3,3,3)

            for perm in [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
                @test permutedims(TestStylishArray(A), perm) == permutedims(A, perm)
                @test permutedims(TestStylishArray(A), perm) isa Swizzles.SwizzledArray
            end
        end
    end
end
