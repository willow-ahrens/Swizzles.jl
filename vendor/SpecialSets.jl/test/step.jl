@testset "Step" begin
    @test Step(2, 0) == Step(2, 0)
    @test Step(2) == Step(2, 0)
    @test Step(2, 1) ≠ Step(2, 0)
    @test eltype(Step(2)) == Int
    @test_throws DivideError Step(0)

    @testset "Step($m, $a)" for m ∈ 1:10, a ∈ 0:m-1
        for x ∈ -2m:2m
            expected = mod(x, m) == a
            @test (x ∈ Step(m, a)) == expected
        end
    end

    @test Step(2, 0) ∩ Step(2, 0) == Step(2, 0)
    @test Step(2, 0) ∩ Step(2, 1) == ∅
    @test Even ∩ Odd == ∅
    @test Step(2) ∩ Step(3) == Step(6)
    @test Step(2) ∩ Step(4) == Step(4)
    @test Step(2) ∩ Step(4, 1) == ∅
    @test Step(2) ∩ Step(3, 1) ∩ Step(7, 3) == Step(42, 10)
    @test Step(3, 1) ∩ Step(5, 4) == Step(15, 4)
end
