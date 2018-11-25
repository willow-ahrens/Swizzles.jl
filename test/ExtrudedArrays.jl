using Base.Broadcast: broadcastable, Broadcasted
using Swizzle.ExtrudedArrays

myidentity(x) = x

@testset "ExtrudedArrays" begin
    for arg in ((1, 2, 3.0), (1, 2, 3), (), [1, 2, 3.0], [1, 2, 3], [], [1 2; 3 4], transpose([1, 2]))

        x = myidentity.(arg)
        y = myidentity.(ExtrudedArray(arg))
        @test x == y
        @test typeof(x) == typeof(y)

        x = arg .+ arg
        y = ExtrudedArray(arg) .+ arg
        @test x == y
        @test typeof(x) == typeof(y)

        x = arg .+ arg
        y = ExtrudedArray(arg) .+ ExtrudedArray(arg)
        @test x == y
        @test typeof(x) == typeof(y)
    end

    bc  = Broadcasted(+, ((1, 2, 3), [1, 2, 3]'))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray([1, 2, 3]')))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(typeof(lift_keeps(bc))) == (true, true)
    @test keeps(bc) == (true, true)
    @test_throws MethodError keeps(typeof(bc))
end
