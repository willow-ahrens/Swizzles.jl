using Base.Broadcast: broadcastable, Broadcasted
using Swizzles.ExtrudedArrays
using Swizzles.ArrayifiedArrays

foo(x) = x

@testset "ExtrudedArrays" begin
    for arg in ((1, 2, 3.0), (1, 2, 3), (), [1, 2, 3.0], [1, 2, 3], [], [1 2; 3 4], transpose([1, 2]))

        x = foo.(arg)
        y = foo.(ExtrudedArray(arg))
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
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true, true)
    @test keeps(bc) == (Keep(), true)
    #@test_throws MethodError inferkeeps(typeof(bc))

    bc  = Broadcasted(+, ((1, 2, 3), [1]'))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray([1]')))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true, false)
    @test keeps(bc) == (Keep(), false)
    #@test_throws MethodError inferkeeps(typeof(bc))

    bc  = Broadcasted(+, ((1, 2, 3), [1, 2, 3]))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray([1, 2, 3])))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true,)
    @test keeps(bc) == (Keep(),)
    #@test_throws MethodError inferkeeps(typeof(bc))

    bc  = Broadcasted(+, ((1,), [1, 2, 3]))
    bc′ = Broadcasted(+, ((1,), ExtrudedArray([1, 2, 3])))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true,)
    @test keeps(bc) == (true,)
    #@test_throws MethodError inferkeeps(typeof(bc))

    bc  = Broadcasted(+, (1, [1, 2, 3]))
    bc′ = Broadcasted(+, (1, ExtrudedArray([1, 2, 3])))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true,)
    @test keeps(bc) == (true,)
    #@test_throws MethodError inferkeeps(typeof(bc))

    bc  = Broadcasted(+, ((1, 2, 3), [1]))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray([1])))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true,)
    @test keeps(bc) == (Keep(),)
    #@test_throws MethodError inferkeeps(typeof(bc))

    bc  = Broadcasted(+, ((1, 2, 3), 1))
    bc′ = Broadcasted(+, ((1, 2, 3), 1))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true,)
    @test keeps(bc) == (Keep(),)
    #@test inferkeeps(typeof(bc)) == (true,)

    bc  = Broadcasted(+, ((1,), 1))
    bc′ = Broadcasted(+, ((1,), 1))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (false,)
    @test keeps(bc) == (Extrude(),)
    #@test inferkeeps(typeof(bc)) == (false,)

    bc  = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1, 2, 3), [1, 2, 3]'))))
    bc′ = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1, 2, 3), ExtrudedArray([1, 2, 3]')))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true, true)
    @test keeps(bc) == (true, true)
    #@test_throws MethodError inferkeeps(typeof(bc))

    bc  = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1,), [1, 2, 3]'))))
    bc′ = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1,), ExtrudedArray([1, 2, 3]')))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    #@test inferkeeps(typeof(lift_keeps(bc))) == (true, false)
    @test keeps(bc) == (true, false)
    #@test_throws MethodError inferkeeps(typeof(bc))
end
