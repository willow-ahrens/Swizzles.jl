using Base.Broadcast: broadcastable, Broadcasted
using Swizzles.ExtrudedArrays
using Swizzles.ArrayifiedArrays

foo(x) = x

@testset "ExtrudedArrays" begin
    for arg in ([1, 2, 3.0], [1, 2, 3], [], [1 2; 3 4], transpose([1, 2]))

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

    bc  = Broadcasted(+, ((1, 2, 3), identity.([1, 2, 3]')))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray(identity.([1, 2, 3]'), (extrude, keep))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (keep, true)

    bc  = Broadcasted(+, ((1, 2, 3), identity.([1]')))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray(identity.([1]'), (extrude, extrude))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (keep, false)

    bc  = Broadcasted(+, ((1, 2, 3), [1, 2, 3]))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray([1, 2, 3], (keep,))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (keep,)

    bc  = Broadcasted(+, ((1,), [1, 2, 3]))
    bc′ = Broadcasted(+, ((1,), ExtrudedArray([1, 2, 3], (keep,))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (true,)

    bc  = Broadcasted(+, (1, [1, 2, 3]))
    bc′ = Broadcasted(+, (1, ExtrudedArray([1, 2, 3], (keep,))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (true,)

    bc  = Broadcasted(+, ((1, 2, 3), [1]))
    bc′ = Broadcasted(+, ((1, 2, 3), ExtrudedArray([1], (extrude,))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (keep,)

    bc  = Broadcasted(+, ((1, 2, 3), 1))
    bc′ = Broadcasted(+, ((1, 2, 3), 1))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (keep,)

    bc  = Broadcasted(+, ((1,), 1))
    bc′ = Broadcasted(+, ((1,), 1))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (extrude,)

    bc  = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1, 2, 3), identity.([1, 2, 3]')))))
    bc′ = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1, 2, 3), ExtrudedArray(identity.([1, 2, 3]'), (extrude, keep))))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (keep, true)

    bc  = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1,), identity.([1, 2, 3]')))))
    bc′ = Delay().(Swizzle(+, (2, 1)).(Broadcasted(+, ((1,), ExtrudedArray(identity.([1, 2, 3]'), (extrude, keep))))))
    @test typeof(lift_keeps(bc)) == typeof(bc′)
    @test keeps(bc) == (true, false)

    a  = [1, 2, 3]'
    a′ = ExtrudedArray([1, 2, 3], (keep,))'
    @test typeof(lift_keeps(a)) == typeof(a′)
    @test keeps(a) == (extrude, true)
end
