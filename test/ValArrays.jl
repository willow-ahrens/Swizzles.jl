using Base.Broadcast: broadcastable, Broadcasted
using Swizzles.ValArrays
using Swizzles.ArrayifiedArrays
using Swizzles

myidentity(x) = x

@testset "ValArrays" begin
    A0 = ValArray(0)
    A1 = ValArray(1)

    @test A1.+A1 == 2

    @test A1.*A1 == 1

    B = rand(3,3)
    B .*= A0
    @test B == zeros(3,3)

    B = rand(3,3)
    B .= A0
    @test B == zeros(3,3)

    @test lift_vals(1) == A1
    @test typeof(lift_vals(1)) == typeof(A1)

    B = reshape(1:27, 3, 3, 3)
    C = reshape(1:9, 3, 3)

    @test Sum(2, 3).(B .* (A1 .+ C .+ A0) .+ A1) == Sum(2, 3).(B .* (1 .+ C .+ 0) .+ 1)
    @test typeof(Sum(2, 3).(B .* (A1 .+ C .+ A0))) == typeof(Sum(2, 3).(B .* (1 .+ C .+ 0)))

    @test typeof(Delay().(Sum(2, 3).(B .* (A1 .+ C .+ A0) .+ A1))) ==
        typeof(lift_vals(Delay().(Sum(2, 3).(B .* (1 .+ C .+ 0) .+ 1))))

end
