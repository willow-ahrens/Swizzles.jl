using Swizzles.BroadcastedArrays

myidentity(x) = x

@testset "BroadcastedArrays" begin
    for arg in ((1, 2, 3.0), (1, 2, 3), (), [1, 2, 3.0], [1, 2, 3], [], [1 2; 3 4], transpose([1, 2]))
        @test myidentity.(BroadcastedArray(arg)) == myidentity.(arg)
    end
end
