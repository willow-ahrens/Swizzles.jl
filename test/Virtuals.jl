using Swizzles.ExtrudedArrays
using Swizzles.NamedArrays
using Swizzles.ValArrays
using Swizzles.ArrayifiedArrays
using Swizzles.Virtuals

using Base.Broadcast: broadcastable, Broadcasted, materialize

using Swizzles

@testset "Virtuals" begin
    for x in [@_((1, 2, 3) .+ [1, 2, 3]'),
              @_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9]),
              Swizzle(+)(@_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9])),
              Swizzle(+, 1)(@_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9]))]
        @test kept.(keeps(virtualize(:root, typeof(lift_keeps(x))))) == kept.(keeps(x))
        @test eltype(virtualize(:root, typeof(lift_keeps(x)))) == eltype(x)
    end
end
