using Base.Broadcast: broadcastable
using Swizzles.ShallowArrays
using Swizzles.WrapperArrays

struct SimpleShallowArray{T, N, P} <: ShallowArray{T, N, P}
    arg::P
end

function SimpleShallowArray(arg)
    arg = broadcastable(arg)
    SimpleShallowArray{eltype(arg), ndims(typeof(arg)), typeof(arg)}(arg)
end
SimpleShallowArray(arg::Tuple) = SimpleShallowArray{eltype(arg), 1, typeof(arg)}(arg)
Base.parent(arr::SimpleShallowArray) = arr.arg
WrapperArrays.adopt(arg, arr::SimpleShallowArray) = SimpleShallowArray(arg)
Base.size(arr::SimpleShallowArray{<:Any, <:Any, <:Tuple}) = length(arr.arg)
Base.axes(arr::SimpleShallowArray{<:Any, <:Any, <:Tuple}) = (Base.OneTo(length(arr.arg)),)

myidentity(x) = x

@testset "ShallowArrays" begin
    for arg in ((1, 2, 3.0), (1, 2, 3), (), [1, 2, 3.0], [1, 2, 3], [], [1 2; 3 4], transpose([1, 2]))

        x = myidentity.(arg)
        y = myidentity.(SimpleShallowArray(arg))
        @test x == y
        @test typeof(x) == typeof(y)

        x = arg .+ arg
        y = SimpleShallowArray(arg) .+ arg
        @test x == y
        @test typeof(x) == typeof(y)

        x = arg .+ arg
        y = SimpleShallowArray(arg) .+ SimpleShallowArray(arg)
        @test x == y
        @test typeof(x) == typeof(y)
    end
end
