using Base.Broadcast: broadcastable
using Swizzle.WrapperArrays

struct SimpleWrapperArray{T, N, P} <: WrapperArray{T, N, P}
    arg::P
end

function SimpleWrapperArray(arg)
    arg = broadcastable(arg)
    SimpleWrapperArray{eltype(arg), ndims(typeof(arg)), typeof(arg)}(arg)
end
SimpleWrapperArray(arg::Tuple) = SimpleWrapperArray{eltype(arg), 1, typeof(arg)}(arg)
Base.parent(arr::SimpleWrapperArray) = arr.arg
WrapperArrays.map_parent(f, arr::SimpleWrapperArray) = SimpleWrapperArray(f(parent(arr)))
Base.size(arr::SimpleWrapperArray{<:Any, <:Any, <:Tuple}) = length(arr.arg)
Base.axes(arr::SimpleWrapperArray{<:Any, <:Any, <:Tuple}) = (Base.OneTo(length(arr.arg)),)

myidentity(x) = x

@testset "WrapperArrays" begin
    for arg in ((1, 2, 3.0), (1, 2, 3), (), [1, 2, 3.0], [1, 2, 3], [], [1 2; 3 4], transpose([1, 2]))

        x = myidentity.(arg)
        y = myidentity.(SimpleWrapperArray(arg))
        @test x == y
        @test typeof(x) == typeof(y)

        x = arg .+ arg
        y = SimpleWrapperArray(arg) .+ arg
        @test x == y
        @test typeof(x) == typeof(y)

        x = arg .+ arg
        y = SimpleWrapperArray(arg) .+ SimpleWrapperArray(arg)
        @test x == y
        @test typeof(x) == typeof(y)
    end
end
