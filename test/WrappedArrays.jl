using Swizzle
using Base.Broadcast: broadcastable

struct SimpleWrappedArray{T, N, P} <: WrappedArray{T, N, P}
    arg::P
end

function SimpleWrappedArray(arg)
    arg = broadcastable(arg)
    SimpleWrappedArray{eltype(arg), ndims(typeof(arg)), typeof(arg)}(arg)
end
SimpleWrappedArray(arg::Tuple) = SimpleWrappedArray{eltype(arg), 1, typeof(arg)}(arg)
Base.parent(arr::SimpleWrappedArray) = arr.arg
Base.size(arr::SimpleWrappedArray{<:Any, <:Any, <:Tuple}) = length(arr.arg)
Base.axes(arr::SimpleWrappedArray{<:Any, <:Any, <:Tuple}) = (Base.OneTo(length(arr.arg)),)

myidentity(x) = x

@testset begin
    for arg in ((1, 2, 3.0), (1, 2, 3), (), [1, 2, 3.0], [1, 2, 3], [], [1 2; 3 4], transpose([1, 2]))

        x = myidentity.(arg)
        y = myidentity.(SimpleWrappedArray(arg))
        println(Unwrap().(myidentity.(SimpleWrappedArray(arg))))
        println(Base.Broadcast.instantiate(Unwrap().(myidentity.(SimpleWrappedArray(arg)))))
        @test x == y
        @test typeof(x) == typeof(y)

#=
        x = arg .+ arg
        y = SimpleWrappedArray(arg) .+ arg
        @test x == y
        @test typeof(x) == typeof(y)

        x = arg .+ arg
        y = SimpleWrappedArray(arg) .+ SimpleWrappedArray(arg)
        @test x == y
        @test typeof(x) == typeof(y)
=#
    end
end
