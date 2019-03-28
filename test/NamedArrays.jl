using Base.Broadcast: broadcastable, Broadcasted
using Swizzles.NamedArrays
using Swizzles.ArrayifiedArrays
using Swizzles

foo(x) = x

@testset "NamedArrays" begin
    A = rand(3,3)
    B = rand(3,3)
    C = rand(3,3)

    bc = Delay().(A .+ B .+ C)
    @test typeof(lift_names(bc)) <: Broadcasted{<:Any, <:Any, typeof(+), <:Tuple{<:Broadcasted{<:Any, <:Any, typeof(+), <:Tuple{<:NamedArray{<:Any, <:Any, <:Any, :obj1}, <:NamedArray{<:Any, <:Any, <:Any, :obj2}}}, NamedArray{<:Any, <:Any, <:Any, :obj3}}}
    @test name(lift_names(bc).args[1].args[1]) == :obj1
    @test name(lift_names(bc).args[1].args[2]) == :obj2
    @test name(lift_names(bc).args[2]) == :obj3

    bc = Delay().(Swizzle(+).(A .+ B))
    @test typeof(lift_names(bc)) <: Swizzles.SwizzledArray{<:Any, <:Any, typeof(+), (), <:Any, <:ArrayifiedArray{<:Any, <:Any, <:Broadcasted{<:Any, <:Any, typeof(+), <:Tuple{<:NamedArray{<:Any, <:Any, <:Any, :obj1}, <:NamedArray{<:Any, <:Any, <:Any, :obj2}}}}}
end
