using Swizzles
using Swizzles.EachindexArrays

@testset "EachindexArrays" begin
    I = CartesianIndices((1:5, 1:5))
    A = copy(I)
    tape = []
    casette(_, i) = push!(tape, i)
    Swizzle(casette).(nothing, I)
    #println(Focus(2, 1).(tape))
    tape = []
    Swizzle(casette).(nothing, Tile(2,2).(I))
    println(Focus(2, 1).(tape))
end
