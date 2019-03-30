using Swizzles
using Swizzles.EachindexArrays

@testset "EachindexArrays" begin
    I = CartesianIndices((1:5, 1:5))
    tape = []
    casette(_, i) = push!(tape, i)
    Swizzle(casette).(nothing, Tile(2,2).(I))
    @test Beam(2).(tape) == [CartesianIndex(1, 1) CartesianIndex(2, 1) CartesianIndex(1, 2) CartesianIndex(2, 2) CartesianIndex(3, 1) CartesianIndex(4, 1) CartesianIndex(3, 2) CartesianIndex(4, 2) CartesianIndex(5, 1) CartesianIndex(5, 2) CartesianIndex(1, 3) CartesianIndex(2, 3) CartesianIndex(1, 4) CartesianIndex(2, 4) CartesianIndex(3, 3) CartesianIndex(4, 3) CartesianIndex(3, 4) CartesianIndex(4, 4) CartesianIndex(5, 3) CartesianIndex(5, 4) CartesianIndex(1, 5) CartesianIndex(2, 5) CartesianIndex(3, 5) CartesianIndex(4, 5) CartesianIndex(5, 5)]
end
