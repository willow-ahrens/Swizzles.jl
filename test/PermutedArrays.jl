using Swizzles
using Swizzles.PermutedArrays

@testset "PermutedArrays" begin
    A = reshape(1:9, 3, 3)
    C = copy(A)
    B = Permute(map(reverse, axes(A))...).(C)
    @test B == reshape(9:-1:1, 3, 3)
    B = Permute(map(reverse, axes(A))...)(C)
    @test B == reshape(9:-1:1, 3, 3)
    B .= reshape(1:9, 3, 3)
    @test C == reshape(9:-1:1, 3, 3)

    A = reshape(1:9, 3, 3)
    B = Permute(reverse(1:3))(reshape(1:9, 3, 3))
    @test_throws PermutationMismatch A * B
    A = Permute(1:3, reverse(1:3))(reshape(1:9, 3, 3))
    @test A * B == reshape(1:9, 3, 3) * reshape(1:9, 3, 3)
    @test Sum().(A) == sum(1:9)
end
