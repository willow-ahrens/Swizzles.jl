using Swizzles
using Swizzles.PermutedArrays

@testset "PermutedArrays" begin
    A = reshape(1:9, 3, 3)
    B = Permute(map(reverse, axes(A))...).(A)
    @test B == reshape(9:-1:1, 3, 3)
end
