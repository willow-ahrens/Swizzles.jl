using Swizzles
using Test

@testset "Swizzles" begin
    include("Swizzles.jl")
    include("ShallowArrays.jl")
    include("ExtrudedArrays.jl")
    include("ArrayifiedArrays.jl")
    include("RecognizeStyles.jl")
    include("util.jl")
end
