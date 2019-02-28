using Swizzles.RecognizeStyles: reprexpr
using LinearAlgebra

@testset "RecognizeStyles" begin

    @generated function test_eval_reprexpr_is_id(val)
        return quote
            oVal = eval(reprexpr(val, $val))

            @test typeof(val) === typeof(oVal)
            @test val == oVal
        end
    end

    test_eval_reprexpr_is_id(42)
    test_eval_reprexpr_is_id(6. * 9.)

    A = [1 2 3; 4 5 6]
    test_eval_reprexpr_is_id(A)
    test_eval_reprexpr_is_id(A')
    test_eval_reprexpr_is_id(transpose(A))
    test_eval_reprexpr_is_id(transpose(A'))
    test_eval_reprexpr_is_id(transpose(A)')

end
