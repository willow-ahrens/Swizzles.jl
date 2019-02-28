using Swizzles.RecognizeStyles: reprexpr
using LinearAlgebra
using Base.Broadcast: broadcasted 


@testset "reprexpr" begin

    @generated function test_eval_reprexpr_is_id(val)
        return quote
            expr = reprexpr(val, $val)
            oVal = eval(expr)

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

    B = [300 200 100]
    C = [1000]
    D = 10000
    test_eval_reprexpr_is_id(broadcasted(+, A, B, C, D))
    test_eval_reprexpr_is_id(broadcasted(*, A, B, C, D))
    test_eval_reprexpr_is_id(
        broadcasted(+,
            broadcasted(+, A, B),
            broadcasted(-, B, C),
            broadcasted(*, C, D),
            broadcasted(/, D, A),
            broadcasted(^, A, B),
        )
    )
end
