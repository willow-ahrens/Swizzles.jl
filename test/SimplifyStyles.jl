using Swizzles
using Swizzles.SimplifyStyles
using Swizzles.ValArrays
using Swizzles.SimplifyStyles: reprexpr
using LinearAlgebra
using Base.Broadcast: broadcasted


@testset "reprexpr" begin

    # Checks faithfulness, i.e. (reprexpr |> eval) == eval
    @generated function test_eval_reprexpr_is_id(val)
        expr = reprexpr(:val, val)
        return quote
            oVal = $expr
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

    # Check that Antennae are being used.
    bd = broadcasted(+, A, B)
    @test reprexpr(:bd, typeof(bd)).args[1] isa Swizzles.Antennae.Antenna

    test_eval_reprexpr_is_id(Swizzle(+, 1)(A))
    test_eval_reprexpr_is_id(Swizzle(*, 1)(A) |> lift_vals)
    @test reprexpr(
            :sw,
            Swizzle(*, 1)(A) |> lift_vals |> typeof
          ).args[2] == ValArray(1)
end

@testset "Simplify" begin
    A, B = [1 2 3; 4 5 6], [100 200 300]
    @test Simplify().(A) == A
    @test Simplify().(A .+ B) == A .+ B
end
