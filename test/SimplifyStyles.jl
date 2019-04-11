using Swizzles
using Swizzles.Properties
using Swizzles.SimplifyStyles
using Swizzles.NamedArrays
using Swizzles.ValArrays
using Swizzles.SimplifyStyles: rewriteable, evaluable
using LinearAlgebra
using Base.Broadcast: broadcasted

@testset "rewriteable" begin

    @testset "rewriteable faithfulness" begin
        # Checks faithfulness, i.e. (rewriteable |> evaluable |> eval) == eval
        @generated function test_evaluable_rewriteable_is_id(val)
            expr = evaluable(rewriteable(:val, val)...)
            return quote
                oVal = $expr
                @test typeof(val) === typeof(oVal)
                @test val == oVal
            end
        end

        test_evaluable_rewriteable_is_id(42)
        test_evaluable_rewriteable_is_id(6. * 9.)

        A = [1 2 3; 4 5 6]
        test_evaluable_rewriteable_is_id(A)
        test_evaluable_rewriteable_is_id(A')
        test_evaluable_rewriteable_is_id(transpose(A))
        test_evaluable_rewriteable_is_id(transpose(A'))
        test_evaluable_rewriteable_is_id(transpose(A)')

        B = [300 200 100]
        C = [1000]
        D = 10000
        test_evaluable_rewriteable_is_id(broadcasted(+, A, B, C, D))
        test_evaluable_rewriteable_is_id(broadcasted(*, A, B, C, D))
        test_evaluable_rewriteable_is_id(
            broadcasted(+,
                broadcasted(+, A, B),
                broadcasted(-, B, C),
                broadcasted(*, C, D),
                broadcasted(/, D, A),
                broadcasted(^, A, B),
            )
        )

        test_evaluable_rewriteable_is_id(Swizzle(+, 1)(A))
        test_evaluable_rewriteable_is_id(Swizzle(*, 1)(A) |> lift_vals)
    end

    @testset "rewriteable uses Antennae" begin
        A = [1 2 3; 4 5 6]
        B = [300 200 100]

        bd = broadcasted(+, A, B)
        @test rewriteable(:bd, typeof(bd))[1].ex.args[1] isa Swizzles.Antennae.Antenna
    end

    @testset "rewriteable uses ValArrays" begin
        A = [1 2 3; 4 5 6]
        @test rewriteable(
            :sw,
            Swizzle(*, 1)(A) |> lift_vals |> typeof
        )[1].ex.args[2] == ValArray(1)
    end

    @testset "rewriteable with NamedArrays" begin
        @generated function get_evaluable_rewriteable(val)
            expr = evaluable(rewriteable(:val, val)...)
            return :($expr)
        end

        A, B = [1 2 3; 4 5 6], [100 200 300]
        bc = broadcasted(+, A, B, A)
        lb = lift_names(bc)

        @test lb |> get_evaluable_rewriteable |> materialize == materialize(bc)
    end
end

@testset "Simplify" begin
    A, B = [1 2 3; 4 5 6], [100 200 300]
    @test Simplify().(A) == A
    @test Simplify().(A .+ B) == A .+ B
end
