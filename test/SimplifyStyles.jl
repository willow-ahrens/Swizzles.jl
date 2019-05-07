using Random

using Swizzles
using Swizzles.Properties
using Swizzles.SimplifyStyles
using Swizzles.ExtrudedArrays
using Swizzles.NamedArrays
using Swizzles.ValArrays
using Swizzles.SimplifyStyles: rewriteable, evaluable, veval, normalize, simplify_and_copy
using LinearAlgebra
using Base.Broadcast: broadcasted, materialize

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

@testset "Simplify and friends" begin
    # Helper function
    @generated function normalize_helper(arr)
        normal_term, syms = normalize(:arr, arr)
        normal_expr = evaluable(normal_term, syms)
        return :($normal_expr)
    end

    @testset "remove identity inits" begin
        A = [1 2 3]
        @test Swizzle(+)(0, A) |>
                lift_vals |>
                normalize_helper |>
                typeof == Swizzle(+)(A) |> typeof

        #= TODO: Fix identity checking to factor in casting.
        A = [1. 2. 3.]
        @test Swizzle(+)(0, A) |> lift_vals |> simplify |> typeof == Swizzle(+)(A) |> typeof
        =#
    end


    @testset "merge nested Swizzles" begin
        A = rand(MersenneTwister(0), 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2)

        @test Swizzle(+)(Swizzle(+)(A)) |>
                lift_vals |>
                normalize_helper == Swizzle(+)(A)

        @test Swizzle(+)(9001, Swizzle(+)(A)) |>
                lift_vals |>
                normalize_helper == Swizzle(+)(9001, A)

        s1 = Swizzle(+, 7, 100, 1, 2, 4, 3)
        s2 = Swizzle(+, 8, 9, 2, 1, 3, 6, 4)
        @test simplify_and_copy(s1(s2(A)) |> lift_vals, nothing) ≈ s1(s2(A)) |> copy
    end


    @testset "Simplify()" begin
        A, B = [1 2 3; 4 5 6], [100 200 300]
        @test Simplify().(A) == A
        @test Simplify().(A .+ B) == A .+ B

        C  = [100 200; 300 400; 500 600]
        D  = [-1 -2; -3 -4]
        D′ = copy(D)

        @test Simplify().(Beam(1,2).(D)) == D

        D .+= Simplify().(Beam(1,2).(D))
        D′ .+= Simplify().(Beam(1,2).(D′))
        @test D == D′

        D  = [-1 -2; -3 -4]
        D′ = copy(D)
        D  .+=              Swizzle(+, 1, 3).(Beam(1, 2).(A) .* Beam(2, 3).(C))
        D′ .+= Simplify().( Swizzle(+, 1, 3).(Beam(1, 2).(A) .* Beam(2, 3).(C)) )
        @test D == D′
    end

    @testset "veval()" begin
        for x in [@_((1, 2, 3) .+ [1, 2, 3]'),
                  @_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9]),
                  Swizzle(+)(@_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9])),
                  Swizzle(+, 1)(@_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9]))]
            @test kept.(keeps(veval(evaluable(rewriteable(:root, typeof(lift_keeps(x)))...)))) == kept.(keeps(x))
            @test eltype(veval(evaluable(rewriteable(:root, typeof(x))...))) == eltype(x)
        end
    end
end
