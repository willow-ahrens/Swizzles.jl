using Base.Broadcast: broadcasted, materialize
using Logging
using Random

using Swizzles
using Swizzles.Properties
using Swizzles.SimplifyStyles
using Swizzles.ExtrudedArrays
using Swizzles.NamedArrays
using Swizzles.ValArrays
using Swizzles.SimplifyStyles: rewriteable, evaluable, veval, normalize,
                               match_term, simplify_and_copy

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

    @testset "remove identity/nothing inits" begin
        A = [1 2 3]
        @test Swizzle(+)(0, A) |>
                lift_vals |>
                normalize_helper |>
                typeof == Swizzle(+)(A) |> typeof

        @test Swizzle(+)(nothing, A) |>
                lift_vals |>
                normalize_helper |>
                typeof == Swizzle(+)(A) |> typeof

        # TODO: Fix identity checking to factor in casting.
        A = [1. 2. 3.]
        @test_broken Swizzle(+)(0, A) |> lift_vals |> simplify |> typeof == Swizzle(+)(A) |> typeof
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

    @testset "matching gemm" begin
        function get_gemm_patterns()
            patterns = []
            for reduction_idx in 1:3
                nr_idxs = filter!(x -> x ≠ reduction_idx, Array(1:3)) # non-reduction indices
                for rpos1 in 1:2
                    for rpos2 in 1:2
                        for nr_ord in 1:2
                            nr1, nr2 = nr_idxs[nr_ord], nr_idxs[3 - nr_ord]

                            a_idxs = zeros(Int64, 2)
                            a_idxs[rpos1] = reduction_idx
                            a_idxs[3 - rpos1] = nr1

                            b_idxs = zeros(Int64, 2)
                            b_idxs[rpos2] = reduction_idx
                            b_idxs[3 - rpos2] = nr2

                            push!(patterns, ([nr1, nr2], a_idxs, b_idxs))
                            push!(patterns, ([nr2, nr1], a_idxs, b_idxs))
                        end
                    end
                end
            end
            return patterns
        end

        @testset "gemm(tA, tB, A, B) matcher" begin
            A, B = [1. 2.; 3. 4.], [200. 300.; 400. 700.]

            LOG_MSG = "matched gemm(tA, tB, A, B)"

            patterns_to_test = get_gemm_patterns()

            idx_maps_to_test = [
                (1, 2, 3),
                (3, 5, 8)
            ]

            for idx_map in idx_maps_to_test
                for pat in patterns_to_test
                    oidxs, aidxs, bidxs = map(p -> map(x -> idx_map[x], p), pat)

                    @test (
                        @test_logs(
                            (:debug, LOG_MSG),
                            min_level=Logging.Debug,
                            Simplify().(Swizzle(+, oidxs...).(Beam(aidxs...).(A) .* Beam(bidxs...).(B)))
                        )
                    ) ≈ Swizzle(+, oidxs...).(Beam(aidxs...).(A) .* Beam(bidxs...).(B))
                end
            end
        end

        @testset "gemm!(tA, tB, alpha, A, B, beta, C) matcher" begin
            A, B = [1. 2.; 3. 4.], [200. 300.; 400. 700.]
            C = [0. 0.; 0. 0.]

            LOG_MSG = "matched gemm!(tA, tB, alpha, A, B, beta, C)"

            patterns_to_test = get_gemm_patterns()

            idx_maps_to_test = [
                (1, 2, 3),
                (3, 5, 8)
            ]

            for idx_map in idx_maps_to_test
                for pat in patterns_to_test
                    oidxs, aidxs, bidxs = map(p -> map(x -> idx_map[x], p), pat)

                    @test (
                        @test_logs(
                            (:debug, LOG_MSG),
                            min_level=Logging.Debug,
                            C .= Simplify().(Swizzle(+, oidxs...).(Beam(aidxs...).(A) .* Beam(bidxs...).(B)))
                        ) ≈ Swizzle(+, oidxs...).(Beam(aidxs...).(A) .* Beam(bidxs...).(B))
                    )
                end
            end
        end
    end

    @testset "Simplify()" begin
        A, B = [1. 2. 3.; 4. 5. 6.], [100. 200. 300.]
        @test Simplify().(A) == A
        @test Simplify().(A .+ B) == A .+ B

        C  = [100. 200.; 300. 400.; 500. 600.]
        D  = [-1. -2.; -3. -4.]
        D′ = copy(D)

        @test Simplify().(Beam(1,2).(D)) == D

        D .+= Simplify().(Beam(1,2).(D))
        D′ .+= Simplify().(Beam(1,2).(D′))
        @test D == D′

        D  = [-1. -2.; -3. -4.]
        D′ = copy(D)
        D  +=              Swizzle(+, 1, 3).(Beam(1, 2).(A) .* Beam(2, 3).(C))
        D′ += Simplify().( Swizzle(+, 1, 3).(Beam(1, 2).(A) .* Beam(2, 3).(C)) )
        @test D == D′
    end

    @testset "veval()" begin
        for x in [@_((1, 2, 3) .+ [1, 2, 3]'),
                  @_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9]),
                  Swizzle(+)(@_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9])),
                  Swizzle(+, 1)(@_(0 .+ [1, 2, 3] .+ [1 2 3; 4 5 6; 7 8 9]))]
            # TODO: Fix interaction of arrayify to not call insantiate.
            @test_skip kept.(keeps(veval(evaluable(rewriteable(:root, typeof(lift_keeps(x)))...)))) == kept.(keeps(x))
            @test_skip eltype(veval(evaluable(rewriteable(:root, typeof(x))...))) == eltype(x)
        end
    end
end
