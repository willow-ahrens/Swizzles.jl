@testset "util" begin
    @testset "scalar getindexinto" begin
        @test getindexinto(-1, [2; 4; 0; 3; 1], 3) === 0
        @test getindexinto(-1, [2; 4; 0; 3; 1], drop) === -1
        @test getindexinto(-1.0, [2.0; 4.0; 0.0; 3.0; 1.0], 3) === 0.0
        @test getindexinto(-1.0, [2.0; 4.0; 0.0; 3.0; 1.0], drop) === -1.0
        @test getindexinto(-1, (2, 4, 0, 3, 1), 3) === 0
        @test getindexinto(-1, (2, 4, 0, 3, 1), drop) === -1
        @test getindexinto(-1.0, (2.0, 4.0, 0.0, 3.0, 1.0), 3) === 0.0
        @test getindexinto(-1.0, (2.0, 4.0, 0.0, 3.0, 1.0), drop) === -1.0

        @test_throws Exception getindexinto(-1, (), 3)
        @test getindexinto(-1, (), drop) === -1
        @test_throws Exception getindexinto(-1, [], 3)
        @test getindexinto(-1, [], drop) === -1
        @test_throws Exception getindexinto(-1, (2, 4, 0, 3, 1), -1)
        @test_throws Exception getindexinto(-1, [2; 4; 0; 3; 1], -1)
        @test_throws Exception getindexinto(-1, (2, 4, 0, 3, 1), (1, 2))
        @test_throws Exception getindexinto(-1, [2; 4; 0; 3; 1], (1, 2))
    end

    @testset "tensor getindexinto" begin
        function refgetindexinto(A, B, I)
            result = similar(I)
            for j in eachindex(I)
                push!(result, getindexinto(A[j], B, I[j]))
            end
            return copyto!(similar(I), result)
        end
        refgetindexinto(A, B, I::Tuple) = (refgetindexinto(A, B, [I...,])...,)

        As = [[],
              [-1],
              [-1.0],
              [-1; -2],
              [-1.0; -2.0]]
        Bs = [[],
              [11],
              [11.0],
              [11; 12],
              [11.0; 12.0],
              [11; 12; 13],
              [11.0; 12.0; 13.0]]
        Is = [[],
              [1],
              [-1],
              [drop],
              [1; 2],
              [drop; 2],
              [2; drop],
              [2; 2; 1],
              [-2; 2; 1],
              [1; 2; 4],
              [0; 2; 4],
              [drop; 2; 2],
              [drop; 2; 4],
              [-1; drop; 4],
              [1; drop; 3; 3]]
        As = vcat(As, [(A...,) for A in As])
        Bs = vcat(Bs, [(B...,) for B in Bs])
        Is = vcat(Is, [(I...,) for I in Is])
        for (A, B, I) in Iterators.product(As, Bs, Is)
            #println((A, B, I))
            try
                refgetindexinto(A, B, I)
            catch E
                #@test_throws typeof(E) getindexinto(A, B, I) #FIXME
                continue
            end
            @test getindexinto(A, B, I) == refgetindexinto(A, B, I)
            @test typeof(getindexinto(A, B, I)) == typeof(refgetindexinto(A, B, I))
        end
    end
end
